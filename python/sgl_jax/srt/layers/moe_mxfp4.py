"""Expert-parallel MoE over packed MXFP4 (compressed-tensors) expert weights.

Kimi-K3 ships its routed experts as compressed-tensors ``mxfp4-pack-quantized``
(see ``layers/quantization/mxfp4.py`` for the bit format). The packed bytes and
E8M0 scales stay on device unchanged; at load time each expert's
``[out, in // 2]`` / ``[out, in // 32]`` tensors are only transposed to the
k-major orientation ``[in // 2, out]`` / ``[in // 32, out]`` and stacked over
experts, which matches ``EPMoE``'s ``[E, k, n]`` weight convention and sharding
on its ``(expert, tensor)`` sub-mesh:

* gate/up (``w1``/``w3``): ``[E, H // 2, I]``, ``P("expert", None, "tensor")``
* down (``w2``): ``[E, I // 2, H]``, ``P("expert", "tensor", None)``

(``H`` = expert input size, the LatentMoE latent for K3; ``I`` = expert
intermediate size, split over expert-TP.) Each local shard is dequantized
inside the MoE ``shard_map`` right before the grouped matmul, so bf16 expert
weights only exist transiently for one layer's local experts.

PERF TODO: replace the XLA dequant + bf16 GMM with an MXFP4 megablox GMM that
dequantizes weight tiles in VMEM (design doc section 4), and fuse SiTU-GLU.
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.quantization.mxfp4 import MXFP4_GROUP_SIZE, dequant_mxfp4
from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.profiling_utils import named_scope

# Fused MXFP4 grouped-matmul Pallas kernel (kernels/mxfp4_gmm): opt-in until validated on TPU.
MXFP4_GMM_KERNEL_ENV = "SGLANG_JAX_MXFP4_GMM_KERNEL"


def mxfp4_gmm_kernel_enabled() -> bool:
    return os.environ.get(MXFP4_GMM_KERNEL_ENV, "0").strip().lower() in ("1", "true", "yes", "on")

# k-major MXFP4 layouts on the (expert, tensor) mesh.
_WI_SPEC = P("expert", None, "tensor")  # gate/up: [E, H//2 | H//32, I]
_WO_SPEC = P("expert", "tensor", None)  # down:    [E, I//2 | I//32, H]


class MXFP4EPMoE(EPMoE):
    """``EPMoE`` whose experts are packed MXFP4, dequantized per forward.

    Parameters (stacked over experts, k-major):

    * ``w1_weight_packed`` / ``w3_weight_packed`` (gate / up): uint8
      ``[E, H // 2, I]``; ``w1_weight_scale`` / ``w3_weight_scale``: uint8
      ``[E, H // 32, I]``.
    * ``w2_weight_packed`` (down): uint8 ``[E, I // 2, H]``; ``w2_weight_scale``:
      uint8 ``[E, I // 32, H]``.

    With expert-TP size ``t`` each shard holds ``I / t`` intermediate rows of
    ``w2``, which must be a multiple of 32 so scale groups never straddle
    shards.
    """

    def __init__(self, *args, use_gmm_kernel: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Fused dequant+GMM kernel: weights are re-encoded once by
        # ``prepare_gmm_kernel_weights()`` (called from the model's ``load_weights``).
        self.use_gmm_kernel = mxfp4_gmm_kernel_enabled() if use_gmm_kernel is None else use_gmm_kernel
        self.gmm_kernel_weights_ready = False
        if self.replicate_experts:
            raise NotImplementedError("MXFP4EPMoE does not support moe_dp_size > 1")
        if self.quantized_dtype is not None or self.activation_quantized_dtype is not None:
            raise ValueError("MXFP4EPMoE experts are statically packed; no online quantization")
        for name, dim in (
            ("hidden_size", self.hidden_size),
            ("intermediate_dim", self.intermediate_dim),
        ):
            if dim % MXFP4_GROUP_SIZE:
                raise ValueError(f"MXFP4 {name}={dim} must be a multiple of {MXFP4_GROUP_SIZE}")
        if (self.intermediate_dim // self.tp_size) % MXFP4_GROUP_SIZE:
            raise ValueError(
                f"MXFP4 expert intermediate {self.intermediate_dim} split over expert-TP "
                f"{self.tp_size} must leave a multiple of {MXFP4_GROUP_SIZE} per shard"
            )

    def _init_expert_weights(self, wi_sharding: P, wo_sharding: P) -> None:
        del wi_sharding, wo_sharding
        e, h, i, g = self.num_experts, self.hidden_size, self.intermediate_dim, MXFP4_GROUP_SIZE

        def _param(shape, sharding):
            return nnx.Param(jnp.zeros(shape, dtype=jnp.uint8, out_sharding=sharding))

        self.w1_weight_packed = _param((e, h // 2, i), _WI_SPEC)
        self.w1_weight_scale = _param((e, h // g, i), _WI_SPEC)
        self.w3_weight_packed = _param((e, h // 2, i), _WI_SPEC)
        self.w3_weight_scale = _param((e, h // g, i), _WI_SPEC)
        self.w2_weight_packed = _param((e, i // 2, h), _WO_SPEC)
        self.w2_weight_scale = _param((e, i // g, h), _WO_SPEC)
        self.wi_0_scale = None
        self.wi_1_scale = None
        self.wo_scale = None

    def quantize_weights(self, is_static: bool = False, *, abstract: bool = False):
        # Packed MXFP4 experts are loaded as-is; there is nothing to quantize.
        return

    def gmm_kernel_supported(self) -> bool:
        """The kernel decodes 64-row blocks of K: H and the per-shard I must be multiples of 64."""
        return self.hidden_size % 64 == 0 and (self.intermediate_dim // self.tp_size) % 64 == 0

    def prepare_gmm_kernel_weights(self) -> bool:
        """Re-encode the loaded packed experts in place for the fused MXFP4 GMM kernel.

        ``uint8 [E, K/2, N]`` -> ``uint32 [E, K/8, N]`` (same bytes, same PartitionSpecs), done
        shard-locally. Scales are unchanged. Returns whether the kernel path is active.
        """
        if self.gmm_kernel_weights_ready:
            return True
        if not self.use_gmm_kernel or not self.gmm_kernel_supported():
            return False
        from sgl_jax.srt.kernels.mxfp4_gmm import prepare_l24

        with jax.set_mesh(self.moe_mesh):
            for name, spec in (
                ("w1_weight_packed", _WI_SPEC),
                ("w3_weight_packed", _WI_SPEC),
                ("w2_weight_packed", _WO_SPEC),
            ):
                param = getattr(self, name)
                packed = param.get_value()
                if packed.dtype != jnp.uint8:
                    raise ValueError(f"{name} is {packed.dtype}, expected packed uint8")
                convert = jax.jit(
                    shard_map(prepare_l24, mesh=self.moe_mesh, in_specs=spec, out_specs=spec, check_vma=False),
                    out_shardings=NamedSharding(self.moe_mesh, spec),
                )
                param.set_value(convert(packed))
        self.gmm_kernel_weights_ready = True
        return True

    @named_scope
    def __call__(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))
        out_specs = P(
            *[
                ("tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None)
                for s in out_sharding.spec
            ]
        )
        scatter_on_tensor = "tensor" in out_specs

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))
            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor),
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    # packed weights (gate, up, down)
                    _WI_SPEC,
                    _WI_SPEC,
                    _WO_SPEC,
                    # E8M0 scales (gate, up, down)
                    _WI_SPEC,
                    _WI_SPEC,
                    _WO_SPEC,
                    # biases (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", None, None),
                ),
                out_specs=out_specs,
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.w1_weight_packed.value,
                self.w3_weight_packed.value,
                self.w2_weight_packed.value,
                self.w1_weight_scale.value,
                self.w3_weight_scale.value,
                self.w2_weight_scale.value,
                None,
                None,
                None,
            )
        return jax.sharding.reshard(result, out_sharding)

    def _gmm_compute(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        group_offset,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        local_range=None,
    ):
        if self.gmm_kernel_weights_ready:
            return self._gmm_compute_kernel(
                inputs_2d,
                token_indices,
                group_sizes,
                w0_kernel,
                w1_kernel,
                wo_kernel,
                group_offset,
                w0_kernel_scale,
                w1_kernel_scale,
                wo_kernel_scale,
            )
        # Local k-major shards decode straight into GMM [E_l, k, n] operands.
        # The barrier ties the packed weights to this layer's activations so XLA
        # cannot hoist (and keep live) the bf16 dequant of every layer up front.
        packed = (
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
        )
        *packed, inputs_2d = jax.lax.optimization_barrier((*packed, inputs_2d))
        w0_kernel, w1_kernel, wo_kernel, w0_kernel_scale, w1_kernel_scale, wo_kernel_scale = packed
        dequant = partial(dequant_mxfp4, axis=-2, out_dtype=self.dtype)
        return super()._gmm_compute(
            inputs_2d,
            token_indices,
            group_sizes,
            dequant(w0_kernel, w0_kernel_scale),
            dequant(w1_kernel, w1_kernel_scale),
            dequant(wo_kernel, wo_kernel_scale),
            group_offset,
            local_range=local_range,
        )

    def _gmm_compute_kernel(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w_gate,
        w_up,
        w_down,
        group_offset,
        s_gate,
        s_up,
        s_down,
    ):
        """Fused path: l24 weights decoded in VMEM, SiTU-GLU fused into the gate/up GMM."""
        from sgl_jax.srt.kernels.mxfp4_gmm import mxfp4_moe_mlp

        if token_indices.shape[0] == 0:
            return jnp.zeros((0, w_down.shape[-1]), dtype=inputs_2d.dtype)
        x = inputs_2d[token_indices].astype(self.dtype)
        return mxfp4_moe_mlp(
            x,
            group_sizes.astype(jnp.int32),
            group_offset,
            w_gate,
            s_gate,
            w_up,
            s_up,
            w_down,
            s_down,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            act_input_dtype=self.dtype,  # EPMoE's GMMs emit self.dtype before the activation
            out_dtype=self.dtype,
            interpret=not is_tpu_runtime(),
        )


def create_mxfp4_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    moe_path: str = "experts",
    expert_type_names: tuple[str, str, str] = ("w1", "w3", "w2"),
    source_expert_pattern: str = "experts.{i}",
) -> dict:
    """Map per-expert compressed-tensors MXFP4 tensors onto ``MXFP4EPMoE``.

    For each of gate/up/down (``expert_type_names``) the ``num_experts``
    checkpoint tensors ``{prefix}.{source_expert_pattern}.{name}.weight_packed``
    (and ``.weight_scale``) are transposed to k-major and stacked into
    ``{target_prefix}.{moe_path}.{w1|w3|w2}_weight_packed`` (``_weight_scale``).
    The loader reads only each host's addressable expert / intermediate slices.
    """
    gate, up, down = expert_type_names
    layouts = {gate: ("w1", _WI_SPEC), up: ("w3", _WI_SPEC), down: ("w2", _WO_SPEC)}
    mappings = {}
    for source_name, (target_name, spec) in layouts.items():
        for suffix in ("weight_packed", "weight_scale"):
            target = f"{target_prefix}.{moe_path}.{target_name}_{suffix}"
            mappings[target] = WeightSpec(
                target_path=target,
                sources=tuple(
                    f"{prefix}.{source_expert_pattern.format(i=i)}.{source_name}.{suffix}"
                    for i in range(num_experts)
                ),
                sharding=tuple(spec),
                transpose=True,
            )
    return mappings
