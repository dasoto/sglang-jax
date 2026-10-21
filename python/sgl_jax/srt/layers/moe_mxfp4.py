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

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import shard_map
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.layers.quantization.mxfp4 import MXFP4_GROUP_SIZE, dequant_mxfp4
from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.utils.profiling_utils import named_scope

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        # Local k-major shards decode straight into GMM [E_l, k, n] operands.
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
