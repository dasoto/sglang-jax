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

import math
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
# Reduce-scatter MoE combine into the model mesh's P("data", "tensor") layout (see MXFP4EPMoE).
MOE_RS_COMBINE_ENV = "SGLANG_JAX_MOE_RS_COMBINE"


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
        # Large (prefill) batches take the local-window path (see _forward_local_window).
        self.local_window_min_rows = int(os.environ.get("SGLANG_JAX_MOE_LOCAL_WINDOW_MIN_ROWS", "4096"))
        self.local_window_capacity_factor = float(
            os.environ.get("SGLANG_JAX_MOE_LOCAL_WINDOW_CF", "4.0")
        )
        self.local_window_capacity = None  # fixed window rows (tests); None = derive from load
        self.supports_token_mask = True
        # Reduce-scatter combine (SGLANG_JAX_MOE_RS_COMBINE=1): instead of all-reducing the full
        # [T, H] output over "tensor" and then "expert" and replicating it everywhere, one
        # reduce-scatter over the whole (expert, tensor) mesh lands each chip's block of the
        # model mesh's P("data", "tensor") layout directly (its DP group's tokens x its slice of
        # H), which is what the row-parallel routed_expert_up_proj consumes.
        self.rs_combine = os.environ.get(MOE_RS_COMBINE_ENV, "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
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
        token_mask: jax.Array | None = None,
    ) -> jax.Array:
        """``token_mask`` [tokens] marks real tokens (DP-padding slots are False).

        Only the local-window (prefill) path uses it: masked tokens are routed to a
        sentinel expert id, so they are neither computed nor counted. With one busy
        DP rank, 7/8 of a step's tokens are padding whose identical rows all route to
        the same experts and would serialize a few expert shards.
        """
        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))
        out_specs = P(
            *[
                ("tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None)
                for s in out_sharding.spec
            ]
        )
        scatter_on_tensor = "tensor" in out_specs
        rows = topk_ids.shape[0] * topk_ids.shape[1]
        use_mask = (
            token_mask is not None
            and token_mask.shape[0] == topk_ids.shape[0]
            and self.gmm_kernel_weights_ready
            and rows >= self.local_window_min_rows
        )
        rs_blocks = self._rs_combine_blocks(hidden_states)
        if rs_blocks is not None:
            out_specs = P(("expert", "tensor"), None, None)

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))
            if use_mask:
                mask = jax.sharding.reshard(token_mask, P(None))
                topk_ids_reshard = jnp.where(
                    mask[:, None], topk_ids_reshard, jnp.asarray(self.num_experts, topk_ids_reshard.dtype)
                )
            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor, rs_blocks=rs_blocks),
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
        if rs_blocks is not None:
            return self._rs_combine_output(result, hidden_states.shape, rs_blocks)
        return jax.sharding.reshard(result, out_sharding)

    def _rs_combine_blocks(self, hidden_states) -> tuple[int, int] | None:
        """(dp, tp) of the model mesh when the reduce-scatter combine applies, else None.

        Needs a 2-D [tokens, H] input, a ("data", "tensor") model mesh covering the same devices
        as the (expert, tensor) MoE mesh, tokens divisible by dp and H divisible by tp.
        """
        if not self.rs_combine or self.replicate_experts or hidden_states.ndim != 2:
            return None
        shape = dict(self.mesh.shape)
        if set(shape) != {"data", "tensor"}:
            return None
        dp, tp = shape["data"], shape["tensor"]
        if dp * tp != self.ep_size * self.tp_size:
            return None
        t, h = hidden_states.shape
        if t % dp or h % tp:
            return None
        return dp, tp

    def _rs_reduce(self, output, rs_blocks):
        """Shard-local: [T, H] partial output -> this chip's [1, T/dp, H/tp] reduced block.

        Block (d, t) holds DP group d's tokens and H slice t, ordered like the flattened model
        mesh, whose device order the (expert, tensor) mesh shares (EPMoE builds it by reshaping
        the model mesh's device list), so a tiled reduce-scatter over both MoE axes hands
        block i to device i. Replaces the all-reduces over "tensor" and "expert".
        """
        dp, tp = rs_blocks
        t, h = output.shape
        blocks = output.reshape(dp, t // dp, tp, h // tp).transpose(0, 2, 1, 3)
        # [ep, moe_tp, T/dp, H/tp]: block (e, x) is device 2e + x's. Two single-axis reduce-scatters
        # (same total traffic as one over both axes): XLA rejects the replica groups of one
        # collective spanning both MoE axes, which are sub-axes of the model mesh's "tensor".
        blocks = blocks.reshape(self.ep_size, self.tp_size, t // dp, h // tp)
        if self.tp_size > 1:
            blocks = jax.lax.psum_scatter(blocks, "tensor", scatter_dimension=1, tiled=True)
        if self.ep_size > 1:
            blocks = jax.lax.psum_scatter(blocks, "expert", scatter_dimension=0, tiled=True)
        return blocks.reshape(1, t // dp, h // tp)

    def _rs_combine_output(self, result, shape, rs_blocks):
        """[dp*tp, T/dp, H/tp] blocks on the MoE mesh -> [T, H] with P("data", "tensor")."""
        dp, tp = rs_blocks
        t, h = shape
        blocks = jax.sharding.reshard(
            result, NamedSharding(self.mesh, P(("data", "tensor"), None, None))
        )
        blocks = jnp.reshape(
            blocks, (dp, tp, t // dp, h // tp), out_sharding=P("data", "tensor", None, None)
        )
        out = jnp.transpose(blocks, (0, 2, 1, 3))
        return jnp.reshape(out, (t, h), out_sharding=P("data", "tensor"))

    def _forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        *,
        scatter_on_tensor: bool = False,
        rs_blocks: tuple[int, int] | None = None,
    ):
        rows = topk_ids.shape[0] * topk_ids.shape[1]
        reduce_fn = partial(self._rs_reduce, rs_blocks=rs_blocks) if rs_blocks else None
        if not (self.gmm_kernel_weights_ready and rows >= self.local_window_min_rows):
            return super()._forward(
                hidden_states,
                topk_weights,
                topk_ids,
                w0_weights,
                w1_weights,
                wo_weights,
                w0_kernel_scale,
                w1_kernel_scale,
                wo_kernel_scale,
                w0_kernel_bias,
                w1_kernel_bias,
                wo_kernel_bias,
                scatter_on_tensor=scatter_on_tensor,
                reduce_fn=reduce_fn,
            )
        output = self._forward_local_window(
            hidden_states,
            topk_weights,
            topk_ids,
            w0_weights,
            w1_weights,
            wo_weights,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
        )
        if reduce_fn is not None:
            return reduce_fn(output)
        # Same reductions as EPMoE._forward.
        if self.tp_size > 1:
            if scatter_on_tensor:
                output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
            else:
                output = jax.lax.psum(output, "tensor")
        if self.ep_size > 1:
            output = self._combine(output)
        return output

    def _window_capacity(self, rows: int) -> int:
        """Static rows per window: capacity_factor x the expected local share, 256-aligned."""
        if self.local_window_capacity:
            return int(self.local_window_capacity)
        expected = rows * self.experts_per_device / self.num_experts
        cap = int(math.ceil(expected * self.local_window_capacity_factor / 256.0)) * 256
        return max(512, min(cap, -(-rows // 256) * 256))

    def _forward_local_window(
        self, hidden_states, topk_weights, topk_ids, w_gate, w_up, w_down, s_gate, s_up, s_down
    ):
        """Prefill path: compute and combine only the rows routed to this shard's experts.

        Tokens are replicated on every expert shard, and EPMoE's generic path does
        O(tokens * top_k) work per shard: it gathers every routed row, zeroes the
        non-local rows of the GMM output, and runs a top_k-pass fp32 unpermute over
        all of them, although only ~E_l/E of the rows are local. The local rows are
        a contiguous range of the expert-sorted order, so this path processes them
        in fixed-size windows (normally one; a loop covers routing imbalance) and
        scatter-adds the weighted outputs per token. The math is unchanged; only the
        f32 summation order differs.
        """
        from sgl_jax.srt.kernels.mxfp4_gmm import mxfp4_moe_mlp

        num_tokens, top_k = topk_ids.shape
        rows = num_tokens * top_k
        e_l = self.experts_per_device
        flat_experts = jnp.ravel(topk_ids).astype(jnp.int32)
        sorted_pos = jnp.argsort(flat_experts, stable=True).astype(jnp.int32)
        # Group sizes from the sorted ids by binary search: jnp.bincount lowers to a
        # scatter-add over all T*top_k rows, which cost ~0.8 ms per layer on v5e.
        # Ids >= num_experts (masked DP padding) sort last and are not counted.
        bounds = jnp.searchsorted(
            flat_experts[sorted_pos],
            jnp.arange(self.num_experts + 1, dtype=jnp.int32),
            side="left",
        ).astype(jnp.int32)
        group_sizes = bounds[1:] - bounds[:-1]
        group_offset = (jax.lax.axis_index("expert") * e_l).astype(jnp.int32)
        local_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, group_offset, e_l)
        start = jnp.sum(jnp.where(jnp.arange(self.num_experts) < group_offset, group_sizes, 0))
        local_rows = jnp.sum(local_sizes)
        local_cum = jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(local_sizes).astype(jnp.int32)]
        )
        flat_w = jnp.ravel(topk_weights).astype(jnp.float32)
        cap = self._window_capacity(rows)
        no_groups = jnp.zeros((self.num_experts,), jnp.int32)

        def window(w, acc):
            base = w * cap
            rel = base + jnp.arange(cap, dtype=jnp.int32)
            valid = rel < local_rows
            pos = sorted_pos[jnp.clip(start + rel, 0, rows - 1)]  # flat (token, k) slot
            tok = pos // top_k
            x = hidden_states[tok].astype(self.dtype)
            # Local experts' row counts inside this window; non-local groups stay empty,
            # so the local rows start at row 0 of x.
            sizes = jnp.clip(local_cum[1:] - base, 0, cap) - jnp.clip(local_cum[:-1] - base, 0, cap)
            win_group_sizes = jax.lax.dynamic_update_slice_in_dim(
                no_groups, sizes.astype(jnp.int32), group_offset, 0
            )
            y = mxfp4_moe_mlp(
                x,
                win_group_sizes,
                group_offset,
                w_gate,
                s_gate,
                w_up,
                s_up,
                w_down,
                s_down,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                act_input_dtype=self.dtype,
                out_dtype=self.dtype,
                zero_unvisited=False,  # rows past the local groups are masked below
                interpret=not is_tpu_runtime(),
            )
            contrib = jnp.where(valid[:, None], y.astype(jnp.float32) * flat_w[pos][:, None], 0.0)
            return acc.at[tok].add(contrib)

        n_windows = (local_rows + cap - 1) // cap
        acc = jnp.zeros((num_tokens, w_down.shape[-1]), jnp.float32)
        acc = jax.lax.fori_loop(0, n_windows, window, acc)
        return acc.astype(self.dtype)

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
