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
# Compact prefill MoE (see MXFP4EPMoE._compact_forward): comma-separated divisors d of the
# step's padded token count T; each T // d >= SGLANG_JAX_MOE_COMPACT_MIN_TOKENS becomes a
# static compact size. "" or "0" disables it.
MOE_COMPACT_ENV = "SGLANG_JAX_MOE_COMPACT"
MOE_COMPACT_MIN_TOKENS_ENV = "SGLANG_JAX_MOE_COMPACT_MIN_TOKENS"
# Two-stage combine for the emit_sumsq paths (SGLANG_JAX_MOE_HIER_COMBINE=1): a reduce-scatter
# over "tensor" inside each DP group, then a reduction over "data" of the 1/tp-wide slices,
# instead of one all-reduce of the full [T, H] partials over the whole model mesh.
MOE_HIER_COMBINE_ENV = "SGLANG_JAX_MOE_HIER_COMBINE"
# Local-window row sort: (local expert, row) packed into one int32 key needs rows < 2^16.
_PACKED_SORT_MAX_ROWS = 1 << 16


# Local-window combine of the weighted expert rows into the per-chip [T, H] accumulator (see
# MXFP4EPMoE._forward_local_window): "scatter" (scatter-add, default), "onehot" (a [T, cap] one-hot
# matmul on the MXU with f32 accumulation, for T <= SGLANG_JAX_MOE_ONEHOT_MAX_TOKENS, else scatter)
# or "sorted" (rows sorted by token, masked rows dropped, sorted-index scatter-add), or "gather"
# (rows sorted by token; each token gathers its first few rows, the rare extra rows go through a
# small scatter; see _segment_gather_sum).
# Above the one-hot limit, "onehot" falls back to SGLANG_JAX_MOE_LOCAL_COMBINE_LARGE ("scatter" or
# "sorted", default "scatter").
MOE_LOCAL_COMBINE_ENV = "SGLANG_JAX_MOE_LOCAL_COMBINE"
MOE_LOCAL_COMBINE_LARGE_ENV = "SGLANG_JAX_MOE_LOCAL_COMBINE_LARGE"
MOE_ONEHOT_MAX_TOKENS_ENV = "SGLANG_JAX_MOE_ONEHOT_MAX_TOKENS"


def _local_combine_mode(
    env: str = MOE_LOCAL_COMBINE_ENV, allowed=("scatter", "onehot", "sorted", "gather")
) -> str:
    mode = os.environ.get(env, "scatter").strip().lower()
    return mode if mode in allowed else "scatter"


# Rows per token that the gather combine sums directly; tokens with more local rows send the
# extra ones through a scatter of at most _GATHER_OVERFLOW_ROWS rows (a full scatter beyond). A
# scatter costs ~0.13 us per update row on v5e whether or not the row is masked, so the small one
# is kept small; with 3 rows per token the overflow is a handful of rows per chip and layer.
_GATHER_ROWS_PER_TOKEN = 3
_GATHER_OVERFLOW_ROWS = 64


def _segment_gather_sum(rows, tok, valid, num_tokens):
    """``zeros([num_tokens, H]).at[tok[valid]].add(rows[valid])`` without a large scatter.

    TPU scatter-adds serialize on their update rows (~0.13 us per 3,584-wide bf16 row on v5e,
    with or without unique indices), while row gathers run ~10x faster. Rows are sorted by
    token; each token gathers its first ``_GATHER_ROWS_PER_TOKEN`` rows (searchsorted gives its
    segment) and sums them in f32. Rows beyond that (a token routed to several local experts)
    are compacted and scatter-added: a small scatter, or the full one when there are more
    than ``_GATHER_OVERFLOW_ROWS``. Exact up to f32 summation order.
    """
    cap, h = rows.shape
    key = jnp.where(valid, tok, num_tokens).astype(jnp.int32)  # masked rows sort last
    key_s, order = jax.lax.sort_key_val(key, jnp.arange(cap, dtype=jnp.int32))
    rows_s = rows[order]
    tokens = jnp.arange(num_tokens, dtype=jnp.int32)
    first = jnp.searchsorted(key_s, tokens, side="left", method="scan_unrolled").astype(jnp.int32)
    last = jnp.searchsorted(key_s, tokens, side="right", method="scan_unrolled").astype(jnp.int32)
    count = last - first
    out = jnp.zeros((num_tokens, h), jnp.float32)
    for j in range(_GATHER_ROWS_PER_TOKEN):
        g = rows_s[jnp.clip(first + j, 0, cap - 1)].astype(jnp.float32)
        out = out + jnp.where((count > j)[:, None], g, 0.0)
    # Rows ranked >= _GATHER_ROWS_PER_TOKEN within their token's segment.
    seg_first = first[jnp.clip(key_s, 0, num_tokens - 1)]
    rank = jnp.arange(cap, dtype=jnp.int32) - seg_first
    extra = (key_s < num_tokens) & (rank >= _GATHER_ROWS_PER_TOKEN)
    n_extra = jnp.sum(extra.astype(jnp.int32))
    ob = min(_GATHER_OVERFLOW_ROWS, cap)

    def small(out):
        (pos,) = jnp.nonzero(extra, size=ob, fill_value=cap - 1)
        ok = jnp.arange(ob) < n_extra
        upd = jnp.where(ok[:, None], rows_s[pos].astype(jnp.float32), 0.0)
        return out.at[jnp.where(ok, key_s[pos], 0)].add(upd)

    def full(out):
        upd = jnp.where(extra[:, None], rows_s.astype(jnp.float32), 0.0)
        return out.at[jnp.clip(key_s, 0, num_tokens - 1)].add(upd)

    return jax.lax.cond(n_extra > ob, full, small, out)


def _parse_compact_divisors(value: str) -> tuple[int, ...]:
    value = value.strip().lower()
    if value in ("", "0", "off", "false", "no"):
        return ()
    return tuple(sorted({int(x) for x in value.split(",") if x.strip() and int(x) > 1}))


def mxfp4_gmm_kernel_enabled() -> bool:
    return os.environ.get(MXFP4_GMM_KERNEL_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _spec_axes(spec, ndim: int) -> tuple[tuple[str, ...], ...]:
    """PartitionSpec -> per-dimension tuples of mesh axes, padded to ``ndim``."""
    entries = list(spec) + [None] * (ndim - len(spec))
    return tuple(
        () if e is None else ((e,) if isinstance(e, str) else tuple(e)) for e in entries[:ndim]
    )


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
        # Per-chip accumulator of the local-window path. f32 by default; with
        # SGLANG_JAX_MOE_BF16_ACC=1 it is the activation dtype: a token rarely gets more than
        # one contribution from one chip's experts, the output is cast to bf16 before the
        # cross-chip combine anyway, and the f32 [T, H] buffer's zero fill, scatter-add and
        # cast cost ~45 ms per 4096-token prefill step on v5e-256.
        bf16_acc = os.environ.get("SGLANG_JAX_MOE_BF16_ACC", "0").strip().lower()
        self.local_acc_dtype = (
            self.dtype if bf16_acc in ("1", "true", "yes", "on") else jnp.float32
        )
        self.supports_token_mask = True
        # Reduce-scatter combine (SGLANG_JAX_MOE_RS_COMBINE=1): instead of all-reducing the full
        # [T, H] output over "tensor" and then "expert" and replicating it everywhere, one
        # reduce-scatter over the whole (expert, tensor) mesh lands each chip's block of the
        # model mesh's P("data", "tensor") layout directly (its DP group's tokens x its slice of
        # H), which is what the row-parallel routed_expert_up_proj consumes.
        # SGLANG_JAX_MOE_RS_COMBINE=2 also runs that reduce-scatter, and the token all-gather
        # that feeds the experts, as collectives over the model mesh's own axes (see
        # _rs_combine_output and _mesh_gather_tokens), which v5e runs faster than collectives
        # over the MoE mesh's strided "expert" axis.
        rs_mode = os.environ.get(MOE_RS_COMBINE_ENV, "0").strip().lower()
        self.rs_combine = rs_mode in ("1", "true", "yes", "on", "2", "mesh")
        self.rs_combine_mesh = rs_mode in ("2", "mesh")
        # Token all-gather: one collective over both model-mesh axes for small inputs; for
        # large ones "data" then "tensor", which measured faster on v5e-256 at 29 MB.
        self.mesh_gather_two_step_bytes = 8 << 20
        # Compact prefill (mesh mode only): with few busy DP groups most of a prefill step's
        # tokens are padding, yet the token gather, the scatter-add and the combine all move
        # the padded [T, H]. A switch on the real-token count runs the experts on the smallest
        # static compact size that holds them (see _compact_forward).
        self.compact_divisors = _parse_compact_divisors(os.environ.get(MOE_COMPACT_ENV, ""))
        self.compact_min_tokens = int(os.environ.get(MOE_COMPACT_MIN_TOKENS_ENV, "512"))
        # Row sums of squares from the combine (mesh mode; set by the model when an RMSNorm
        # follows the experts, e.g. Kimi-K3's routed_expert_norm): XLA runs the combine as an
        # all-reduce of every chip's whole [dp*tp, ., H/(dp*tp) or H/tp] block array, so each chip
        # already holds its DP group's full output rows; their f32 sums of squares come out next
        # to the output ([T] P("data")) and the norm needs no all-reduce of its own. The compact
        # path also takes its tensor slice from that all-reduce instead of an extra all-gather.
        self.emit_sumsq = False
        # With emit_sumsq, reduce the partials in two stages (see MOE_HIER_COMBINE_ENV): each
        # chip receives 1/tp of [T, H] from its group instead of all of it from every chip.
        self.hier_combine = os.environ.get(MOE_HIER_COMBINE_ENV, "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.local_combine = _local_combine_mode()
        self.local_combine_large = _local_combine_mode(
            MOE_LOCAL_COMBINE_LARGE_ENV, ("scatter", "sorted", "gather")
        )
        self.onehot_max_tokens = int(os.environ.get(MOE_ONEHOT_MAX_TOKENS_ENV, "2048"))
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
        compact_pos: jax.Array | None = None,
    ) -> jax.Array:
        """``token_mask`` [tokens] marks real tokens (DP-padding slots are False).

        Only the local-window (prefill) path uses it: masked tokens are routed to a
        sentinel expert id, so they are neither computed nor counted. With one busy
        DP rank, 7/8 of a step's tokens are padding whose identical rows all route to
        the same experts and would serialize a few expert shards. ``compact_pos`` is
        ``compact_positions(token_mask)`` precomputed once per step for the compact path
        (computed here when None).

        With ``emit_sumsq`` the result is ``(output, sumsq)``: ``sumsq`` [tokens] f32 holds
        each output row's sum of squares over H (P("data") in mesh mode).
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
        compact_sizes = self._compact_sizes(hidden_states, topk_ids, rs_blocks) if use_mask else ()
        if compact_sizes and compact_pos is None:
            compact_pos = self.compact_positions(token_mask)
        if compact_sizes and compact_pos is not None:
            # Smallest static compact size >= the step's real-token count, else the full path.
            n_real = jnp.sum(token_mask.astype(jnp.int32))
            index = jnp.sum((n_real > jnp.asarray(compact_sizes, jnp.int32)).astype(jnp.int32))
            branches = [
                partial(self._compact_forward, rs_blocks=rs_blocks, c=c) for c in compact_sizes
            ]
            full = partial(
                self._call_full,
                out_sharding=out_sharding,
                out_specs=out_specs,
                scatter_on_tensor=scatter_on_tensor,
                use_mask=True,
                rs_blocks=rs_blocks,
            )
            branches.append(
                lambda hs, tw, ti, n, p: full(hs, tw, ti, token_mask)  # noqa: ARG005
            )
            return jax.lax.switch(
                index, branches, hidden_states, topk_weights, topk_ids, n_real, compact_pos
            )
        return self._call_full(
            hidden_states,
            topk_weights,
            topk_ids,
            token_mask,
            out_sharding=out_sharding,
            out_specs=out_specs,
            scatter_on_tensor=scatter_on_tensor,
            use_mask=use_mask,
            rs_blocks=rs_blocks,
        )

    def _call_full(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        token_mask,
        *,
        out_sharding,
        out_specs,
        scatter_on_tensor: bool,
        use_mask: bool,
        rs_blocks: tuple[int, int] | None,
    ) -> jax.Array:
        """Every (padded) token goes through the token gather, the experts and the combine."""
        if rs_blocks is not None:
            # Mesh mode emits each chip's unreduced [dp*tp, T/dp, H/tp] blocks (reduced after).
            if self.rs_combine_mesh and self.emit_sumsq:
                out_specs = P(("expert", "tensor"), None, None)  # unreduced [1, T, H] per chip
            else:
                out_specs = P(("expert", "tensor"), None, None, *([None] if self.rs_combine_mesh else []))
            if self.rs_combine_mesh:
                hidden_states, topk_weights, topk_ids = self._mesh_gather_tokens(
                    hidden_states, rs_blocks, topk_weights, topk_ids
                )

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
        result = jax.sharding.reshard(result, out_sharding)
        if self.emit_sumsq:  # no combine to piggyback on: a plain reduction
            r32 = result.astype(jnp.float32)
            return result, jnp.sum(r32 * r32, axis=-1)
        return result

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
        if self.rs_combine_mesh and self.emit_sumsq:
            # Unreduced, as computed: _rs_combine_output all-reduces it and slices this chip's
            # block (no [T, H] -> block-order transpose, ~0.13 ms per layer at T = 4096).
            return output[None]
        blocks = output.reshape(dp, t // dp, tp, h // tp).transpose(0, 2, 1, 3)
        if self.rs_combine_mesh:
            # Unreduced; _rs_combine_output reduce-scatters it over the model mesh.
            return blocks.reshape(1, dp * tp, t // dp, h // tp)
        # [ep, moe_tp, T/dp, H/tp]: block (e, x) is device 2e + x's. Two single-axis reduce-scatters
        # (same total traffic as one over both axes): XLA rejects the replica groups of one
        # collective spanning both MoE axes, which are sub-axes of the model mesh's "tensor".
        # "expert" first: XLA lowers the 2-way "tensor" reduce-scatter as an all-reduce of its
        # whole input, which is then only [moe_tp, T/dp, H/tp] instead of all of [T, H].
        blocks = blocks.reshape(self.ep_size, self.tp_size, t // dp, h // tp)
        if self.ep_size > 1:
            blocks = jax.lax.psum_scatter(blocks, "expert", scatter_dimension=0, tiled=True)
        if self.tp_size > 1:
            blocks = jax.lax.psum_scatter(blocks, "tensor", scatter_dimension=1, tiled=True)
        return blocks.reshape(1, t // dp, h // tp)

    def _rs_combine_output(self, result, shape, rs_blocks):
        """MoE-mesh shard_map output -> [T, H] with P("data", "tensor").

        ``result`` is [dp*tp, T/dp, H/tp] reduced blocks, or in mesh mode every chip's
        unreduced [1, dp*tp, T/dp, H/tp] blocks stacked to [dp*tp, dp*tp, T/dp, H/tp] (with
        ``emit_sumsq``, every chip's unreduced [1, T, H] stacked to [dp*tp, T, H]).
        """
        dp, tp = rs_blocks
        t, h = shape
        if self.rs_combine_mesh:
            # Same devices in the same order on both meshes: this reshard moves no data.
            spec = (None, None) if self.emit_sumsq else (None, None, None)
            stacked = jax.sharding.reshard(
                result, NamedSharding(self.mesh, P(("data", "tensor"), *spec))
            )

            def reduce_blocks(x):  # [1, dp*tp, T/dp, H/tp] partial -> [T/dp, H/tp] reduced
                # One reduce-scatter over both model-mesh axes (block i -> device i): 0.57 ms
                # vs 0.88 ms for v2's expert + tensor pair at T=4096 on v5e-256.
                y = jax.lax.psum_scatter(x, ("data", "tensor"), scatter_dimension=1, tiled=True)
                return y.reshape(t // dp, h // tp)

            def reduce_rows_sumsq(x):  # [1, T, H] partial -> (this chip's block, row sums of squares)
                # XLA lowers the reduce-scatter above as an all-reduce plus a slice anyway
                # (v5e-256, AOT and profiles); the explicit all-reduce of the [T, H] partials
                # needs no block-order transpose and leaves this DP group's full rows for the
                # sums of squares.
                y = jax.lax.psum(x[0], ("data", "tensor"))  # [T, H]
                rows = jax.lax.dynamic_slice_in_dim(y, jax.lax.axis_index("data") * (t // dp), t // dp, 0)
                mine = jax.lax.dynamic_slice_in_dim(
                    rows, jax.lax.axis_index("tensor") * (h // tp), h // tp, 1
                )
                r32 = rows.astype(jnp.float32)
                return mine, jnp.sum(r32 * r32, axis=1)

            def reduce_rows_sumsq_hier(x):  # as reduce_rows_sumsq, in two stages
                # A real reduce-scatter inside each DP group ([T, H] -> [T, H/tp]), then the
                # small cross-group reduction of the slices (XLA runs it as an all-reduce plus
                # a slice); the norm's row sums of squares take one f32 all-reduce of [T/dp].
                y = jax.lax.psum_scatter(x[0], "tensor", scatter_dimension=1, tiled=True)
                y = jax.lax.psum_scatter(y, "data", scatter_dimension=0, tiled=True)
                y32 = y.astype(jnp.float32)
                return y, jax.lax.psum(jnp.sum(y32 * y32, axis=1), "tensor")

            if self.emit_sumsq:
                return shard_map(
                    reduce_rows_sumsq_hier if self.hier_combine else reduce_rows_sumsq,
                    mesh=self.mesh,
                    in_specs=P(("data", "tensor"), None, None),
                    out_specs=(P("data", "tensor"), P("data")),
                    check_vma=False,
                )(stacked)
            return shard_map(
                reduce_blocks,
                mesh=self.mesh,
                in_specs=P(("data", "tensor"), None, None, None),
                out_specs=P("data", "tensor"),
                check_vma=False,
            )(stacked)
        blocks = jax.sharding.reshard(
            result, NamedSharding(self.mesh, P(("data", "tensor"), None, None))
        )
        blocks = jnp.reshape(
            blocks, (dp, tp, t // dp, h // tp), out_sharding=P("data", "tensor", None, None)
        )
        out = jnp.transpose(blocks, (0, 2, 1, 3))
        out = jnp.reshape(out, (t, h), out_sharding=P("data", "tensor"))
        if self.emit_sumsq:  # not mesh mode: a plain (all-reduced) reduction
            o32 = out.astype(jnp.float32)
            return out, jnp.sum(o32 * o32, axis=-1)
        return out

    def _mesh_gather_tokens(self, hidden_states, rs_blocks, topk_weights=None, topk_ids=None):
        """[T, H] with P("data", "tensor") -> replicated, by explicit model-mesh all-gathers.

        Replaces the implicit reshard to replicated (0.72 ms at T=4096 on v5e-256) with
        "data" then "tensor" for large inputs (0.52 ms) or one gather over both axes plus
        a local transpose for small ones (0.060 vs 0.077 ms at T=256). Other input
        shardings are returned unchanged (the caller's reshard handles them).

        When the router outputs (P("data", None), f32 weights and int32 ids) are passed and
        the hidden states are 16-bit, they ride in the same collective: their 32-bit words
        are split into 16-bit halves, each "tensor" shard sends its slice of those columns
        next to its hidden-state block, and they are bit-cast back after the gather. Decode
        steps are latency-bound, so this saves the two separate all-gathers per MoE layer.
        Only inputs below the two-step size (decode-sized batches) are packed: for
        prefill-sized ones the pack and unpack copies cost more than the saved gathers
        (v5e-256: +12 ms per 8 x 512-token prefill step), so the router outputs are returned
        unchanged for the caller's reshard.
        Returns ``(hidden_states, topk_weights, topk_ids)``.
        """
        if _spec_axes(jax.typeof(hidden_states).sharding.spec, 2) != (("data",), ("tensor",)):
            return hidden_states, topk_weights, topk_ids
        dp, tp = rs_blocks
        t, h = hidden_states.shape
        two_step = t * h * hidden_states.dtype.itemsize >= self.mesh_gather_two_step_bytes
        pack = (
            not two_step
            and topk_weights is not None
            and topk_ids is not None
            and hidden_states.dtype.itemsize == 2
            and topk_weights.dtype == jnp.float32
            and topk_ids.dtype == jnp.int32
            and topk_weights.shape == topk_ids.shape == (t, topk_ids.shape[1])
            and (4 * topk_ids.shape[1]) % tp == 0
            and all(
                _spec_axes(jax.typeof(a).sharding.spec, 2) == (("data",), ())
                for a in (topk_weights, topk_ids)
            )
        )
        hdt = hidden_states.dtype
        k = topk_ids.shape[1] if pack else 0
        c = 4 * k // tp  # 16-bit routing columns per "tensor" shard

        def all_gather(x):  # [T/dp, w] -> [T, tp, w] (tensor-shard blocks along axis 1)
            if two_step:
                y = jax.lax.all_gather(x, "data", axis=0, tiled=True)
                y = jax.lax.all_gather(y, "tensor", axis=1, tiled=True)
                return y.reshape(t, tp, x.shape[1])
            y = jax.lax.all_gather(x[None], ("data", "tensor"), axis=0, tiled=True)
            return y.reshape(dp, tp, t // dp, x.shape[1]).transpose(0, 2, 1, 3).reshape(
                t, tp, x.shape[1]
            )

        def gather(x):  # [T/dp, H/tp] -> [T, H]
            return all_gather(x).reshape(t, h)

        def gather_packed(x, wts, ids):
            words = jnp.concatenate(
                [
                    jax.lax.bitcast_convert_type(wts, jnp.uint32),
                    jax.lax.bitcast_convert_type(ids, jnp.uint32),
                ],
                axis=1,
            )  # [T/dp, 2k]
            halves = jax.lax.bitcast_convert_type(words, jnp.uint16).reshape(t // dp, 4 * k)
            mine = jax.lax.dynamic_slice_in_dim(halves, jax.lax.axis_index("tensor") * c, c, 1)
            x16 = jax.lax.bitcast_convert_type(x, jnp.uint16)
            full = all_gather(jnp.concatenate([x16, mine], axis=1))  # [T, tp, H/tp + c]
            hid = jax.lax.bitcast_convert_type(full[:, :, : h // tp].reshape(t, h), hdt)
            rout = full[:, :, h // tp :].reshape(t, 2 * k, 2)
            rout = jax.lax.bitcast_convert_type(rout, jnp.uint32)  # [T, 2k]
            return (
                hid,
                jax.lax.bitcast_convert_type(rout[:, :k], jnp.float32),
                jax.lax.bitcast_convert_type(rout[:, k:], jnp.int32),
            )

        rep = P(None, None)
        if not pack:
            hidden_states = shard_map(
                gather, mesh=self.mesh, in_specs=P("data", "tensor"), out_specs=rep, check_vma=False
            )(hidden_states)
            return hidden_states, topk_weights, topk_ids
        return shard_map(
            gather_packed,
            mesh=self.mesh,
            in_specs=(P("data", "tensor"), P("data", None), P("data", None)),
            out_specs=(rep, rep, rep),
            check_vma=False,
        )(hidden_states, topk_weights, topk_ids)

    def _compact_sizes(self, hidden_states, topk_ids, rs_blocks) -> tuple[int, ...]:
        """Static compact token counts (ascending, each < T) for this call; () = full path only.

        Needs mesh mode with a P("data", "tensor") [T, H] input, H divisible by dp * tp (the
        combine's column blocks) and [T, k] router outputs.
        """
        if not self.compact_divisors or rs_blocks is None or not self.rs_combine_mesh:
            return ()
        if hidden_states.ndim != 2 or topk_ids.ndim != 2:
            return ()
        if _spec_axes(jax.typeof(hidden_states).sharding.spec, 2) != (("data",), ("tensor",)):
            return ()
        dp, tp = rs_blocks
        t, h = hidden_states.shape
        if h % (dp * tp) or topk_ids.shape[0] != t:
            return ()
        sizes = {
            t // d
            for d in self.compact_divisors
            if t % d == 0 and t // d >= max(self.compact_min_tokens, 1)
        }
        return tuple(sorted(sizes))

    def wants_compact_positions(self, num_tokens: int, top_k: int) -> bool:
        """Whether a ``num_tokens``-token step can take the compact path (static, shapes only)."""
        return bool(
            self.compact_divisors
            and self.rs_combine_mesh
            and self.gmm_kernel_weights_ready
            and num_tokens * top_k >= self.local_window_min_rows
        )

    def compact_positions(self, token_mask: jax.Array) -> jax.Array | None:
        """Global compact row of every token (DP-group-major order), -1 for padding.

        [T] int32 with P("data"); the same for every MoE layer of a step, so the model computes
        it once and passes it as ``compact_pos`` (each layer would otherwise all-gather the
        per-group counts). None when the model mesh is not ("data", "tensor").
        """
        shape = dict(self.mesh.shape)
        if set(shape) != {"data", "tensor"} or token_mask.shape[0] % shape["data"]:
            return None
        dp = shape["data"]

        def local(m):  # [T/dp] bool
            m = m.astype(jnp.int32)
            counts = jax.lax.all_gather(jnp.sum(m), "data")  # real tokens per DP group
            group = jax.lax.axis_index("data")
            offset = jnp.sum(jnp.where(jnp.arange(dp) < group, counts, 0))
            return jnp.where(m > 0, offset + jnp.cumsum(m) - 1, -1).astype(jnp.int32)

        return shard_map(
            local, mesh=self.mesh, in_specs=P("data"), out_specs=P("data"), check_vma=False
        )(jax.sharding.reshard(token_mask, NamedSharding(self.mesh, P("data"))))

    @named_scope
    def _compact_forward(
        self, hidden_states, topk_weights, topk_ids, n_real, pos, *, rs_blocks, c: int
    ) -> jax.Array:
        """Prefill MoE over only the real tokens, packed into ``c`` static rows.

        The caller guarantees ``n_real`` <= ``c`` real tokens; ``pos`` is
        ``compact_positions(token_mask)``. Same math as the full path; the token gather, the
        experts' scatter-add and the combine move [c, H] instead of the padded [T, H]:

        1. Model mesh: each DP group writes its real tokens' [., H/tp] rows and router
           outputs (bit-packed into one int32 buffer) at their global compact positions of
           a zero [c, .] buffer; one psum over "data" assembles them and an all-gather over
           "tensor" the full [c, H] rows. Compact slots past the real tokens route to the
           sentinel expert, so they are neither computed nor counted.
        2. MoE mesh: the local-window experts on the c rows. Each chip emits its unreduced
           output as dp * tp column blocks of width H / (dp * tp); block (d, t) holds
           sub-slice d of tensor slice t.
        3. Model mesh: one reduce-scatter hands block (d, t) to chip (d, t), an all-gather
           over "data" rebuilds tensor slice t for all c rows, and each chip gathers its DP
           group's tokens back into the P("data", "tensor") [T, H] layout (padding rows 0).
        """
        dp, tp = rs_blocks
        t, h = hidden_states.shape
        k = topk_ids.shape[1]
        hl, bw = h // tp, h // (tp * dp)
        sentinel = self.num_experts
        hdt = hidden_states.dtype

        def to_model(x, spec):
            return jax.sharding.reshard(x, NamedSharding(self.mesh, spec))

        def gather_compact(x, wts, ids, p, n):  # [T/dp, H/tp], [T/dp, k] x2, [T/dp], [] (local)
            dst = jnp.where(p >= 0, p, c)  # padding -> out of bounds, dropped
            # One int32 psum for the hidden slice and both router outputs: the rows are
            # disjoint across groups, so integer sums of the bit patterns are exact.
            if hdt.itemsize == 2 and hl % 2 == 0:
                xw = jax.lax.bitcast_convert_type(x.reshape(x.shape[0], hl // 2, 2), jnp.int32)
            else:
                xw = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
            words = jnp.concatenate(
                [
                    xw,
                    jax.lax.bitcast_convert_type(wts.astype(jnp.float32), jnp.int32),
                    ids.astype(jnp.int32),
                ],
                axis=1,
            )
            buf = jnp.zeros((c, words.shape[1]), jnp.int32).at[dst].set(words, mode="drop")
            buf = jax.lax.psum(buf, "data")
            nx = xw.shape[1]
            if hdt.itemsize == 2 and hl % 2 == 0:
                xs = jax.lax.bitcast_convert_type(buf[:, :nx], hdt).reshape(c, hl)
            else:
                xs = jax.lax.bitcast_convert_type(buf[:, :nx], jnp.float32).astype(hdt)
            wc = jax.lax.bitcast_convert_type(buf[:, nx : nx + k], jnp.float32)
            idc = jnp.where((jnp.arange(c) < n)[:, None], buf[:, nx + k :], sentinel)
            xc = jax.lax.all_gather(xs, "tensor", axis=1, tiled=True)  # [c, H]
            return xc, wc, idc

        rep = P(None, None)
        xc, wc, idc = shard_map(
            gather_compact,
            mesh=self.mesh,
            in_specs=(P("data", "tensor"), P("data", None), P("data", None), P("data"), P()),
            out_specs=(rep, rep, rep),
            check_vma=False,
        )(
            hidden_states,
            to_model(topk_weights, P("data", None)),
            to_model(topk_ids, P("data", None)),
            to_model(pos, P("data")),
            to_model(jnp.asarray(n_real, jnp.int32), P()),
        )

        natural = self.emit_sumsq  # all-reduce the [c, H] partials as computed (no transpose)

        def experts(x, wts, ids, w_gate, w_up, w_down, s_gate, s_up, s_down):
            out = self._forward_local_window(
                x, wts, ids, w_gate, w_up, w_down, s_gate, s_up, s_down
            )  # [c, H] partial
            if natural:
                return out[None]
            # Column (t, d, j) -> block d * tp + t, the model-mesh chip (d, t).
            out = out.reshape(c, tp, dp, bw).transpose(2, 1, 0, 3)
            return out.reshape(1, dp * tp, c, bw)

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            blocks = shard_map(
                experts,
                mesh=self.moe_mesh,
                in_specs=(P(None), P(None), P(None), _WI_SPEC, _WI_SPEC, _WO_SPEC, _WI_SPEC, _WI_SPEC, _WO_SPEC),
                out_specs=P(("expert", "tensor"), None, None, *([] if natural else [None])),
                check_vma=False,
            )(
                jax.sharding.reshard(xc, P(None)),
                jax.sharding.reshard(wc, P(None)),
                jax.sharding.reshard(idc, P(None)),
                self.w1_weight_packed.value,
                self.w3_weight_packed.value,
                self.w2_weight_packed.value,
                self.w1_weight_scale.value,
                self.w3_weight_scale.value,
                self.w2_weight_scale.value,
            )
        # Same devices in the same order on both meshes: this reshard moves no data.
        blocks = to_model(blocks, P(("data", "tensor"), None, None, *([] if natural else [None])))

        def combine(x, p):  # [1, dp*tp, c, bw] partial, [T/dp] compact rows (-1 = padding)
            y = jax.lax.psum_scatter(x, ("data", "tensor"), scatter_dimension=1, tiled=True)
            y = jax.lax.all_gather(y.reshape(c, bw), "data", axis=0)  # [dp, c, bw]
            y = y.transpose(1, 0, 2).reshape(c, hl)  # this chip's tensor slice, all c rows
            rows = y[jnp.clip(p, 0, c - 1)]
            return jnp.where((p >= 0)[:, None], rows, jnp.zeros_like(rows))

        def combine_sumsq(x, p):  # [1, c, H] partial -> (as combine, row sums of squares [T/dp])
            # The all-reduce XLA runs for the reduce-scatter above anyway, on the [c, H] partials
            # as computed: this chip's tensor slice is a column slice (no transpose, no
            # all-gather), and the full rows give the sums of squares.
            y_all = jax.lax.psum(x[0], ("data", "tensor"))  # [c, H]
            y = jax.lax.dynamic_slice_in_dim(y_all, jax.lax.axis_index("tensor") * hl, hl, 1)
            y32 = y_all.astype(jnp.float32)
            ss = jnp.sum(y32 * y32, axis=1)  # [c]
            q = jnp.clip(p, 0, c - 1)
            real = p >= 0
            return (
                jnp.where(real[:, None], y[q], jnp.zeros((), y.dtype)),
                jnp.where(real, ss[q], 0.0),
            )

        def combine_sumsq_hier(x, p):  # as combine_sumsq: reduce-scatter in the group first
            y = jax.lax.psum_scatter(x[0], "tensor", scatter_dimension=1, tiled=True)  # [c, hl]
            y = jax.lax.psum(y, "data")  # every compact row of this tensor slice
            y32 = y.astype(jnp.float32)
            ss = jax.lax.psum(jnp.sum(y32 * y32, axis=1), "tensor")  # [c]
            q = jnp.clip(p, 0, c - 1)
            real = p >= 0
            return (
                jnp.where(real[:, None], y[q], jnp.zeros((), y.dtype)),
                jnp.where(real, ss[q], 0.0),
            )

        if self.emit_sumsq:
            return shard_map(
                combine_sumsq_hier if self.hier_combine else combine_sumsq,
                mesh=self.mesh,
                in_specs=(P(("data", "tensor"), None, None), P("data")),
                out_specs=(P("data", "tensor"), P("data")),
                check_vma=False,
            )(blocks, pos)
        return shard_map(
            combine,
            mesh=self.mesh,
            in_specs=(P(("data", "tensor"), None, None, None), P("data")),
            out_specs=P("data", "tensor"),
            check_vma=False,
        )(blocks, pos)

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
        from sgl_jax.srt.kernels.mxfp4_gmm import mxfp4_moe_mlp, pick_row_tile

        num_tokens, top_k = topk_ids.shape
        rows = num_tokens * top_k
        e_l = self.experts_per_device
        flat_experts = jnp.ravel(topk_ids).astype(jnp.int32)
        group_offset = (jax.lax.axis_index("expert") * e_l).astype(jnp.int32)
        local_e = flat_experts - group_offset
        # Group sizes by comparison and reduction (jnp.bincount's scatter-add cost ~0.8 ms per
        # layer on v5e), over this shard's experts only. Ids >= num_experts (masked DP padding)
        # are never local.
        local_sizes = jnp.sum(
            local_e[:, None] == jnp.arange(e_l, dtype=jnp.int32)[None, :], axis=0, dtype=jnp.int32
        )
        if rows <= _PACKED_SORT_MAX_ROWS:
            # One single-operand sort of (local expert, row) keys, non-local rows last: the
            # local rows come first in the same order as the stable expert sort below, and
            # sorting one int32 array instead of a key/value pair moves half the data.
            is_local = (local_e >= 0) & (local_e < e_l)
            key = jnp.where(
                is_local,
                (local_e << 16) | jnp.arange(rows, dtype=jnp.int32),
                jnp.iinfo(jnp.int32).max,
            )
            sorted_pos = jnp.minimum(jax.lax.sort(key) & jnp.int32(0xFFFF), rows - 1)
            start = jnp.int32(0)
        else:
            # One stable key/value sort gives the expert-sorted order without a separate
            # gather (a 65K-index gather cost ~0.6 ms per layer on v5e).
            _, sorted_pos = jax.lax.sort_key_val(
                flat_experts, jnp.arange(rows, dtype=jnp.int32), is_stable=True
            )
            start = jnp.sum((flat_experts < group_offset).astype(jnp.int32))
        local_rows = jnp.sum(local_sizes)
        local_cum = jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(local_sizes).astype(jnp.int32)]
        )
        flat_w = jnp.ravel(topk_weights).astype(jnp.float32)
        cap = self._window_capacity(rows)
        no_groups = jnp.zeros((self.num_experts,), jnp.int32)
        # Row tile of the local-window GMMs (SGLANG_JAX_MOE_LOCAL_WINDOW_TM): "auto" picks it from
        # the window's rows per *local* expert, an integer fixes it; unset keeps the kernel's
        # default, which divides by the global group count (896 for Kimi-K3) and always picks 16
        # here, so a shard with hundreds of local rows runs them as 16-row matmuls.
        tm_env = os.environ.get("SGLANG_JAX_MOE_LOCAL_WINDOW_TM", "").strip().lower()
        tm, cache = None, None
        if tm_env == "auto":
            # A loaded expert spans several row tiles: keep its decoded weights in VMEM.
            tm, cache = pick_row_tile(cap, e_l, hi=256), True
        elif tm_env.isdigit():
            tm, cache = int(tm_env), True

        combine = self.local_combine
        if combine == "onehot" and num_tokens > self.onehot_max_tokens:
            combine = self.local_combine_large
        acc_dtype = self.local_acc_dtype

        def window(w, acc):  # acc None: the first window, whose sum is the accumulator
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
                tm=tm,
                cache=cache,
                zero_unvisited=False,  # rows past the local groups are masked below
                interpret=not is_tpu_runtime(),
            )
            contrib = jnp.where(valid[:, None], y.astype(jnp.float32) * flat_w[pos][:, None], 0.0)
            if combine == "onehot":
                # [T, cap] 0/1 matrix (masked rows are zero columns) on the MXU: no scatter.
                onehot = (
                    (tok[None, :] == jnp.arange(num_tokens, dtype=tok.dtype)[:, None])
                    & valid[None, :]
                ).astype(acc_dtype)
                add = jnp.dot(
                    onehot,
                    contrib.astype(acc_dtype),
                    preferred_element_type=jnp.float32,
                    precision=jax.lax.Precision.HIGHEST if acc_dtype == jnp.float32 else None,
                ).astype(acc_dtype)
                return add if acc is None else acc + add
            if acc is None:
                acc = jnp.zeros((num_tokens, w_down.shape[-1]), acc_dtype)
            if combine == "gather":
                return acc + _segment_gather_sum(
                    contrib.astype(acc.dtype), tok, valid, num_tokens
                ).astype(acc.dtype)
            if combine == "sorted":
                key = jnp.where(valid, tok, num_tokens)  # masked rows sort last, then drop
                key_s, order = jax.lax.sort_key_val(key, jnp.arange(cap, dtype=jnp.int32))
                return acc.at[key_s].add(
                    contrib[order].astype(acc.dtype), indices_are_sorted=True, mode="drop"
                )
            return acc.at[tok].add(contrib.astype(acc.dtype))

        n_windows = (local_rows + cap - 1) // cap
        # Window 0 runs unconditionally (an empty one contributes zeros): with the one-hot
        # combine its matmul output is the accumulator, without a zero fill and an add of the
        # whole [T, H] (~55 us per MoE layer at T = 2304 on v6e).
        acc = window(0, None)
        acc = jax.lax.fori_loop(1, n_windows, window, acc)
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
    expert_order=None,
) -> dict:
    """Map per-expert compressed-tensors MXFP4 tensors onto ``MXFP4EPMoE``.

    For each of gate/up/down (``expert_type_names``) the ``num_experts``
    checkpoint tensors ``{prefix}.{source_expert_pattern}.{name}.weight_packed``
    (and ``.weight_scale``) are transposed to k-major and stacked into
    ``{target_prefix}.{moe_path}.{w1|w3|w2}_weight_packed`` (``_weight_scale``).
    The loader reads only each host's addressable expert / intermediate slices.

    ``expert_order`` (EPLB): the logical expert stored in each physical slot, i.e. this
    layer's row of ``ExpertLocationMetadata.physical_to_logical_map``. Slot ``p`` then
    holds checkpoint expert ``expert_order[p]`` (a logical expert may fill several slots
    when there are redundant experts). None keeps the checkpoint order.
    """
    gate, up, down = expert_type_names
    layouts = {gate: ("w1", _WI_SPEC), up: ("w3", _WI_SPEC), down: ("w2", _WO_SPEC)}
    order = list(range(num_experts)) if expert_order is None else [int(i) for i in expert_order]
    if any(not 0 <= i < num_experts for i in order):
        raise ValueError(f"expert_order entries must be in [0, {num_experts})")
    mappings = {}
    for source_name, (target_name, spec) in layouts.items():
        for suffix in ("weight_packed", "weight_scale"):
            target = f"{target_prefix}.{moe_path}.{target_name}_{suffix}"
            mappings[target] = WeightSpec(
                target_path=target,
                sources=tuple(
                    f"{prefix}.{source_expert_pattern.format(i=i)}.{source_name}.{suffix}"
                    for i in order
                ),
                sharding=tuple(spec),
                transpose=True,
            )
    return mappings
