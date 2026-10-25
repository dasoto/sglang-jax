"""Attention backend for the absorbed MLA path (MLA v2 Pallas kernel).

This backend wraps `mla_ragged_paged_attention` with a 4D paged latent KV cache
(see `MLATokenToKVPool`). It follows the same `(q, k, v, layer, forward_batch,
pool, **kwargs)` contract as the MHA FlashAttention backend so that
`RadixAttention` remains the unified entry point. Callers pass the absorbed
latent Q as `q`, the latent `c_kv` as both `k` and `v`, and the rope parts as
the `q_rope` / `k_rope` kwargs; the backend reassembles the 4-tuple
`(ql_nope, q_pe, new_kv_c, new_k_pe)` the Pallas kernel consumes. The output
is the latent attention `o_latent: [T, n_h, kv_lora_rank]`; the caller
projects it through `W_UV → W_O` (see `docs/design/MLA.md` §3.9).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.mla.v2.kernel import cdiv, mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# DCP combine by reduce-scatter instead of all_to_all (see _run_dcp). Read at trace time.
MLA_DCP_RS_COMBINE_ENV = "SGLANG_JAX_MLA_DCP_RS_COMBINE"


def _dcp_rs_combine_enabled() -> bool:
    return os.environ.get(MLA_DCP_RS_COMBINE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@register_pytree_node_class
@dataclass
class MLAAttentionMetadata:
    """Per-forward metadata for the MLA v2 kernel.

    Drops `custom_mask` / `swa_page_indices` / `attention_sink` (RFC §3.8). The
    `cu_kv_lens` field carries the page-aligned KV cumulative lengths used by
    the kernel to locate each sequence's pages in the ragged `page_indices`
    layout (matches RPA v3 addressing).
    """

    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None
    page_seq_positions: jax.Array = None
    global_seq_lens: jax.Array = None

    def tree_flatten(self):
        children = (
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
            self.page_seq_positions,
            self.global_seq_lens,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.cu_q_lens = children[0]
        obj.cu_kv_lens = children[1]
        obj.page_indices = children[2]
        obj.seq_lens = children[3]
        obj.distribution = children[4]
        obj.page_seq_positions = children[5] if len(children) > 5 else None
        obj.global_seq_lens = children[6] if len(children) > 6 else None
        return obj


@dataclass
class MLAAttentionBackend(AttentionBackend):
    """Absorbed-MLA attention backend backed by the v2 Pallas kernel."""

    def __init__(
        self,
        num_attn_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        page_size: int = 1,
        mesh: jax.sharding.Mesh = None,
        attention_data_partition_axis: str = "data",
        vmem_limit_bytes: int | None = None,
        # Default block params to None so the kernel's auto-tuned-table
        # lookup path triggers (see kernel.py mla_ragged_paged_attention
        # entry). Callers (e.g. accuracy tests) can still pass explicit
        # tuples to bypass lookup.
        num_kv_pages_per_block: tuple[int, int, int] | None = None,
        num_queries_per_block: tuple[int, int, int] | None = None,
        # decode_batch_size: kernel-internal microbatch for the BATCHED_DECODE
        # branch. The v2 kernel runs `floor(num_decode_seqs / decode_batch_size)
        # * decode_batch_size` requests through a batched path (q_len=1, q-tile
        # of size `decode_batch_size`) to keep the MXU busy on MLA's MQA-on-
        # latent decode (single q row per seq otherwise underutilizes the q
        # axis); the remainder falls back to per-seq decode.
        # When num_kv_pages_per_block is None, the kernel's lookup path also
        # overrides this with the tuned value if the "decode" entry hits.
        decode_batch_size: int = 4,
        dcp_size: int = 1,
    ):
        assert page_size > 1, (
            "MLA attention backend does not support page_size=1. "
            "The MLA v2 kernel packs KV slots and infers the effective page size "
            "from cache_kv.shape[1] * kv_packing, which makes page_size=1 "
            "inconsistent with allocator and metadata page semantics."
        )
        self.num_heads = num_attn_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.mesh = mesh
        self.attention_data_partition_axis = attention_data_partition_axis
        self.dcp_size = dcp_size
        if vmem_limit_bytes is None:
            if jax.default_backend() == "tpu":
                from jax.experimental.pallas import tpu as pltpu

                vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
            else:
                # CPU (Pallas interpret) runs have no VMEM to size against.
                vmem_limit_bytes = 64 * 1024 * 1024
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_kv_pages_per_block = num_kv_pages_per_block
        self.num_queries_per_block = num_queries_per_block
        self.decode_batch_size = decode_batch_size

        self.forward_metadata = nnx.data(MLAAttentionMetadata())

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Build per-batch metadata, DP-aware.

        Mirrors `FlashAttention.get_forward_metadata`: reshapes all per-request
        arrays to `(dp_size, per_dp_bs_size)` and computes cumsums per DP rank
        so each rank's metadata is independent.
        """
        metadata = MLAAttentionMetadata()

        if batch.dp_size <= 0:
            raise ValueError(f"Invalid dp_size: {batch.dp_size}")
        if batch.per_dp_bs_size <= 0:
            raise ValueError(f"Invalid per_dp_bs_size: {batch.per_dp_bs_size}")
        if batch.per_dp_bs_size * batch.dp_size != len(batch.seq_lens):
            raise ValueError(
                "Inconsistent DP batch metadata: expected per_dp_bs_size * dp_size == len(seq_lens), "
                f"got {batch.per_dp_bs_size} * {batch.dp_size} != {len(batch.seq_lens)}"
            )
        if len(batch.cache_loc) % batch.dp_size != 0:
            raise ValueError(
                "Inconsistent cache_loc layout for DP sharding: "
                f"len(cache_loc)={len(batch.cache_loc)} is not divisible by dp_size={batch.dp_size}"
            )

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size

        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        strided_2d = cache_loc_2d[:, :: self.page_size]
        logical_pages_2d = strided_2d // self.page_size

        if batch.forward_mode == ForwardMode.EXTEND:
            ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
            cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
            cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
            cu_q_lens = cu_q_2d.ravel()
        elif batch.forward_mode == ForwardMode.DECODE:
            single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
            cu_q_lens = np.tile(single_cu, batch.dp_size)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = batch.seq_lens
        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size

        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        cu_kv_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)

        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), np.zeros_like(local_num_seqs), local_num_seqs]
            ).ravel()
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        if self.dcp_size <= 1:
            page_indices = logical_pages_2d.ravel()
            cu_kv_lens = cu_kv_2d.ravel()
            (
                metadata.cu_q_lens,
                metadata.cu_kv_lens,
                metadata.page_indices,
                metadata.seq_lens,
                metadata.distribution,
            ) = device_array(
                (cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
                sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
            )
            return metadata

        # DCP > 1: build per-(dp_rank, dcp_rank) local page tables and sequence positions.
        per_dp_pages = per_dp_loc_len // self.page_size
        max_pages_per_rank = (per_dp_pages // self.dcp_size) + 2 * batch.per_dp_bs_size

        dcp_page_indices = np.zeros(
            (batch.dp_size, self.dcp_size, max_pages_per_rank), dtype=np.int32
        )
        dcp_page_seq_pos = np.full(
            (batch.dp_size, self.dcp_size, max_pages_per_rank), 0x3FFFFFFF, dtype=np.int32
        )
        dcp_cu_kv_lens = np.zeros(
            (batch.dp_size, self.dcp_size, batch.per_dp_bs_size + 1), dtype=np.int32
        )
        dcp_seq_lens = np.zeros(
            (batch.dp_size, self.dcp_size, batch.per_dp_bs_size), dtype=np.int32
        )

        for dp_r in range(batch.dp_size):
            write_cursors = [0] * self.dcp_size
            for s in range(batch.per_dp_bs_size):
                s_len = int(seq_lens_2d[dp_r, s])
                if s_len <= 0:
                    for r in range(self.dcp_size):
                        dcp_cu_kv_lens[dp_r, r, s + 1] = write_cursors[r] * self.page_size
                    continue
                p_start = int(cu_kv_2d[dp_r, s]) // self.page_size
                p_end = int(cu_kv_2d[dp_r, s + 1]) // self.page_size
                seq_pages = logical_pages_2d[dp_r, p_start:p_end]
                seq_pos = np.arange(len(seq_pages), dtype=np.int32) * self.page_size
                valid_mask = seq_pages > 0
                valid_pages = seq_pages[valid_mask]
                valid_pos = seq_pos[valid_mask]
                owners = (valid_pages - 1) % self.dcp_size
                local_pages = ((valid_pages - 1) // self.dcp_size) + 1

                for r in range(self.dcp_size):
                    r_mask = owners == r
                    r_pages = local_pages[r_mask]
                    r_pos = valid_pos[r_mask]
                    c = write_cursors[r]
                    if len(r_pages) == 0:
                        # Ensure >= 1 page so kernel bkv loop executes at least once and drains DMA semaphores.
                        dcp_page_indices[dp_r, r, c] = 0
                        dcp_page_seq_pos[dp_r, r, c] = 0x3FFFFFFF
                        write_cursors[r] = c + 1
                        dcp_seq_lens[dp_r, r, s] = 1
                    else:
                        n_p = len(r_pages)
                        dcp_page_indices[dp_r, r, c : c + n_p] = r_pages
                        dcp_page_seq_pos[dp_r, r, c : c + n_p] = r_pos
                        write_cursors[r] = c + n_p
                        last_pos = int(r_pos[-1])
                        last_tokens = min(self.page_size, max(1, s_len - last_pos))
                        dcp_seq_lens[dp_r, r, s] = (n_p - 1) * self.page_size + last_tokens
                    dcp_cu_kv_lens[dp_r, r, s + 1] = write_cursors[r] * self.page_size

        dpa = self.attention_data_partition_axis
        (
            metadata.cu_q_lens,
            metadata.distribution,
            metadata.global_seq_lens,
        ) = device_array(
            (cu_q_lens, distribution, seq_lens),
            sharding=(NamedSharding(self.mesh, P(dpa))),
        )
        (
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.page_seq_positions,
        ) = device_array(
            (
                dcp_cu_kv_lens.ravel(),
                dcp_page_indices.ravel(),
                dcp_seq_lens.ravel(),
                dcp_page_seq_pos.ravel(),
            ),
            sharding=(NamedSharding(self.mesh, P((dpa, "tensor")))),
        )
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_attn_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "mesh": self.mesh,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "num_kv_pages_per_block": self.num_kv_pages_per_block,
            "num_queries_per_block": self.num_queries_per_block,
            "decode_batch_size": self.decode_batch_size,
            "dcp_size": self.dcp_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=aux_data.get("mesh"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            vmem_limit_bytes=aux_data["vmem_limit_bytes"],
            num_kv_pages_per_block=aux_data["num_kv_pages_per_block"],
            num_queries_per_block=aux_data["num_queries_per_block"],
            decode_batch_size=aux_data["decode_batch_size"],
            dcp_size=aux_data.get("dcp_size", 1),
        )
        obj.forward_metadata = children[0]
        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        """Absorbed-MLA forward, called through ``RadixAttention``.

        Signature mirrors the other attention backends so ``RadixAttention``
        can serve as the unified entry point. The MLA-specific tensors land in
        standard slots:

            q:         [T, n_h, kv_lora_rank]            — absorbed latent query (``ql_nope``)
            k:         [T, 1, kv_lora_rank]              — latent KV (c_kv); v is the same tensor
            v:         [T, 1, kv_lora_rank]              — same object as ``k`` (MLA is MQA on the latent)
            q_rope=…:  [T, n_h, qk_rope_head_dim]        — ``q_pe``
            k_rope=…:  [T, 1, qk_rope_head_dim]          — ``k_pe`` (shared across heads)

        Internally reassembles the 4-tuple ``(ql_nope, q_pe, new_kv_c, new_k_pe)``
        the v2 Pallas kernel consumes. Returns ``(o_latent, updated_cache)``
        with ``o_latent: [T, n_h, kv_lora_rank]`` — the caller is responsible
        for projecting via ``W_UV → W_O``.

        The kernel handles ``qk_rope_head_dim=64 → 128`` and other
        ``align_to(*, 128)`` padding internally, so callers pass logical dims.
        """
        # v is the same latent tensor as k in MLA-MQA — we only need one.
        del v
        q_rope = kwargs.get("q_rope")
        k_rope = kwargs.get("k_rope")
        # Optional (w_uk, w_uv), replicated [R, n_h, D]: with DCP, q is then the un-absorbed
        # [T, n_h, D_k] query and the result the [T, n_h, D_v] output (see _run_dcp).
        absorb_weights = kwargs.get("absorb_weights")
        if absorb_weights is not None and self.dcp_size <= 1:
            raise ValueError("absorb_weights is only supported with dcp_size > 1")
        if q_rope is None or k_rope is None:
            raise ValueError(
                "MLAAttentionBackend requires q_rope/k_rope kwargs (q_pe/k_pe) "
                "alongside the non-rope q/k tensors."
            )

        # Strip the (single) KV-head axis from the latent K/K-rope tensors.
        # Squeezing a length-1 axis preserves any caller-provided sharding on
        # the remaining dims (replicated stays replicated).
        new_kv_c = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)
        dpa = self.attention_data_partition_axis
        new_k_pe = jax.sharding.reshard(new_k_pe, P(dpa, None))
        ql_nope = q
        q_pe = q_rope

        cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        sliding_window = layer.sliding_window_size if layer is not None else None
        soft_cap = layer.logit_cap if layer is not None else None

        if self.dcp_size <= 1:
            in_specs = (
                P(dpa, "tensor", None),  # ql_nope    [T, n_h/tp, lkv]
                P(dpa, "tensor", None),  # q_pe       [T, n_h/tp, r]
                P(dpa, None),  # new_kv_c   [T, lkv]  (single latent, no head axis)
                P(dpa, None),  # new_k_pe   [T, r]    (single latent)
                P(dpa, None, None, None),  # cache (page axis sharded by data)
                P(dpa),  # seq_lens
                P(dpa),  # page_indices
                P(dpa),  # cu_q_lens
                P(dpa),  # cu_kv_lens
                P(dpa),  # distribution
            )
            out_specs = (
                P(dpa, "tensor", None),  # o_latent       [T, n_h/tp, lkv]
                P(dpa, None, None, None),  # updated cache  4D
            )

            def _run(
                ql_nope_,
                q_pe_,
                new_kv_c_,
                new_k_pe_,
                cache_,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
            ):
                return mla_ragged_paged_attention(
                    ql_nope_,
                    q_pe_,
                    new_kv_c_,
                    new_k_pe_,
                    cache_,
                    seq_lens_,
                    page_indices_,
                    cu_q_lens_,
                    cu_kv_lens_,
                    distribution_,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    soft_cap=soft_cap,
                    num_kv_pages_per_block=self.num_kv_pages_per_block,
                    num_queries_per_block=self.num_queries_per_block,
                    decode_batch_size=self.decode_batch_size,
                    vmem_limit_bytes=self.vmem_limit_bytes,
                )

            o_latent, updated_cache = jax.shard_map(
                _run,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache,
                self.forward_metadata.seq_lens,
                self.forward_metadata.page_indices,
                self.forward_metadata.cu_q_lens,
                self.forward_metadata.cu_kv_lens,
                self.forward_metadata.distribution,
            )

            return o_latent, updated_cache

        # DCP > 1 path:
        # 1. Pre-scatter owned new KV tokens into local cache_
        # 2. AllGather queries across "tensor" so each DCP rank scores all query heads against its local KV pages
        # 3. Run mla_ragged_paged_attention with return_lse=True, page_seq_positions, global_kv_lens
        # 4. Combine partial (o_partial, lse_partial) across "tensor" using LSE reduction and slice local TP heads
        from sgl_jax.srt.kernels.mla.v2.kernel import align_to

        in_specs_dcp = (
            P(dpa, "tensor", None),  # ql_nope
            P(dpa, "tensor", None),  # q_pe
            P(dpa, None),  # new_kv_c
            P(dpa, None),  # new_k_pe
            P((dpa, "tensor"), None, None, None),  # cache sharded by (data, tensor)
            P((dpa, "tensor")),  # local seq_lens
            P((dpa, "tensor")),  # local page_indices
            P(dpa),  # cu_q_lens
            P((dpa, "tensor")),  # local cu_kv_lens
            P(dpa),  # distribution
            P((dpa, "tensor")),  # page_seq_positions
            P(dpa),  # global_seq_lens
            P(dpa),  # out_cache_loc
        )
        out_specs_dcp = (
            P(dpa, "tensor", None),  # o_latent [T, n_h/tp, lkv]
            P((dpa, "tensor"), None, None, None),  # updated cache
        )

        def _run_dcp(
            ql_nope_,
            q_pe_,
            new_kv_c_,
            new_k_pe_,
            cache_,
            seq_lens_,
            page_indices_,
            cu_q_lens_,
            cu_kv_lens_,
            distribution_,
            page_seq_positions_,
            global_seq_lens_,
            out_cache_loc_,
            w_uk_=None,
            w_uv_=None,
        ):
            dcp_rank = jax.lax.axis_index("tensor")
            lkv_dim = align_to(new_kv_c_.shape[-1], 128)
            r_dim = align_to(new_k_pe_.shape[-1], 128)

            kv_c_pad = (
                jnp.pad(new_kv_c_, ((0, 0), (0, lkv_dim - new_kv_c_.shape[-1])))
                if lkv_dim != new_kv_c_.shape[-1]
                else new_kv_c_
            )
            k_pe_pad = (
                jnp.pad(new_k_pe_, ((0, 0), (0, r_dim - new_k_pe_.shape[-1])))
                if r_dim != new_k_pe_.shape[-1]
                else new_k_pe_
            )
            new_kv_fused = jnp.concatenate([kv_c_pad, k_pe_pad], axis=-1).astype(cache_.dtype)

            logical_page = out_cache_loc_ // self.page_size
            in_page = out_cache_loc_ % self.page_size
            is_owned = (out_cache_loc_ >= self.page_size) & (
                ((logical_page - 1) % self.dcp_size) == dcp_rank
            )
            local_page = ((logical_page - 1) // self.dcp_size) + 1

            # Scatter into the cache in its own (packed, paged) shape: reshaping it to
            # [-1, dim] and back changes the TPU tiling, which copied the whole layer cache
            # twice per step (~4 ms per decode step on v5e-256). Same slots as the flat view.
            flat_slot = local_page * self.page_size + in_page
            idx = list(jnp.unravel_index(jnp.where(is_owned, flat_slot, 0), cache_.shape[:-1]))
            idx[0] = jnp.where(is_owned, idx[0], cache_.shape[0])  # out of bounds: dropped
            cache_updated = cache_.at[tuple(idx)].set(new_kv_fused, mode="drop")

            # Gather all query heads so every DCP rank computes attention for all heads on its local KV pages.
            if absorb_weights is not None:
                # Replicated absorb: gather the [T, n_h, D_k] queries (a quarter of the latent
                # ones for D_k = 128, R = 512) and fold W_UK in for every head locally.
                q_nope_all = jax.lax.all_gather(ql_nope_, axis_name="tensor", axis=1, tiled=True)
                ql_nope_all = jnp.einsum(
                    "thd,rhd->thr", q_nope_all, w_uk_, preferred_element_type=jnp.float32
                ).astype(ql_nope_.dtype)
            else:
                ql_nope_all = jax.lax.all_gather(ql_nope_, axis_name="tensor", axis=1, tiled=True)
            q_pe_all = jax.lax.all_gather(q_pe_, axis_name="tensor", axis=1, tiled=True)

            o_partial, cache_updated, lse_partial = mla_ragged_paged_attention(
                ql_nope_all,
                q_pe_all,
                new_kv_c_,
                new_k_pe_,
                cache_updated,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
                page_seq_positions=page_seq_positions_,
                global_kv_lens=global_seq_lens_,
                return_lse=True,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )
            if absorb_weights is not None:
                # Fold W_UV into every head's partial output before the combine (the combine
                # is linear in it), so the cross-rank reduction moves [T, n_h, D_v] instead of
                # [T, n_h, R]: a quarter of the bytes for D_v = 128, R = 512.
                o_partial = jnp.einsum(
                    "thr,rhd->thd",
                    o_partial[..., : w_uv_.shape[0]],
                    w_uv_,
                    preferred_element_type=jnp.float32,
                )

            if _dcp_rs_combine_enabled():
                # Same LSE-weighted combine as below, with a reduce-scatter over "tensor"
                # (scattering the head axis, so rank g gets head group g) instead of the
                # all_to_all: on v5e's torus a reduce-scatter moves each byte about one hop,
                # while the all_to_all is bisection-bound (~3 ms per MLA layer for a
                # 4096-token prefill on v5e-256). Weighting by exp(lse - global max) before
                # the sum keeps every term <= 1; the sums run in f32.
                lse_max = jax.lax.pmax(lse_partial, "tensor")
                lse_max_safe = jnp.where(jnp.isneginf(lse_max), 0.0, lse_max)
                w = jnp.where(jnp.isneginf(lse_partial), 0.0, jnp.exp(lse_partial - lse_max_safe))
                num = jnp.where(
                    w[..., None] > 0.0, o_partial.astype(jnp.float32) * w[..., None], 0.0
                )
                num = jax.lax.psum_scatter(num, "tensor", scatter_dimension=1, tiled=True)
                den = jax.lax.psum_scatter(w, "tensor", scatter_dimension=1, tiled=True)
                o_combined = num / jnp.where(den > 0.0, den, 1.0)[..., None]
                return o_combined.astype(ql_nope_.dtype), cache_updated

            # Combine (o_partial, lse_partial) across the "tensor" (DCP) axis.
            # o_partial: [T_dp, N_h, lkv_dim], lse_partial: [T_dp, N_h]
            # Each DCP rank only keeps its own n_h_local query heads, so an all_to_all
            # that sends head group g to rank g moves 1/dcp_size of the data an
            # all_gather of every head would (the all_gather was ~23% of a 4096-token
            # prefill step on v5e-256). Result: [dcp_size (source rank), T_dp, n_h_local, ...],
            # the same layout as slicing this rank's heads out of the all_gather.
            n_h_local = ql_nope_.shape[1]
            dcp = o_partial.shape[1] // n_h_local
            o_local_heads = jax.lax.all_to_all(
                o_partial.reshape(o_partial.shape[0], dcp, n_h_local, o_partial.shape[-1]),
                axis_name="tensor",
                split_axis=1,
                concat_axis=0,
                tiled=False,
            ).astype(jnp.float32)
            lse_local_heads = jax.lax.all_to_all(
                lse_partial.reshape(lse_partial.shape[0], dcp, n_h_local),
                axis_name="tensor",
                split_axis=1,
                concat_axis=0,
                tiled=False,
            )

            lse_max = jnp.max(lse_local_heads, axis=0, keepdims=True)
            lse_max_safe = jnp.where(jnp.isneginf(lse_max), 0.0, lse_max)
            weights = jnp.where(
                jnp.isneginf(lse_local_heads),
                0.0,
                jnp.exp(lse_local_heads - lse_max_safe),
            )
            weights_sum = jnp.sum(weights, axis=0)
            weights_norm = weights / jnp.where(weights_sum > 0.0, weights_sum, 1.0)[None, ...]
            o_combined = jnp.sum(o_local_heads * weights_norm[..., None], axis=0).astype(
                ql_nope_.dtype
            )
            return o_combined, cache_updated

        absorb_args = ()
        if absorb_weights is not None:
            # Replicated (rank, n_h, dim) W_UK / W_UV: q is the [T, n_h, D_k] query and the
            # result is the [T, n_h, D_v] attention output (see _run_dcp).
            in_specs_dcp = in_specs_dcp + (P(None, None, None), P(None, None, None))
            absorb_args = tuple(absorb_weights)

        o_latent, updated_cache = jax.shard_map(
            _run_dcp,
            in_specs=in_specs_dcp,
            out_specs=out_specs_dcp,
            check_vma=False,
        )(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache,
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            self.forward_metadata.page_seq_positions,
            self.forward_metadata.global_seq_lens,
            forward_batch.out_cache_loc,
            *absorb_args,
        )

        return o_latent, updated_cache

    def get_max_running_reqests(self, max_context_len: int, page_size: int) -> int:
        # Same scalar-prefetch budget heuristic as FlashAttention.get_max_running_reqests:
        # caps the number of concurrent requests by the per-request page slot count
        # so metadata arrays fit in TPU scalar-prefetch memory.
        eff_page_size = page_size * max(1, getattr(self, "dcp_size", 1))
        num_page_per_req = cdiv(max_context_len, eff_page_size)
        res = 1024 * 1024 // 2 // max(1, num_page_per_req) // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res

    def get_eagle_base_metadata(self, batch, **kwargs):
        """Metadata for the fused speculative JITs that rebuild dynamic fields on device.

        The FlashAttention backend uploads only page ids here; the MLA/DSA verify and
        draft-extend helpers (``_make_target_verify_metadata`` /
        ``_make_draft_extend_metadata``) rebuild seq_lens, cu_q_lens, cu_kv_lens and the
        page table from ``page_indices`` themselves, so the full host metadata is a
        valid (if slightly larger) base. Keep one construction path until MLA grows a
        page-only upload.
        """
        return self.get_eagle_forward_metadata(batch, **kwargs)

    def get_eagle_forward_metadata(self, batch, **kwargs):
        # We don't support custom_mask in MLA yet, but for NEXTN it's just sequential draft tokens.
        import numpy as np

        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        metadata = MLAAttentionMetadata()

        dp_size = batch.dp_size
        per_dp_bs = batch.per_dp_bs_size if dp_size > 1 else len(batch.seq_lens)

        if batch.forward_mode.is_target_verify():
            padded_batch_size = len(batch.seq_lens)
            extend_seq_lens = np.zeros(padded_batch_size, dtype=np.int32)
            extend_seq_lens[batch.logits_indices_selector] = batch.spec_info_padded.draft_token_num
        else:
            extend_seq_lens = batch.extend_seq_lens

        ext_2d = extend_seq_lens.reshape(dp_size, per_dp_bs)
        cu_q_2d = np.zeros((dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
        metadata.cu_q_lens = cu_q_2d.ravel()

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size
        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        strided_2d = cache_loc_2d[:, :: self.page_size]
        metadata.page_indices = (strided_2d // self.page_size).ravel()

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, per_dp_bs)
        cu_kv_2d = np.zeros((batch.dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
        metadata.cu_kv_lens = cu_kv_2d.ravel()

        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, per_dp_bs)
        metadata.seq_lens = seq_lens_2d.ravel()

        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        else:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), np.zeros_like(local_num_seqs), local_num_seqs]
            ).ravel()
        metadata.distribution = distribution

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.utils.jax_utils import device_array

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (
                metadata.cu_q_lens,
                metadata.cu_kv_lens,
                metadata.page_indices,
                metadata.seq_lens,
                metadata.distribution,
            ),
            sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
        )

        return metadata
