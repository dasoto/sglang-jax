"""DSA sparse attention with decode context parallelism: dcp=1 vs dcp=4 parity.

Runs the DSA backend on one host's 4 chips (mesh data=1, tensor=4) through an
extend, decode steps crossing a page boundary, and a chunked extend over a
cached prefix, and compares the attention outputs of a dcp=1 and a dcp=4 pool.
TPU-only: the indexer kernels have no interpret-mode lowering.
"""

import functools
import os

os.environ.setdefault("DSA_INDEXER_KERNEL", "1")
os.environ.setdefault("DSA_INDEXER_KERNEL_PREFILL", "1")
os.environ.setdefault("DSA_PREFILL_SPARSE", "1")
os.environ.setdefault("DSA_INDEXER_PREFILL_QB", "128")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

if jax.default_backend() != "tpu" or len(jax.local_devices()) < 4:
    pytest.skip("DSA DCP test needs 4 local TPU chips", allow_module_level=True)

from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSASparseAttentionBackend
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

PS = 128
HEADS = 16  # 4 per chip at TP4
LKV, ROPE = 512, 64
IDX_H, IDX_D = 32, 128
INDEX_TOPK = 256  # 2 pages per prefill query, 256-token decode top-k


class Setup:
    def __init__(self, mesh, dcp):
        size = 64 * PS
        self.pool = MLATokenToKVPool(
            size=size,
            page_size=PS,
            dtype=jnp.bfloat16,
            kv_lora_rank=LKV,
            qk_rope_head_dim=ROPE,
            layer_num=1,
            mesh=mesh,
            dp_size=1,
            dcp_size=dcp,
            indexer_key_dim=IDX_D,
            num_indexer_layers=1,
        )
        self.alloc = PagedTokenToKVPoolAllocator(
            size=size, page_size=PS, kvcache=self.pool, dp_size=1
        )
        self.backend = DSASparseAttentionBackend(
            index_topk=INDEX_TOPK,
            index_head_dim=IDX_D,
            index_n_heads=IDX_H,
            skip_offset=0,
            full_slot={0: 0},
            num_attn_heads=HEADS,
            kv_lora_rank=LKV,
            qk_nope_head_dim=192,
            qk_rope_head_dim=ROPE,
            v_head_dim=256,
            page_size=PS,
            mesh=mesh,
            attention_data_partition_axis="data",
            dcp_size=dcp,
        )
        self.tokens = []  # per-request slot lists


def _cache_loc(tokens, seq_lens):
    chunks = []
    for toks, sl in zip(tokens, seq_lens):
        aligned = -(-sl // PS) * PS
        chunks.extend(list(toks) + [0] * (aligned - len(toks)))
    return np.array(chunks, dtype=np.int32)


@functools.partial(jax.jit, static_argnames=("mode",))
def _attend(
    backend, kv, idx, out_loc, positions, ql, qpe, kvc, kpe, qi, ki, wi, *, mode
):
    layer = SimpleNamespace(
        layer_id=0,
        scaling=1.0 / np.sqrt(192 + ROPE),
        sliding_window_size=None,
        logit_cap=None,
    )
    pool = SimpleNamespace(
        get_fused_kv_buffer=lambda _: kv, get_indexer_key_buffer=lambda _: idx
    )
    fb = SimpleNamespace(forward_mode=mode, out_cache_loc=out_loc, positions=positions)
    o, fused = backend(
        ql,
        kvc,
        None,
        layer,
        fb,
        pool,
        q_rope=qpe,
        k_rope=kpe,
        indexer_type="full",
        q_idx=qi,
        k_idx=ki,
        idx_weights=wi,
    )
    return o, fused.kv, fused.idx


def _step(mesh, st, mode, seq_lens, ext_lens, out_loc, positions, inputs):
    batch = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=len(seq_lens),
        seq_lens=np.array(seq_lens, np.int32),
        extend_seq_lens=None if ext_lens is None else np.array(ext_lens, np.int32),
        cache_loc=_cache_loc(st.tokens, seq_lens),
        forward_mode=mode,
    )
    st.backend.forward_metadata = st.backend.get_forward_metadata(batch)
    rep = NamedSharding(mesh, P("data"))
    heads = NamedSharding(mesh, P("data", "tensor", None))
    row = NamedSharding(mesh, P("data", None))
    ql, qpe, kvc, kpe, qi, ki, wi = inputs
    with jax.set_mesh(mesh):
        o, kv, idx = _attend(
            st.backend,
            st.pool.get_fused_kv_buffer(0),
            st.pool.get_indexer_key_buffer(0),
            jax.device_put(np.array(out_loc, np.int32), rep),
            jax.device_put(np.array(positions, np.int32), rep),
            jax.device_put(ql, heads),
            jax.device_put(qpe, heads),
            jax.device_put(kvc, row),
            jax.device_put(kpe, row),
            jax.device_put(qi, NamedSharding(mesh, P("data", None, None))),
            jax.device_put(ki, row),
            jax.device_put(wi, row),
            mode=mode,
        )
    st.pool.replace_buffer(([kv], [idx]))
    return np.asarray(jax.device_get(o), np.float32)


def _inputs(rng, t):
    f = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)  # noqa: E731
    return (
        jnp.asarray(f(t, HEADS, LKV), jnp.bfloat16),
        jnp.asarray(f(t, HEADS, ROPE), jnp.bfloat16),
        jnp.asarray(f(t, LKV), jnp.bfloat16),
        jnp.asarray(f(t, ROPE), jnp.bfloat16),
        jnp.asarray(f(t, IDX_H, IDX_D), jnp.bfloat16),
        jnp.asarray(f(t, IDX_D), jnp.bfloat16),
        jnp.asarray(np.abs(f(t, IDX_H)), jnp.float32),
    )


def _close(a, b, what):
    err = np.abs(a - b).max()
    scale = np.abs(b).max()
    print(f"[dsa-dcp] {what}: max|dcp4-dcp1|={err:.3e} (max|out|={scale:.3e})")
    assert err <= 3e-2 * max(scale, 1e-3), what


def test_dsa_dcp4_matches_dcp1():
    mesh = Mesh(np.array(jax.local_devices()[:4]).reshape(1, 4), ("data", "tensor"))
    rng = np.random.default_rng(20261030)
    s1, s4 = Setup(mesh, 1), Setup(mesh, 4)

    # 1) extend from scratch: 6 pages, exactly 1 page (ranks 1-3 own nothing), 8 pages
    seq_lens = [700, 128, 1000]
    for st in (s1, s4):
        loc = st.alloc.alloc_extend([0, 0, 0], seq_lens, [-1, -1, -1], sum(seq_lens))
        c = np.cumsum([0] + seq_lens)
        st.tokens = [list(loc[c[i] : c[i + 1]]) for i in range(3)]
        st.loc = loc
    pos = np.concatenate([np.arange(n) for n in seq_lens])
    x = _inputs(rng, sum(seq_lens))
    o1 = _step(mesh, s1, ForwardMode.EXTEND, seq_lens, seq_lens, s1.loc, pos, x)
    o4 = _step(mesh, s4, ForwardMode.EXTEND, seq_lens, seq_lens, s4.loc, pos, x)
    _close(o4, o1, "extend")

    # 2) decode steps; request 1 crosses into its second page (owned by rank 1)
    for step in range(3):
        seq_lens = [n + 1 for n in seq_lens]
        for st in (s1, s4):
            last = [t[-1] for t in st.tokens]
            st.loc = st.alloc.alloc_decode(seq_lens, last)
            for i in range(3):
                st.tokens[i].append(int(st.loc[i]))
        x = _inputs(rng, 3)
        pos = np.array(seq_lens) - 1
        o1 = _step(mesh, s1, ForwardMode.DECODE, seq_lens, None, s1.loc, pos, x)
        o4 = _step(mesh, s4, ForwardMode.DECODE, seq_lens, None, s4.loc, pos, x)
        _close(o4, o1, f"decode step {step}")

    # 3) chunked extend over cached prefixes
    new = [300, 64, 129]
    prefix = list(seq_lens)
    seq_lens = [p + n for p, n in zip(prefix, new)]
    for st in (s1, s4):
        last = [t[-1] for t in st.tokens]
        st.loc = st.alloc.alloc_extend(prefix, seq_lens, last, sum(new))
        c = np.cumsum([0] + new)
        for i in range(3):
            st.tokens[i].extend(int(v) for v in st.loc[c[i] : c[i + 1]])
    pos = np.concatenate([np.arange(p, p + n) for p, n in zip(prefix, new)])
    x = _inputs(rng, sum(new))
    o1 = _step(mesh, s1, ForwardMode.EXTEND, seq_lens, new, s1.loc, pos, x)
    o4 = _step(mesh, s4, ForwardMode.EXTEND, seq_lens, new, s4.loc, pos, x)
    _close(o4, o1, "chunked extend")
