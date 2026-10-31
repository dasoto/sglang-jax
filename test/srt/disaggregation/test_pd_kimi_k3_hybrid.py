"""Unit tests for Kimi-K3 hybrid (MLA DCP>1 + KDA recurrent state) PD disaggregation."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sgl_jax.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
    _globalize_rank_local_page_ids,
)
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def _make_mesh(*, explicit: bool = True) -> Mesh:
    devices = np.array(jax.devices()[:4]).reshape(2, 2)
    axis_type = jax.sharding.AxisType.Explicit if explicit else jax.sharding.AxisType.Auto
    return Mesh(devices, ("data", "tensor"), axis_types=(axis_type, axis_type))


def _make_hybrid_pools(mesh: Mesh, *, page_size: int = 8):
    dp_size = 2
    dcp_size = 2
    # 16 pages per (dp, dcp) shard => size = 16 * page_size * dp_size * dcp_size = 512 tokens
    size = 16 * page_size * dp_size * dcp_size
    kv_pool = HybridLinearKVPool(
        size=size,
        page_size=page_size,
        dtype=jnp.bfloat16,
        full_attention_layer_ids=[1, 3],
        mesh=mesh,
        token_to_kv_pool_class=MLATokenToKVPool,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        dp_size=dp_size,
        dcp_size=dcp_size,
    )
    rec_pool = RecurrentStatePool(
        size=16,
        num_heads=4,
        head_dim=16,
        num_k_heads=4,
        head_k_dim=16,
        conv_kernel_size=4,
        linear_recurrent_layer_ids=[0, 2],
        mesh=mesh,
        dp_size=dp_size,
    )
    req_pool = HybridReqToTokenPool(
        size=8,
        max_context_len=128,
        dtype=np.int32,
        recurrent_state_pool=rec_pool,
        dp_size=dp_size,
        enable_recurrent_extra_buffer=True,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        kvcache=kv_pool,
        dp_size=dp_size,
    )
    tree_cache = ChunkCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
    )
    return kv_pool, rec_pool, req_pool, allocator, tree_cache


class _DummyHybridScheduler(
    SchedulerDisaggregationPrefillMixin,
    SchedulerDisaggregationDecodeMixin,
):
    def __init__(self, mesh: Mesh):
        (
            self.kv_pool,
            self.rec_pool,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
        ) = _make_hybrid_pools(mesh)
        self.mesh = mesh
        self.dp_size = 2
        self.page_size = self.kv_pool.page_size
        self.enable_overlap = False
        self.spec_algorithm = None
        self.embedding_pool = None
        self.per_dp_max_running_requests = 2
        self.model_config = SimpleNamespace(vocab_size=1024)
        self.server_args = SimpleNamespace(
            disaggregation_enable_overlap_schedule=False,
            enable_request_time_stats_logging=False,
            disaggregation_mode="decode",
        )
        self.streamed: list[tuple[str, list[int]]] = []
        self.waiting_queue = []
        self.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(self.dp_size)],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            dp_size=self.dp_size,
            spec_algorithm=None,
            mesh=self.mesh,
        )

    def stream_output(self, reqs, return_logprob, return_output_logprob_only):
        for r in reqs:
            self.streamed.append((r.rid, list(r.output_ids)))


def test_mla_dcp_and_kda_roundtrip_and_pure_decode_admission():
    mesh = _make_mesh()
    prefill_sched = _DummyHybridScheduler(mesh)
    decode_sched = _DummyHybridScheduler(mesh)

    # Create a prefill request on dp_rank=1 spanning 3 pages (20 tokens, page_size=8)
    prompt_ids = list(range(10, 30))
    sp = SamplingParams(max_new_tokens=4, temperature=0.0)
    p_req = Req(
        rid="req-k3-1",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=sp,
        dp_rank=1,
    )
    p_req.sampling_params.normalize(None)
    prefill_sched.req_to_token_pool.alloc([p_req])
    p_kv_indices = prefill_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=1)
    assert p_kv_indices is not None
    prefill_sched.req_to_token_pool.write(
        (p_req.req_pool_idx, slice(0, 20)), p_kv_indices[:20].astype(np.int32)
    )
    p_req._pd_first_token_id = 777

    # Populate distinct data in P's MLA KV pages and KDA recurrent state slot
    p_page_ids = p_kv_indices[:24:8] // 8
    p_global_pages = _globalize_rank_local_page_ids(
        p_page_ids,
        dp_rank=1,
        dp_size=2,
        total_pages=int(prefill_sched.kv_pool.kv_buffer[0].shape[0]),
        dcp_size=2,
    )
    for l_idx in range(prefill_sched.kv_pool.layer_num):
        buf = np.asarray(jax.device_get(prefill_sched.kv_pool.kv_buffer[l_idx])).copy()
        for k, gp in enumerate(p_global_pages):
            buf[gp] = (l_idx + 1) * 10 + (k + 1)
        prefill_sched.kv_pool.kv_buffer[l_idx] = jax.device_put(
            jnp.asarray(buf, dtype=jnp.bfloat16), prefill_sched.kv_pool.kv_sharding
        )

    p_gslot = prefill_sched.rec_pool.global_slot_index(1, p_req.recurrent_pool_idx)
    for l_idx in range(prefill_sched.rec_pool.num_linear_recurrent_layers):
        t_np = np.asarray(jax.device_get(prefill_sched.rec_pool.recurrent_buffers[l_idx])).copy()
        c_np = np.asarray(jax.device_get(prefill_sched.rec_pool.conv_buffers[l_idx][0])).copy()
        t_np[p_gslot] = (l_idx + 1) * 3.5
        c_np[p_gslot] = (l_idx + 1) * -1.25
        prefill_sched.rec_pool.recurrent_buffers[l_idx] = jax.device_put(
            jnp.asarray(t_np), prefill_sched.rec_pool.recurrent_sharding
        )
        prefill_sched.rec_pool.conv_buffers[l_idx][0] = jax.device_put(
            jnp.asarray(c_np), prefill_sched.rec_pool.conv_sharding
        )

    # Extract payload in simulated multi-host mode (process_count=2) to test local-mesh pack/unpack
    with patch("jax.process_count", return_value=2):
        payload = prefill_sched._build_prefill_payload(p_req)
        d_req = Req(
            rid="req-k3-1",
            origin_input_text="",
            origin_input_ids=prompt_ids,
            sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
            dp_rank=0,
        )
        d_req.sampling_params.normalize(None)
        specs = decode_sched._build_kv_spec_for_req(d_req)
        assert set(specs.keys()) == {"kv", "kda_temporal", "kda_conv", "first_token"}
        for key in specs:
            assert payload[key].shape == specs[key].shape
            assert payload[key].dtype == specs[key].dtype
            assert payload[key].is_fully_addressable

        # Allocate destination pages on D's dp_rank=0 (different DP rank than P's dp_rank=1!)
        d_kv_indices = decode_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=0)
        assert d_kv_indices is not None
        decode_sched._install_received_kv(d_req, d_kv_indices, payload)

    assert d_req._pd_first_token_id == 777
    assert d_req.recurrent_pool_idx is not None

    # Verify MLA KV pages written on D's dp_rank=0 match P's dp_rank=1 pages bit-for-bit
    d_page_ids = d_kv_indices[:24:8] // 8
    d_global_pages = _globalize_rank_local_page_ids(
        d_page_ids,
        dp_rank=0,
        dp_size=2,
        total_pages=int(decode_sched.kv_pool.kv_buffer[0].shape[0]),
        dcp_size=2,
    )
    for l_idx in range(decode_sched.kv_pool.layer_num):
        p_buf = np.asarray(jax.device_get(prefill_sched.kv_pool.kv_buffer[l_idx]))
        d_buf = np.asarray(jax.device_get(decode_sched.kv_pool.kv_buffer[l_idx]))
        np.testing.assert_array_equal(d_buf[d_global_pages], p_buf[p_global_pages])

    # Verify KDA temporal and conv states on D's dp_rank=0 match P's dp_rank=1 slot bit-for-bit
    d_gslot = decode_sched.rec_pool.global_slot_index(0, d_req.recurrent_pool_idx)
    for l_idx in range(decode_sched.rec_pool.num_linear_recurrent_layers):
        p_t = np.asarray(jax.device_get(prefill_sched.rec_pool.recurrent_buffers[l_idx]))[p_gslot]
        d_t = np.asarray(jax.device_get(decode_sched.rec_pool.recurrent_buffers[l_idx]))[d_gslot]
        np.testing.assert_array_equal(d_t, p_t)
        p_c = np.asarray(jax.device_get(prefill_sched.rec_pool.conv_buffers[l_idx][0]))[p_gslot]
        d_c = np.asarray(jax.device_get(decode_sched.rec_pool.conv_buffers[l_idx][0]))[d_gslot]
        np.testing.assert_array_equal(d_c, p_c)

    # Verify initially empty running_batch does NOT have vacuous True flags from all([])
    assert not decode_sched.running_batch.is_prefill_only
    assert not decode_sched.running_batch.return_output_logprob_only

    # Enqueue and drain pure decode into running_batch
    decode_sched._enqueue_hybrid_pure_decode(d_req, d_kv_indices, d_req._pd_first_token_id)
    assert decode_sched.streamed == [("req-k3-1", [777])]
    assert decode_sched.waiting_queue == []
    decode_sched._drain_ready_pure_decode()
    assert len(decode_sched._pd_ready_pure_decode_queue) == 0
    assert len(decode_sched.running_batch.reqs_info[0].reqs) == 1
    assert not decode_sched.running_batch.is_prefill_only
    assert not decode_sched.running_batch.return_output_logprob_only

    # Run prepare_for_decode and verify invariants for next decode step
    decode_sched.running_batch.prepare_for_decode()
    info0 = decode_sched.running_batch.reqs_info[0]
    assert decode_sched.running_batch.forward_mode == ForwardMode.DECODE
    assert info0.input_ids.tolist() == [777]
    assert info0.seq_lens.tolist() == [21]
    assert info0.recurrent_indices.tolist() == [d_req.recurrent_pool_idx]
    assert info0.out_cache_loc.tolist() == [int(d_kv_indices[20])]
    info0.output_ids = np.array([778], dtype=np.int32)

    # Admit a second pure-decode request onto the same dp_rank=0, merge, finish req 1, and filter
    d_req2 = Req(
        rid="req-k3-2",
        origin_input_text="world",
        origin_input_ids=list(range(100, 112)),
        sampling_params=SamplingParams(max_new_tokens=8, temperature=0.0),
        dp_rank=0,
    )
    d_req2.sampling_params.normalize(None)
    d_req2.fill_ids = list(d_req2.origin_input_ids)
    d_req2.bootstrap_room = 987654
    d_kv_indices2 = decode_sched.token_to_kv_pool_allocator.alloc(16, dp_rank=0)
    assert d_kv_indices2 is not None
    decode_sched._enqueue_hybrid_pure_decode(d_req2, d_kv_indices2, 888)
    decode_sched._drain_ready_pure_decode()
    assert len(decode_sched.running_batch.reqs_info[0].reqs) == 2
    assert decode_sched.running_batch.reqs_info[0].top_logprobs_nums is None

    # Finish the first request and verify filter_batch succeeds without IndexError
    from sgl_jax.srt.managers.schedule_batch import FINISH_LENGTH

    d_req.finished_reason = FINISH_LENGTH(8)
    decode_sched.running_batch.filter_batch()
    assert len(decode_sched.running_batch.reqs_info[0].reqs) == 1
    assert decode_sched.running_batch.reqs_info[0].reqs[0].rid == "req-k3-2"


def test_multihost_sync_63bit_bootstrap_room_without_x64():
    """Verify synced_terminal_rooms preserves 63-bit rooms when jax_enable_x64 is False."""
    from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
    from sgl_jax.srt.disaggregation.common.multihost_sync import synced_terminal_rooms
    from sgl_jax.srt.disaggregation.mini_lb_helpers import generate_bootstrap_room

    for _ in range(100):
        r = generate_bootstrap_room()
        assert 0 <= r < 2**31

    big_room_ok = (1 << 62) | 0xDEADBEEF
    big_room_fail = (1 << 61) | 0xCAFEBABE
    entries = [
        SimpleNamespace(room=big_room_ok, state=KVPoll.SUCCESS),
        SimpleNamespace(room=big_room_fail, state=KVPoll.FAILED),
    ]
    success, failed = synced_terminal_rooms(
        entries,
        poll_fn=lambda e: e.state,
        room_fn=lambda e: e.room,
    )
    assert success == {big_room_ok}
    assert failed == {big_room_fail}


def test_pd_prefill_chunked_unified_radix_cache():
    """Verify multi-chunk PD prefill with UnifiedRadixCache preserves last_node and hits prefix cache."""
    from sgl_jax.srt.mem_cache.common import release_kv_cache
    from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    mesh = _make_mesh()
    prefill_sched = _DummyHybridScheduler(mesh)
    tree_cache = UnifiedRadixCache(
        req_to_token_pool=prefill_sched.req_to_token_pool,
        token_to_kv_pool_allocator=prefill_sched.token_to_kv_pool_allocator,
        page_size=8,
        disable=False,
    )
    prefill_sched.tree_cache = tree_cache

    req1 = Req(
        rid="pd-chunk-1",
        origin_input_text="hello world",
        origin_input_ids=list(range(10, 42)),  # 32 tokens = 4 pages of 8
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=0,
    )
    req1.sampling_params.normalize(None)
    req1.bootstrap_room = 111222

    # Round 1: initial admission with tree_cache (16-token chunk = 2 pages)
    req1.init_next_round_input(tree_cache)
    assert req1.last_node is tree_cache.root_node
    lock1 = tree_cache.inc_lock_ref(req1.last_node)
    req1.cache_lock_params = lock1.to_dec_params()
    req1.extend_input_len = 16
    req1.fill_ids = req1.origin_input_ids[:16]
    b1 = ScheduleBatch.init_new(
        reqs=[[req1], []],
        req_to_token_pool=prefill_sched.req_to_token_pool,
        token_to_kv_pool_allocator=prefill_sched.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=prefill_sched.model_config,
        enable_overlap=False,
        dp_size=2,
        mesh=mesh,
    )
    b1.prepare_for_extend()
    tree_cache.cache_unfinished_req(req1)
    saved_node = req1.last_node
    assert saved_node is not None and saved_node is not tree_cache.root_node
    assert len(req1.prefix_indices) == 16

    # Round 2: continuation round (scheduler calls init_next_round_input() with tree_cache=None)
    req1.init_next_round_input()
    assert req1.last_node is saved_node
    assert req1.extend_input_len == 16
    req1.fill_ids = req1.origin_input_ids[:32]
    # Allocate only the second chunk's KV pages without re-allocating req_pool_idx
    kv_chunk2 = prefill_sched.token_to_kv_pool_allocator.alloc(16, dp_rank=0)
    prefill_sched.req_to_token_pool.write((req1.req_pool_idx, slice(16, 32)), kv_chunk2)
    req1.kv_committed_len = 32
    req1.kv_allocated_len = 32
    # Verify a second cache_unfinished_req (e.g. if there were a 3rd chunk) does NOT crash on None last_node
    tree_cache.cache_unfinished_req(req1)
    assert req1.last_node is not None and req1.last_node is not tree_cache.root_node
    release_kv_cache(req1, tree_cache)

    # Request 2 shares the first 24 tokens with req1 -> should hit 24 tokens in UnifiedRadixCache
    req2 = Req(
        rid="pd-chunk-2",
        origin_input_text="hello world again",
        origin_input_ids=list(range(10, 34)) + [99, 100, 101, 102, 103, 104, 105, 106],
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=0,
    )
    req2.sampling_params.normalize(None)
    req2.bootstrap_room = 333444
    req2.init_next_round_input(tree_cache)
    assert len(req2.prefix_indices) == 24
    assert req2.extend_input_len == 8
    assert req2.last_node is not None and req2.last_node is not tree_cache.root_node


def test_mla_kv_multi_block_gather_and_write():
    """Verify >32-page (64-page) multi-block fori_loop MLA KV gather & write bit-exactness."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers
    from sgl_jax.srt.mem_cache.memory_pool import write_all_mla_kv_layers

    mesh = _make_mesh()
    dp_size = 2
    dcp_size = 2
    page_size = 8
    # 40 pages per (dp, dcp) shard => 160 pages total
    size = 40 * page_size * dp_size * dcp_size
    pools = [
        MLATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=jnp.bfloat16,
            layer_num=3,
            mesh=mesh,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            dp_size=dp_size,
            dcp_size=dcp_size,
        )
        for _ in range(2)
    ]
    p_pool, d_pool = pools
    rng = np.random.default_rng(777)
    for l_idx in range(3):
        arr = jnp.asarray(rng.standard_normal(p_pool.kv_buffer[l_idx].shape), dtype=jnp.bfloat16)
        p_pool.kv_buffer[l_idx] = jax.device_put(arr, p_pool.kv_sharding)

    # 30 valid pages (global 41..70), padded to 64 pages (> the 32-page block).
    valid_pages = 30
    padded_pages = 64
    p_global_pages = np.arange(41, 41 + valid_pages, dtype=np.int32)
    padded_src = np.zeros(padded_pages, dtype=np.int32)
    padded_src[:valid_pages] = p_global_pages

    out_sharding = NamedSharding(mesh, P("data", "tensor", None, None))
    idx_sharding = NamedSharding(mesh, P(None))
    dev_idx = jax.device_put(jnp.asarray(padded_src, dtype=jnp.int32), idx_sharding)
    gathered = _jit_gather_all_layers(
        list(p_pool.kv_buffer),
        dev_idx,
        out_sharding,
        mla_buf_sharding=p_pool.kv_sharding,
    )

    # Write to global pages 3..32; padding targets an out-of-range page and is dropped.
    total_pages = int(d_pool.kv_buffer[0].shape[0])
    d_global_pages = np.arange(3, 3 + valid_pages, dtype=np.int32)
    padded_dst = np.full(padded_pages, total_pages, dtype=np.int32)
    padded_dst[:valid_pages] = d_global_pages
    dev_dst = jax.device_put(jnp.asarray(padded_dst, dtype=jnp.int32), idx_sharding)
    before = [np.asarray(jax.device_get(b)) for b in d_pool.kv_buffer]
    d_pool.kv_buffer[:] = write_all_mla_kv_layers(gathered, dev_dst, list(d_pool.kv_buffer))

    untouched = np.setdiff1d(np.arange(total_pages), d_global_pages)
    for l_idx in range(3):
        p_np = np.asarray(jax.device_get(p_pool.kv_buffer[l_idx]))
        d_np = np.asarray(jax.device_get(d_pool.kv_buffer[l_idx]))
        np.testing.assert_array_equal(d_np[d_global_pages], p_np[p_global_pages])
        np.testing.assert_array_equal(d_np[untouched], before[l_idx][untouched])


class _DummyDsaScheduler(
    SchedulerDisaggregationPrefillMixin,
    SchedulerDisaggregationDecodeMixin,
):
    """GLM-5 style DSA pools: MLA latent + indexer keys, DCP=2, no recurrent state."""

    def __init__(self, mesh: Mesh, *, page_size: int = 8, dp_size: int = 2, dcp_size: int = 2):
        size = 16 * page_size * dp_size * dcp_size
        self.kv_pool = MLATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=jnp.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=3,
            mesh=mesh,
            dp_size=dp_size,
            dcp_size=dcp_size,
            indexer_key_dim=128,
            num_indexer_layers=2,
        )
        self.req_to_token_pool = ReqToTokenPool(size=8, max_context_len=128, dtype=np.int32)
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            kvcache=self.kv_pool,
            dp_size=dp_size,
        )
        self.tree_cache = ChunkCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=page_size,
        )
        self.mesh = mesh
        self.dp_size = dp_size
        self.page_size = page_size
        self.enable_overlap = False
        self.spec_algorithm = None
        self.embedding_pool = None
        self.per_dp_max_running_requests = 2
        self.model_config = SimpleNamespace(vocab_size=1024)
        self.server_args = SimpleNamespace(
            disaggregation_enable_overlap_schedule=False,
            enable_request_time_stats_logging=False,
            disaggregation_mode="decode",
        )
        self.streamed: list[tuple[str, list[int]]] = []
        self.waiting_queue = []
        self.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(dp_size)],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            dp_size=dp_size,
            spec_algorithm=None,
            mesh=mesh,
        )

    def stream_output(self, reqs, return_logprob, return_output_logprob_only):
        for r in reqs:
            self.streamed.append((r.rid, list(r.output_ids)))


def _fill_pages(buffers, global_pages, sharding, base):
    for l_idx in range(len(buffers)):
        buf = np.asarray(jax.device_get(buffers[l_idx])).copy()
        for k, gp in enumerate(global_pages):
            buf[gp] = base + (l_idx + 1) * 10 + (k + 1)
        buffers[l_idx] = jax.device_put(jnp.asarray(buf, dtype=jnp.bfloat16), sharding)


def test_dsa_mla_dcp_roundtrip_with_indexer_and_first_token():
    """GLM-5 DSA: P ships latent KV, indexer keys and its first token; D starts decoding."""
    mesh = _make_mesh()
    prefill_sched = _DummyDsaScheduler(mesh)
    decode_sched = _DummyDsaScheduler(mesh)
    page_size = 8

    # 20 prompt tokens = 3 pages on P's dp_rank=1.
    prompt_ids = list(range(10, 30))
    p_req = Req(
        rid="req-glm5-1",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=1,
    )
    p_req.sampling_params.normalize(None)
    prefill_sched.req_to_token_pool.alloc([p_req])
    p_kv_indices = prefill_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=1)
    assert p_kv_indices is not None
    prefill_sched.req_to_token_pool.write(
        (p_req.req_pool_idx, slice(0, 20)), p_kv_indices[:20].astype(np.int32)
    )
    p_req._pd_first_token_id = 777

    p_pool = prefill_sched.kv_pool
    total_pages = int(p_pool.kv_buffer[0].shape[0])
    p_global_pages = _globalize_rank_local_page_ids(
        p_kv_indices[:24:page_size] // page_size,
        dp_rank=1,
        dp_size=2,
        total_pages=total_pages,
        dcp_size=2,
    )
    _fill_pages(p_pool.kv_buffer, p_global_pages, p_pool.kv_sharding, base=0)
    _fill_pages(p_pool.indexer_key_buffer, p_global_pages, p_pool.kv_sharding, base=100)

    d_req = Req(
        rid="req-glm5-1",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=0,
    )
    d_req.sampling_params.normalize(None)
    with patch("jax.process_count", return_value=2):
        payload = prefill_sched._build_prefill_payload(p_req)
        specs = decode_sched._build_kv_spec_for_req(d_req)
        assert set(specs) == {"kv", "idx", "first_token"}
        assert set(payload) == set(specs)
        for key in specs:
            assert payload[key].shape == specs[key].shape, key
            assert payload[key].dtype == specs[key].dtype, key
            assert payload[key].is_fully_addressable, key
        d_kv_indices = decode_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=0)
        assert d_kv_indices is not None
        decode_sched._install_received_kv(d_req, d_kv_indices, payload)

    assert d_req._pd_first_token_id == 777
    d_pool = decode_sched.kv_pool
    d_global_pages = _globalize_rank_local_page_ids(
        d_kv_indices[:24:page_size] // page_size,
        dp_rank=0,
        dp_size=2,
        total_pages=total_pages,
        dcp_size=2,
    )
    for p_bufs, d_bufs in (
        (p_pool.kv_buffer, d_pool.kv_buffer),
        (p_pool.indexer_key_buffer, d_pool.indexer_key_buffer),
    ):
        for l_idx in range(len(p_bufs)):
            p_np = np.asarray(jax.device_get(p_bufs[l_idx]))
            d_np = np.asarray(jax.device_get(d_bufs[l_idx]))
            np.testing.assert_array_equal(d_np[d_global_pages], p_np[p_global_pages])
            others = np.setdiff1d(np.arange(total_pages), d_global_pages)
            assert not np.any(d_np[others]), "transfer wrote outside the request's pages"

    # D streams P's token, then admits the request straight into decode.
    decode_sched._enqueue_hybrid_pure_decode(d_req, d_kv_indices, d_req._pd_first_token_id)
    assert decode_sched.streamed == [("req-glm5-1", [777])]
    assert decode_sched.waiting_queue == []
    decode_sched._drain_ready_pure_decode()
    assert decode_sched._pd_ready_pure_decode_queue == []
    assert len(decode_sched.running_batch.reqs_info[0].reqs) == 1
    decode_sched.running_batch.prepare_for_decode()
    info0 = decode_sched.running_batch.reqs_info[0]
    assert decode_sched.running_batch.forward_mode == ForwardMode.DECODE
    assert info0.input_ids.tolist() == [777]
    assert info0.seq_lens.tolist() == [21]
    assert info0.out_cache_loc.tolist() == [int(d_kv_indices[20])]


def test_dsa_first_token_opt_out(monkeypatch):
    """SGLANG_JAX_PD_FIRST_TOKEN_FROM_P=0 restores the D-side last-token recompute."""
    from sgl_jax.srt.disaggregation.prefill import _pd_sends_first_token

    sched = _DummyDsaScheduler(_make_mesh())
    assert _pd_sends_first_token(sched.kv_pool, sched.req_to_token_pool)
    monkeypatch.setenv("SGLANG_JAX_PD_FIRST_TOKEN_FROM_P", "0")
    assert not _pd_sends_first_token(sched.kv_pool, sched.req_to_token_pool)
    d_req = Req(
        rid="req-glm5-2",
        origin_input_text="",
        origin_input_ids=list(range(20)),
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=0,
    )
    specs = sched._build_kv_spec_for_req(d_req)
    assert set(specs) == {"kv", "idx"}


def test_dsa_wire_layout_between_different_dp_layouts(monkeypatch):
    """SGLANG_JAX_PD_WIRE: P (DP2 x DCP2) and D (DP4, no DCP) exchange the payload in
    the device-order wire layout; D's pages match P's for latent KV and indexer keys."""
    from sgl_jax.srt.disaggregation import prefill as prefill_mod

    monkeypatch.setattr(prefill_mod, "_PD_WIRE", True)
    monkeypatch.setattr(prefill_mod, "_PD_WIRE_MIN_PAGES", 4)
    monkeypatch.setattr(prefill_mod, "_WIRE_MESH", None)
    p_mesh = _make_mesh()
    d_devices = np.array(jax.devices()[:4]).reshape(4, 1)
    axis_type = jax.sharding.AxisType.Explicit
    d_mesh = Mesh(d_devices, ("data", "tensor"), axis_types=(axis_type, axis_type))
    prefill_sched = _DummyDsaScheduler(p_mesh)
    decode_sched = _DummyDsaScheduler(d_mesh, dp_size=4, dcp_size=1)
    page_size = 8

    prompt_ids = list(range(10, 30))
    p_req = Req(
        rid="req-glm5-wire",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=1,
    )
    p_req.sampling_params.normalize(None)
    prefill_sched.req_to_token_pool.alloc([p_req])
    p_kv_indices = prefill_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=1)
    prefill_sched.req_to_token_pool.write(
        (p_req.req_pool_idx, slice(0, 20)), p_kv_indices[:20].astype(np.int32)
    )
    p_req._pd_first_token_id = 555
    p_pool = prefill_sched.kv_pool
    p_global = _globalize_rank_local_page_ids(
        p_kv_indices[:24:page_size] // page_size,
        dp_rank=1,
        dp_size=2,
        total_pages=int(p_pool.kv_buffer[0].shape[0]),
        dcp_size=2,
    )
    _fill_pages(p_pool.kv_buffer, p_global, p_pool.kv_sharding, base=0)
    _fill_pages(p_pool.indexer_key_buffer, p_global, p_pool.kv_sharding, base=100)

    d_req = Req(
        rid="req-glm5-wire",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=SamplingParams(max_new_tokens=4, temperature=0.0),
        dp_rank=3,
    )
    d_req.sampling_params.normalize(None)
    with patch("jax.process_count", return_value=2):
        payload = prefill_sched._build_prefill_payload(p_req)
        specs = decode_sched._build_kv_spec_for_req(d_req)
        for key in specs:
            assert payload[key].shape == specs[key].shape, key
            assert payload[key].dtype == specs[key].dtype, key
        d_kv_indices = decode_sched.token_to_kv_pool_allocator.alloc(24, dp_rank=3)
        decode_sched._install_received_kv(d_req, d_kv_indices, payload)

    assert d_req._pd_first_token_id == 555
    d_pool = decode_sched.kv_pool
    d_global = _globalize_rank_local_page_ids(
        d_kv_indices[:24:page_size] // page_size,
        dp_rank=3,
        dp_size=4,
        total_pages=int(d_pool.kv_buffer[0].shape[0]),
        dcp_size=1,
    )
    for p_bufs, d_bufs in (
        (p_pool.kv_buffer, d_pool.kv_buffer),
        (p_pool.indexer_key_buffer, d_pool.indexer_key_buffer),
    ):
        for l_idx in range(len(p_bufs)):
            p_np = np.asarray(jax.device_get(p_bufs[l_idx]))
            d_np = np.asarray(jax.device_get(d_bufs[l_idx]))
            np.testing.assert_array_equal(d_np[d_global], p_np[p_global])


def test_wire_layout_accepts_more_prefill_dp_ranks(monkeypatch):
    import pytest

    from sgl_jax.srt.disaggregation import decode as decode_mod
    from sgl_jax.srt.disaggregation import prefill as prefill_mod

    req = SimpleNamespace(rid="r", dp_rank=1, disagg_prefill_dp_rank=5)
    monkeypatch.setattr(prefill_mod, "_PD_WIRE", False)
    with pytest.raises(ValueError):
        decode_mod._request_dp_rank(req, 4, prefill=True)
    monkeypatch.setattr(prefill_mod, "_PD_WIRE", True)
    assert decode_mod._request_dp_rank(req, 4, prefill=True) == 5
    with pytest.raises(ValueError):
        decode_mod._request_dp_rank(SimpleNamespace(rid="r", dp_rank=4), 4)
