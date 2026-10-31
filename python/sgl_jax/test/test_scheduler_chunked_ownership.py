import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.prefill import PrefillBootstrapQueue
from sgl_jax.srt.managers.io_struct import AbortReq, PauseGenerationReqInput
from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sgl_jax.srt.managers.schedule_policy import AddReqResult
from sgl_jax.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestSchedulerChunkedOwnership(unittest.TestCase):
    def _make_req(self, rid, fill_ids, prefix_indices, dp_rank=0, return_logprob=False):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=fill_ids,
            sampling_params=SamplingParams(max_new_tokens=8),
            dp_rank=dp_rank,
            eos_token_ids={2},
            vocab_size=100,
            return_logprob=return_logprob,
        )
        req.fill_ids = fill_ids
        req.prefix_indices = prefix_indices
        req.req_pool_idx = 6
        return req

    def _make_batch(self, reqs, chunked_reqs):
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            model_config=SimpleNamespace(vocab_size=100),
            enable_overlap=False,
            dp_size=len(reqs),
            chunked_reqs=chunked_reqs,
            mesh=None,
            spec_algorithm=None,
        )
        batch.forward_mode = SimpleNamespace(is_extend=lambda: True)
        batch.per_dp_bs_size = max((len(rank_reqs) for rank_reqs in reqs), default=0)
        return batch

    def _make_scheduler(self, reqs, active_reqs=None, batch_owners=None):
        if isinstance(reqs, Req):
            reqs = [[reqs]]
        dp_size = len(reqs)
        if active_reqs is None:
            active_reqs = [None] * dp_size
        elif isinstance(active_reqs, Req):
            active_reqs = [active_reqs]
        if batch_owners is None:
            batch_owners = [rank_reqs[0] if rank_reqs else None for rank_reqs in reqs]

        cached = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_size = dp_size
        scheduler.pd = ""
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.chunked_reqs = active_reqs
        scheduler._pending_chunked_abort_reqs = [None] * dp_size
        scheduler.last_batch = self._make_batch(reqs, batch_owners)
        scheduler.cur_batch = scheduler.last_batch
        scheduler.running_batch = self._make_batch([[] for _ in range(dp_size)], [None] * dp_size)
        scheduler.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda cached_req: cached.append(cached_req)
        )
        scheduler.spec_algorithm = None
        scheduler.is_generation = True
        scheduler.is_hybrid = False
        scheduler.is_mixed_chunk = False
        scheduler.waiting_queue = []
        scheduler.grammar_queue = []
        scheduler.disagg_prefill_queue = None
        scheduler.disagg_prealloc_queue = None
        scheduler.disagg_transfer_queue = None
        scheduler._pd_pending_bootstrap = []
        scheduler._pending_h2d = []
        scheduler._comm_backend = None
        scheduler.policy = SimpleNamespace(calc_priority=Mock())
        scheduler.page_size = 1
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.new_token_ratio = 0.5
        scheduler.max_prefill_tokens = 8
        scheduler.chunked_prefill_size = 2
        scheduler.lora_paths = None
        scheduler.send_to_tokenizer = SimpleNamespace(send_pyobj=Mock())
        scheduler.send_to_detokenizer = SimpleNamespace(send_pyobj=Mock())
        scheduler._release_prefill_host_buffer = Mock()
        scheduler.set_next_batch_sampling_info_done = Mock()
        scheduler.stream_output = Mock()
        scheduler.skip_tokenizer_init = False
        scheduler.stream_interval = 1
        return scheduler, cached

    def _make_prefill_result(self, next_token_ids, input_token_logprobs=None):
        logits_output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=input_token_logprobs,
            hidden_states=None,
        )
        batch_size = len(next_token_ids)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            extend_input_len_per_req=[2] * batch_size,
            extend_logprob_start_len_per_req=[0] * batch_size,
            cache_miss_count=0,
            bid=1,
        )

    def test_parked_chunk_without_new_kv_is_not_cached_again(self):
        req = self._make_req("parked", [1, 2], [10, 11])
        scheduler, cached = self._make_scheduler(req, active_reqs=req)

        excluded = scheduler._prepare_chunked_reqs_to_exclude()

        self.assertIs(excluded[0], req)
        self.assertEqual(cached, [])

    def test_conflicting_chunked_owner_is_rejected(self):
        batch_req = self._make_req("batch", [1, 2, 3], [10, 11])
        active_req = self._make_req("active", [4, 5, 6], [12, 13])
        scheduler, _ = self._make_scheduler(batch_req, active_reqs=active_req)

        with self.assertRaisesRegex(AssertionError, "Chunked request mismatch"):
            scheduler._prepare_chunked_reqs_to_exclude()

    def test_failed_pd_chunk_clears_batch_owner_before_terminal_release(self):
        req = self._make_req("failed", [1, 2, 3], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler._pending_chunked_abort_reqs[0] = req

        scheduler._retire_chunk_producer_ownership(req)

        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)

        scheduler._sync_chunked_req_owners()
        self.assertEqual(scheduler.chunked_reqs, [None])

    def test_terminal_chunk_failure_retires_owner_before_resource_release(self):
        req = self._make_req("failed-read", [1, 2, 3], [10, 11])
        req.bootstrap_room = 91
        sender = SimpleNamespace(
            poll=Mock(return_value=KVPoll.FAILED),
            clear=Mock(),
            failure_exception=Mock(),
        )
        req.disagg_chunk_sender = sender
        scheduler, cached = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)
        scheduler._pending_chunked_abort_reqs[0] = req

        def assert_ownership_retired(released_req):
            self.assertIs(released_req, req)
            self.assertIsNone(scheduler.chunked_reqs[0])
            self.assertIsNone(scheduler._pending_chunked_abort_reqs[0])
            self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)

        scheduler._release_prefill_req_resources = Mock(side_effect=assert_ownership_retired)

        scheduler._on_prefill_transfer_terminal(req, sender)

        scheduler._sync_chunked_req_owners()
        self.assertEqual(scheduler._prepare_chunked_reqs_to_exclude(), {})
        self.assertEqual(cached, [])
        scheduler._release_prefill_req_resources.assert_called_once_with(req)
        sender.clear.assert_called_once_with()
        self.assertIsNone(req.disagg_chunk_sender)

    def test_restoring_one_dp_rank_preserves_the_other_rank_owner(self):
        req0 = self._make_req("rank-0", [1, 2, 3], [10, 11], dp_rank=0)
        req1 = self._make_req("rank-1", [4, 5, 6], [12, 13], dp_rank=1)
        scheduler, cached = self._make_scheduler([[req0], [req1]], active_reqs=[None, req1])

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertEqual(scheduler.chunked_reqs, [req0, req1])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None, None])
        self.assertIs(excluded[0], req0)
        self.assertIs(excluded[1], req1)
        self.assertTrue(scheduler.last_batch.is_empty())
        self.assertEqual(req0.req_pool_idx, 6)
        self.assertEqual(req1.req_pool_idx, 6)
        self.assertEqual(cached, [req0, req1])

    def test_final_chunk_stays_in_batch_without_creating_an_owner(self):
        req = self._make_req("final", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req, batch_owners=[None])

        excluded = scheduler._prepare_chunked_reqs_to_exclude()
        scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        self.assertEqual(excluded, {})
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [req])
        self.assertEqual(cached, [])

    def test_abort_releases_parked_chunk_and_clears_owner(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, cached = self._make_scheduler(req)

        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler.abort_request(AbortReq(rid=req.rid))
            self.assertIs(scheduler.chunked_reqs[0], req)
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            release.assert_not_called()

            excluded = scheduler._prepare_chunked_reqs_to_exclude()
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            self.assertEqual(cached, [])
            scheduler._process_pending_chunked_aborts()
            scheduler.last_batch.filter_batch(chunked_req_to_exclude=excluded)

        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        sent = scheduler.send_to_tokenizer.send_pyobj.call_args.args[0]
        self.assertEqual(sent.rid, req.rid)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])

    def test_abort_defers_pd_chunk_release_until_sender_is_terminal(self):
        req = self._make_req("pd-sending", [1, 2, 3], [10, 11])
        sender = SimpleNamespace(
            abort=Mock(),
            poll=Mock(return_value=KVPoll.FAILED),
            clear=Mock(),
        )
        req.disagg_chunk_sender = sender
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)
        scheduler.disagg_prefill_queue = SimpleNamespace(
            cancel_matching=lambda *_args: [SimpleNamespace(req_id=req.rid, sender=sender, req=req)]
        )

        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler.abort_request(AbortReq(rid=req.rid))
            consumed = scheduler._process_pending_chunked_aborts()

        sender.abort.assert_called_once_with()
        self.assertIsNotNone(req.to_finish)
        release.assert_not_called()
        scheduler._release_prefill_host_buffer.assert_not_called()
        self.assertEqual(consumed, {0: req})
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.send_to_tokenizer.send_pyobj.call_count, 0)

        scheduler._release_prefill_req_resources = Mock()
        scheduler._on_prefill_transfer_terminal(req, sender)

        scheduler._release_prefill_req_resources.assert_called_once_with(req)
        sender.clear.assert_called_once_with()
        self.assertTrue(req.finished())
        self.assertIsNone(req.disagg_chunk_sender)
        scheduler.stream_output.assert_called_once()

    def test_pause_retires_consumed_chunk_owner_from_batch(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.abort_request(AbortReq(rid=req.rid))

        with patch("sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])
        self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)
        scheduler._sync_chunked_req_owners()
        self.assertEqual(scheduler.chunked_reqs, [None])

    def test_abort_consumes_parked_chunk_while_engine_is_paused(self):
        req = self._make_req("parked", [1, 2, 3], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler.abort_request(AbortReq(rid=req.rid))

        release.assert_called_once()
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertTrue(req.finished())
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        self.assertEqual(scheduler.last_batch.reqs_info[0].reqs, [])
        self.assertIsNone(scheduler.last_batch.reqs_info[0].chunked_req)
        self.assertEqual(scheduler.send_to_tokenizer.send_pyobj.call_count, 1)

    def test_retract_leaves_active_pd_transfer_queues_draining(self):
        req = self._make_req("pd-retract", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req)
        scheduler.disagg_prefill_queue = Mock()
        scheduler.disagg_transfer_queue = Mock()

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache"):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.disagg_prefill_queue.retract_all.assert_not_called()
        scheduler.disagg_transfer_queue.retract_all.assert_not_called()

    def test_retract_keeps_prefill_chunk_draining_to_terminal(self):
        req = self._make_req("pd-retract-drain", [1, 2, 3], [10, 11])
        req.bootstrap_room = 92
        req.disagg_transfer_id = "wire-pd-retract-drain"

        class _Sender:
            def __init__(self):
                self.state = KVPoll.TRANSFERRING
                self.clear = Mock()

            def poll(self):
                return self.state

        sender = _Sender()
        req.disagg_chunk_sender = sender
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)
        scheduler.disagg_prefill_queue = PrefillBootstrapQueue()
        scheduler._release_prefill_req_resources = Mock()
        scheduler.disagg_prefill_queue.add(
            req.rid,
            sender,
            req=req,
            on_terminal=lambda: scheduler._on_prefill_transfer_terminal(req, sender),
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.disagg_prefill_queue), 1)
        self.assertEqual(req.disagg_transport_id, "wire-pd-retract-drain")

        scheduler.send_kv_chunk()
        scheduler._release_prefill_req_resources.assert_not_called()

        sender.state = KVPoll.SUCCESS
        scheduler.send_kv_chunk()

        self.assertEqual(len(scheduler.disagg_prefill_queue), 0)
        scheduler._release_prefill_req_resources.assert_called_once_with(req)
        self.assertIsNone(req.disagg_chunk_sender)
        self.assertIsNone(scheduler.chunked_reqs[0])
        self.assertEqual(req.disagg_transport_id, "wire-pd-retract-drain")

        scheduler.continue_generation(SimpleNamespace())
        self.assertFalse(scheduler._engine_paused)

    def test_chunk_budget_skips_only_exhausted_dp_for_current_round(self):
        for dp_size in (1, 2):
            with self.subTest(dp_size=dp_size):
                scheduler, _ = self._make_scheduler([[] for _ in range(dp_size)])
                scheduler.per_dp_max_running_requests = 256
                scheduler.tree_cache.supports_recurrent = lambda: False
                first = self._make_req("first", [1], [], dp_rank=0)
                skipped = self._make_req("skipped", [2], [], dp_rank=0)
                scheduler.waiting_queue = [first, skipped]
                if dp_size == 2:
                    scheduler.waiting_queue.append(self._make_req("other-dp", [3], [], dp_rank=1))
                for req in scheduler.waiting_queue:
                    req.init_next_round_input = Mock()
                adder = SimpleNamespace(
                    can_run_list={dp: [] for dp in range(dp_size)},
                    pending_h2d=[],
                    new_chunked_reqs=[None] * dp_size,
                    add_one_req=Mock(return_value=AddReqResult.DP_BUDGET_EXHAUSTED),
                )
                with patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder):
                    self.assertIsNone(scheduler.get_new_batch_prefill())
                    self.assertEqual(adder.add_one_req.call_count, dp_size)
                    skipped.init_next_round_input.assert_not_called()
                    if dp_size == 2:
                        scheduler.waiting_queue[-1].init_next_round_input.assert_called_once()
                    self.assertFalse(scheduler.running_batch.batch_is_full)
                    self.assertTrue(
                        all(not info.batch_is_full for info in scheduler.running_batch.reqs_info)
                    )
                    # Exhaustion is local to one pass; every queued request is retried
                    # when the next adder has budget again.
                    adder.add_one_req.reset_mock()
                    adder.add_one_req.return_value = AddReqResult.CONTINUE
                    self.assertIsNone(scheduler.get_new_batch_prefill())
                    self.assertEqual(adder.add_one_req.call_count, len(scheduler.waiting_queue))
                    skipped.init_next_round_input.assert_called_once()

    def test_pending_abort_chunk_is_not_rescheduled(self):
        req = self._make_req("inflight", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        req.to_finish = FINISH_ABORT()
        scheduler._pending_chunked_abort_reqs[0] = req
        req.init_next_round_input = Mock()
        adder = SimpleNamespace(
            can_run_list={0: []},
            pending_h2d=[],
            new_chunked_reqs=[None],
            add_chunked_req=Mock(),
        )

        with patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder):
            self.assertIsNone(scheduler.get_new_batch_prefill())

        req.init_next_round_input.assert_not_called()
        adder.add_chunked_req.assert_not_called()
        self.assertIs(scheduler.chunked_reqs[0], req)

    def test_pending_abort_owner_is_not_counted_when_other_ranks_schedule(self):
        """A chunked owner held back by a pending abort has no chunk in this batch: its count
        stays at the one chunk still in flight while another rank's request is scheduled, so
        that chunk's result finalizes the abort (is_chunked back to 0)."""
        req = self._make_req("inflight", [1, 2, 3, 4], [10, 11], dp_rank=0)
        other = self._make_req("other", [5, 6], [], dp_rank=1)
        scheduler, _ = self._make_scheduler([[req], []], active_reqs=[req, None])
        scheduler.per_dp_max_running_requests = 64
        scheduler.tree_cache.supports_recurrent = lambda: False
        req.is_chunked = 1  # the previous batch's chunk, result not processed yet
        req.to_finish = FINISH_ABORT()
        scheduler._pending_chunked_abort_reqs[0] = req
        req.init_next_round_input = Mock()
        other.init_next_round_input = Mock()
        scheduler.waiting_queue = [other]
        adder = SimpleNamespace(
            can_run_list={0: [], 1: []},
            pending_h2d=[],
            new_chunked_reqs=[None, None],
            add_chunked_req=Mock(),
        )

        def add_one_req(r):
            adder.can_run_list[r.dp_rank].append(r)
            return AddReqResult.CONTINUE

        adder.add_one_req = Mock(side_effect=add_one_req)
        scheduler.log_prefill_stats = Mock()
        for name in ("req_to_token_pool", "model_config", "mesh", "embedding_pool"):
            setattr(scheduler, name, None)
        with (
            patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder),
            patch(
                "sgl_jax.srt.managers.scheduler.ScheduleBatch.init_new",
                side_effect=_StopAfterScheduling,
            ),
            self.assertRaises(_StopAfterScheduling),
        ):
            scheduler.get_new_batch_prefill()
        adder.add_chunked_req.assert_not_called()
        self.assertEqual(adder.can_run_list, {0: [], 1: [other]})
        self.assertEqual(req.is_chunked, 1)
        self.assertIs(scheduler.chunked_reqs[0], req)

    def test_abort_finalizes_when_inflight_overlap_chunk_returns(self):
        req = self._make_req("inflight", [1, 2], [10, 11], return_logprob=True)
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.last_batch.return_logprob = True
        scheduler.enable_overlap = True
        req.is_chunked = 1
        result = self._make_prefill_result([3])
        scheduler.tp_worker = SimpleNamespace(
            resolve_last_batch_result=Mock(return_value=(result.logits_output, [3], 0))
        )

        with (
            patch(
                "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
            ) as release,
            patch(
                "sgl_jax.srt.managers.scheduler_output_processor_mixin._complete_precision_trace"
            ) as complete_trace,
        ):
            scheduler.abort_request(AbortReq(rid=req.rid))
            self.assertIsNotNone(req.to_finish)
            self.assertIs(scheduler._pending_chunked_abort_reqs[0], req)
            release.assert_not_called()
            complete_trace.assert_not_called()

            scheduler._process_pending_chunked_aborts()
            release.assert_not_called()

            SchedulerOutputProcessorMixin.process_batch_result_prefill(
                scheduler, scheduler.last_batch, result
            )

        self.assertTrue(req.finished())
        self.assertIsNone(req.to_finish)
        self.assertEqual(req.is_chunked, 0)
        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])
        skip_reqs = scheduler.stream_output.call_args.args[3]
        self.assertNotIn(id(req), skip_reqs)
        self.assertTrue(req.input_logprob_sent)

        SchedulerOutputProcessorMixin.stream_output_generation(
            scheduler,
            [req],
            return_logprob=True,
            return_output_logprob_only=False,
        )

        sent = scheduler.send_to_detokenizer.send_pyobj.call_args.args[0]
        self.assertEqual(sent.rids, [req.rid])
        self.assertEqual(sent.input_token_logprobs_val, [[]])
        self.assertEqual(sent.input_token_logprobs_idx, [[]])
        complete_trace.assert_called_once_with(req)

    def test_aborted_dp_chunk_advances_logprob_cursor_for_later_rank(self):
        aborted_req = self._make_req("aborted", [1, 2], [10], dp_rank=0, return_logprob=True)
        live_req = self._make_req("live", [3, 4], [11], dp_rank=1, return_logprob=True)
        batch = self._make_batch([[aborted_req], [live_req]], [aborted_req, live_req])
        for info in batch.reqs_info:
            info.extend_lens = [2]
            info.extend_logprob_start_lens = [0]

        scheduler, _ = self._make_scheduler(
            [[aborted_req], [live_req]], active_reqs=[aborted_req, live_req]
        )
        aborted_req.is_chunked = 1
        aborted_req.to_finish = FINISH_ABORT()
        scheduler._pending_chunked_abort_reqs[0] = aborted_req
        live_req.is_chunked = 1
        result = self._make_prefill_result([5, 6], input_token_logprobs=(0.1, 0.2, 0.3, 0.4))

        with patch("sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"):
            SchedulerOutputProcessorMixin.process_batch_result_prefill(scheduler, batch, result)

        self.assertEqual(live_req.input_token_logprobs, [0.3, 0.4])

    def test_pd_prefill_result_makes_middle_chunk_abort_releasable(self):
        req = self._make_req("pd-inflight", [1, 2], [10], return_logprob=False)
        req.bootstrap_room = 0
        req.is_chunked = 1
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)

        scheduler.process_prefill_chunk(scheduler.last_batch, SimpleNamespace())

        self.assertEqual(req.is_chunked, 0)
        scheduler.abort_request(AbortReq(rid=req.rid))
        with patch(
            "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as release:
            scheduler._process_pending_chunked_aborts()

        release.assert_called_once()
        self.assertTrue(req.finished())
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])

    def test_pd_continuation_preserves_committed_prefix_after_middle_result(self):
        req = self._make_req("pd-continuation", [1, 2, 3, 4], [10, 11])
        req.bootstrap_room = 7
        req.is_chunked = 0
        req.kv_committed_len = 2

        req.init_next_round_input(SimpleNamespace(root_node=object(), disable=True))

        self.assertEqual(list(req.prefix_indices), [10, 11])
        self.assertEqual(req.extend_input_len, 2)

    def test_pd_fresh_request_drops_uncommitted_stale_prefix(self):
        req = self._make_req("pd-fresh", [1, 2, 3, 4], [10, 11])
        req.bootstrap_room = 7
        req.kv_committed_len = 0

        req.init_next_round_input(SimpleNamespace(root_node=object(), disable=True))

        self.assertEqual(req.prefix_indices, [])
        self.assertEqual(req.extend_input_len, 4)

    def test_pd_raiden_handoff_streams_page_aligned_chunks_once(self):
        req = self._make_req("pd-chunks", [1, 2, 3, 4, 5, 6], [])
        req.bootstrap_room = 91
        req.disagg_transfer_id = "wire-pd-chunks"
        req.req_pool_idx = 0
        req.fill_ids = req.origin_input_ids[:4]
        req.extend_input_len = 4
        req.is_chunked = 1

        class _Sender:
            def __init__(self):
                self.calls = []
                self.has_pending_failure = False
                self.has_started_chunks = False

            def init(self, _indices, transfer_id=None):
                self.transfer_id = transfer_id

            def send_chunk(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                kwargs["on_ready"]()
                self.has_started_chunks = True

            def poll(self):
                return None

            def clear(self):
                return None

        class _Manager:
            host_pool = None

            def __init__(self):
                self.sender = _Sender()
                self.prepared = []

            def create_sender(self, _req_id):
                return self.sender

            def prepare_prefill_batch(self, kv_buffer):
                self.prepared.append(kv_buffer)

        class _Queue:
            def __init__(self):
                self._entries = {}

            def add(self, req_id, sender, on_terminal=None, req=None):
                del req
                self._entries[req_id] = (sender, on_terminal)

        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(
            enable_request_time_stats_logging=False,
            disaggregation_enable_chunk_prefill_transfer=True,
        )
        scheduler.disagg_kv_manager = _Manager()
        scheduler.disagg_prefill_queue = _Queue()
        kv_pool = SimpleNamespace(page_size=2, kv_buffer=object())
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: kv_pool)
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=np.asarray([[8, 9, 20, 21, 30, 31]], dtype=np.int32)
        )

        scheduler.process_prefill_chunk(scheduler.last_batch, SimpleNamespace())

        sender = scheduler.disagg_kv_manager.sender
        self.assertEqual(sender.transfer_id, "wire-pd-chunks")
        self.assertEqual(sender.calls[0][0], (0, [4, 10]))
        self.assertEqual(sender.calls[0][1]["chunk_page_offset"], 0)
        self.assertEqual(sender.calls[0][1]["expected_total_pages"], 3)
        self.assertFalse(sender.calls[0][1]["is_final"])
        self.assertEqual(req.start_send_idx, 4)
        self.assertEqual(req.is_chunked, 0)
        self.assertEqual(len(scheduler.disagg_prefill_queue._entries), 1)

        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = 2
        scheduler.chunked_reqs = [None]
        scheduler.process_prefill_chunk(scheduler.last_batch, SimpleNamespace())

        self.assertEqual(sender.calls[1][0], (1, [15]))
        self.assertEqual(sender.calls[1][1]["chunk_page_offset"], 2)
        self.assertEqual(sender.calls[1][1]["expected_total_pages"], 3)
        self.assertTrue(sender.calls[1][1]["is_final"])
        self.assertEqual(req.start_send_idx, 6)
        self.assertEqual(req.disagg_chunk_index, 2)
        self.assertEqual(len(scheduler.disagg_prefill_queue._entries), 1)
        self.assertEqual(len(scheduler.disagg_kv_manager.prepared), 2)

    def test_retract_requeues_parked_chunk_after_releasing_allocation(self):
        req = self._make_req("parked", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache") as release:
            scheduler._retract_parked_chunked_reqs([])

        release.assert_called_once_with(
            req,
            scheduler.tree_cache,
            is_insert=False,
            allow_overallocated=False,
        )
        scheduler._release_prefill_host_buffer.assert_called_once_with(req)
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertTrue(req.is_retracted)
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])

    def test_retract_preserves_active_pd_chunk_sender_and_wire_id(self):
        req = self._make_req("pd-retract", [1, 2], [10, 11])
        req.disagg_transfer_id = "wire-pd-retract"
        sender = SimpleNamespace(
            abort=Mock(),
            poll=Mock(return_value=KVPoll.FAILED),
            clear=Mock(),
        )
        req.disagg_chunk_sender = sender
        scheduler, _ = self._make_scheduler(req, active_reqs=req)
        scheduler.pd = "prefill"
        scheduler.server_args = SimpleNamespace(enable_request_time_stats_logging=False)

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache") as release:
            scheduler._retract_parked_chunked_reqs([])

        sender.abort.assert_not_called()
        release.assert_not_called()
        self.assertFalse(req.is_retracted)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler.chunked_reqs, [req])
        self.assertEqual(req.disagg_transport_id, "wire-pd-retract")
        sender.clear.assert_not_called()
        scheduler.stream_output.assert_not_called()

    def test_retract_does_not_release_batch_owned_chunk_twice(self):
        req = self._make_req("active", [1, 2], [10, 11])
        scheduler, _ = self._make_scheduler(req, active_reqs=req)

        with patch("sgl_jax.srt.managers.scheduler.release_kv_cache") as release:
            scheduler._retract_parked_chunked_reqs([req])

        release.assert_not_called()
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler._pending_chunked_abort_reqs, [None])

    def _make_steal_scheduler(self, home_cached, chunked_home=True):
        # Two DP ranks: rank 0 is busy with a chunked prefill, rank 1 has nothing to prefill.
        scheduler, _ = self._make_scheduler([[], []])
        scheduler.per_dp_max_running_requests = 32
        scheduler._dp_steal_max_cached = 4096
        scheduler._dp_route_log = False
        chunk = self._make_req("chunk", [1, 2, 3], [], dp_rank=0)
        chunk.init_next_round_input = Mock()
        scheduler.chunked_reqs = [chunk if chunked_home else None, None]
        req = self._make_req("cold", [5, 6, 7, 8], [], dp_rank=0)
        req.init_next_round_input = Mock()
        req.match_key = Mock(return_value="key")
        scheduler.waiting_queue = [req]
        scheduler.tree_cache = SimpleNamespace(
            match_prefix=Mock(return_value=SimpleNamespace(device_indices=[0] * home_cached)),
            supports_recurrent=lambda: False,
        )
        adder = SimpleNamespace(
            can_run_list={0: [], 1: []},
            new_chunked_reqs=[None, None],
            pending_h2d=[],
            add_chunked_req=Mock(return_value=chunk),
        )

        def add_one_req(r):
            adder.can_run_list[r.dp_rank].append(r)
            return AddReqResult.CONTINUE

        adder.add_one_req = Mock(side_effect=add_one_req)
        return scheduler, adder, req

    def test_idle_dp_rank_steals_a_cold_request_from_a_busy_rank(self):
        scheduler, adder, req = self._make_steal_scheduler(home_cached=128)
        scheduler._steal_for_idle_dp_ranks(adder, set())
        self.assertEqual(req.dp_rank, 1)
        req.init_next_round_input.assert_called_once_with(scheduler.tree_cache)
        self.assertEqual(adder.can_run_list, {0: [], 1: [req]})

    def test_warm_request_is_not_stolen(self):
        scheduler, adder, req = self._make_steal_scheduler(home_cached=8192)
        scheduler._steal_for_idle_dp_ranks(adder, set())
        self.assertEqual(req.dp_rank, 0)
        adder.add_one_req.assert_not_called()

    def test_no_steal_when_the_home_rank_can_start_the_request(self):
        scheduler, adder, req = self._make_steal_scheduler(home_cached=0, chunked_home=False)
        scheduler._steal_for_idle_dp_ranks(adder, set())
        self.assertEqual(req.dp_rank, 0)
        adder.add_one_req.assert_not_called()

    def test_no_steal_into_a_rank_whose_chunk_budget_is_exhausted(self):
        scheduler, adder, req = self._make_steal_scheduler(home_cached=0)
        scheduler._steal_for_idle_dp_ranks(adder, {1})
        self.assertEqual(req.dp_rank, 0)
        adder.add_one_req.assert_not_called()

    def test_failed_steal_restores_the_home_rank(self):
        scheduler, adder, req = self._make_steal_scheduler(home_cached=0)
        adder.add_one_req = Mock(return_value=AddReqResult.DP_BUDGET_EXHAUSTED)
        scheduler._steal_for_idle_dp_ranks(adder, set())
        self.assertEqual(req.dp_rank, 0)
        self.assertEqual(adder.can_run_list, {0: [], 1: []})

    def test_get_new_batch_prefill_steals_only_when_enabled(self):
        for limit in (None, 4096):
            with self.subTest(limit=limit):
                scheduler, adder, req = self._make_steal_scheduler(home_cached=0)
                scheduler._dp_steal_max_cached = limit
                scheduler._steal_for_idle_dp_ranks = Mock()
                with patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder):
                    self.assertIsNone(scheduler.get_new_batch_prefill())
                if limit is None:
                    scheduler._steal_for_idle_dp_ranks.assert_not_called()
                else:
                    scheduler._steal_for_idle_dp_ranks.assert_called_once_with(adder, set())
                # The busy home rank never admitted it in the normal pass.
                adder.add_one_req.assert_not_called()


class _StopAfterScheduling(Exception):
    pass


class TestPrefillShortestFirst(unittest.TestCase):
    """SGLANG_JAX_PREFILL_SJF: shortest-first queue order and parking of long chunked prefills."""

    _make_req = TestSchedulerChunkedOwnership._make_req
    _make_batch = TestSchedulerChunkedOwnership._make_batch
    _make_scheduler = TestSchedulerChunkedOwnership._make_scheduler

    def _sjf_scheduler(self, dp_size=1):
        from sgl_jax.srt.managers.scheduler import _PrefillSjfConfig

        scheduler, _ = self._make_scheduler([[] for _ in range(dp_size)])
        scheduler._sjf = _PrefillSjfConfig(
            short_tokens=4096, park_min_remaining=8192, aging_tokens_per_step=100, max_park_steps=50
        )
        scheduler.parked_chunked_reqs = [[] for _ in range(dp_size)]
        scheduler.per_dp_max_running_requests = 64
        scheduler.forward_ct = 0
        scheduler._dp_route_log = False
        return scheduler

    def _queued(self, scheduler, rid, prompt_len, cached, dp_rank=0):
        req = self._make_req(rid, list(range(prompt_len)), [], dp_rank=dp_rank)
        req._dp_match_len = cached
        req._sjf_enqueue_ct = scheduler.forward_ct
        req.init_next_round_input = Mock()
        return req

    def _long_chunk(self, prompt_len=30000, done=2048, dp_rank=0):
        req = self._make_req("long", list(range(prompt_len)), list(range(done)), dp_rank=dp_rank)
        req.init_next_round_input = Mock()
        return req

    def test_router_match_reaches_the_queued_request(self):
        """The DP router records its prefix match on the intake request; the scheduler's Req
        built from it keeps the match, so shortest-first estimates uncached tokens."""
        scheduler = self._sjf_scheduler()
        scheduler.model_config = SimpleNamespace(hf_eos_token_id={2}, vocab_size=100)
        scheduler.tokenizer = None
        scheduler.server_args = SimpleNamespace(
            disaggregation_enable_overlap_schedule=False, allow_auto_truncate=False
        )
        scheduler.max_req_input_len = 1 << 20
        scheduler.max_req_len = 1 << 20
        scheduler.grammar_backend = None
        scheduler.forward_ct = 3
        recv = SimpleNamespace(
            rid="warm",
            text="",
            input_ids=list(range(5000)),
            sampling_params=SamplingParams(max_new_tokens=8),
            radix_input_ids=None,
            return_logprob=False,
            return_output_logprob_only=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            extra_key=None,
            dp_rank=0,
            return_routed_experts=False,
            return_hidden_states=False,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            disagg_transfer_id=None,
            logprob_start_len=-1,
            _dp_match_len=4608,
        )
        scheduler.handle_generate_request(recv)
        (req,) = scheduler.waiting_queue
        self.assertEqual(scheduler._sjf_new_tokens(req), 392)
        self.assertEqual(req._sjf_enqueue_ct, 3)

    def test_queue_order_is_shortest_new_prompt_first_with_aging(self):
        scheduler = self._sjf_scheduler()
        cold = self._queued(scheduler, "cold", 20000, 0)  # 20000 new tokens
        scheduler.forward_ct = 150
        warm = self._queued(scheduler, "warm", 60000, 59000)  # 1000 new tokens
        mid = self._queued(scheduler, "mid", 9000, 0)
        queue = [cold, mid, warm]
        queue.sort(key=scheduler._sjf_priority)
        # cold has aged 150 steps (-15000): 5000 < mid's 9000.
        self.assertEqual([r.rid for r in queue], ["warm", "cold", "mid"])

    def test_long_chunk_parks_for_a_short_request_and_resumes_when_drained(self):
        scheduler = self._sjf_scheduler()
        long = self._long_chunk()
        scheduler.chunked_reqs = [long]
        short = self._queued(scheduler, "short", 50000, 48000)
        scheduler.waiting_queue = [short]

        scheduler._sjf_park_and_resume()
        self.assertEqual(scheduler.chunked_reqs, [None])
        self.assertEqual(scheduler.parked_chunked_reqs, [[long]])

        scheduler.forward_ct = 7
        scheduler.waiting_queue = []  # the short request was admitted and finished
        scheduler._sjf_park_and_resume()
        self.assertIs(scheduler.chunked_reqs[0], long)
        self.assertEqual(scheduler.parked_chunked_reqs, [[]])
        self.assertEqual(long._sjf_parked_steps, 7)

    def test_no_parking_for_a_nearly_done_chunk_or_a_not_much_shorter_request(self):
        for prompt_len, done, short_new in ((10000, 4000, 1000), (30000, 22000, 4000)):
            with self.subTest(prompt_len=prompt_len, done=done, short_new=short_new):
                scheduler = self._sjf_scheduler()
                long = self._long_chunk(prompt_len, done)
                scheduler.chunked_reqs = [long]
                scheduler.waiting_queue = [self._queued(scheduler, "short", short_new, 0)]
                scheduler._sjf_park_and_resume()
                self.assertIs(scheduler.chunked_reqs[0], long)
                self.assertEqual(scheduler.parked_chunked_reqs, [[]])

    def test_no_parking_when_the_short_request_cannot_start(self):
        scheduler = self._sjf_scheduler()
        long = self._long_chunk()
        scheduler.chunked_reqs = [long]
        scheduler.waiting_queue = [self._queued(scheduler, "short", 1000, 0)]
        scheduler.running_batch.reqs_info[0].batch_is_full = True
        scheduler._sjf_park_and_resume()
        self.assertIs(scheduler.chunked_reqs[0], long)

    def test_parked_request_resumes_after_its_park_budget(self):
        scheduler = self._sjf_scheduler()
        long = self._long_chunk()
        scheduler.chunked_reqs = [long]
        scheduler.waiting_queue = [self._queued(scheduler, "short", 1000, 0)]
        scheduler._sjf_park_and_resume()
        self.assertEqual(scheduler.parked_chunked_reqs, [[long]])
        scheduler.forward_ct = 50
        scheduler._sjf_park_and_resume()  # short still waiting, but 50 steps parked
        self.assertIs(scheduler.chunked_reqs[0], long)
        scheduler._sjf_park_and_resume()  # and it is not parked again
        self.assertIs(scheduler.chunked_reqs[0], long)

    def test_aborted_parked_request_resumes_as_a_pending_chunked_abort(self):
        scheduler = self._sjf_scheduler()
        long = self._long_chunk()
        scheduler.parked_chunked_reqs = [[long]]
        long._sjf_parked_at = 0
        scheduler.waiting_queue = [self._queued(scheduler, "short", 1000, 0)]
        scheduler.abort_request(AbortReq(rid="long"))
        self.assertIsNotNone(long.to_finish)
        self.assertEqual(scheduler.waiting_queue[0].rid, "short")  # only the parked one matched
        scheduler._sjf_park_and_resume()
        self.assertIs(scheduler.chunked_reqs[0], long)
        self.assertIs(scheduler._pending_chunked_abort_reqs[0], long)

    def test_parked_requests_keep_the_scheduler_busy(self):
        scheduler = self._sjf_scheduler()
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.pending_dp_reqs = []
        long = self._long_chunk()
        scheduler.parked_chunked_reqs = [[long]]
        can_flush, msg = scheduler._can_flush_cache()
        self.assertFalse(can_flush)
        self.assertIn("chunked=True", msg)
        self.assertEqual([r for _, r in scheduler._iter_dp_requests()], [long])

    def _adder(self, dp_size=1, admit=False):
        adder = SimpleNamespace(
            can_run_list={dp: [] for dp in range(dp_size)},
            pending_h2d=[],
            new_chunked_reqs=[None] * dp_size,
            add_chunked_req=Mock(side_effect=lambda r: r),
        )

        def add_one_req(r):
            if admit:
                adder.can_run_list[r.dp_rank].append(r)
            return AddReqResult.CONTINUE

        adder.add_one_req = Mock(side_effect=add_one_req)
        return adder

    def test_get_new_batch_prefill_admits_the_short_request_instead_of_the_chunk(self):
        scheduler = self._sjf_scheduler()
        scheduler.tree_cache.supports_recurrent = lambda: False
        long = self._long_chunk()
        scheduler.chunked_reqs = [long]
        short = self._queued(scheduler, "short", 50000, 48000)
        aged_long = self._queued(scheduler, "aged-long", 40000, 0)
        aged_long._sjf_enqueue_ct = -1000  # sorts first, but may not start on a parked rank
        scheduler.waiting_queue = [aged_long, short]
        adder = self._adder(admit=True)
        scheduler.log_prefill_stats = Mock()
        for name in ("req_to_token_pool", "model_config", "mesh", "embedding_pool"):
            setattr(scheduler, name, None)
        with (
            patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder),
            patch(
                "sgl_jax.srt.managers.scheduler.ScheduleBatch.init_new",
                side_effect=_StopAfterScheduling,
            ),
            self.assertRaises(_StopAfterScheduling),
        ):
            scheduler.get_new_batch_prefill()
        adder.add_chunked_req.assert_not_called()
        self.assertEqual(adder.can_run_list, {0: [short]})
        self.assertEqual(scheduler.parked_chunked_reqs, [[long]])
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ["aged-long"])
        long.init_next_round_input.assert_not_called()

    def test_parked_chunk_resumes_in_the_same_round_when_the_short_request_cannot_start(self):
        scheduler = self._sjf_scheduler()
        scheduler.tree_cache.supports_recurrent = lambda: False
        long = self._long_chunk()
        scheduler.chunked_reqs = [long]
        scheduler.waiting_queue = [self._queued(scheduler, "short", 1000, 0)]
        adder = self._adder(admit=False)
        with patch("sgl_jax.srt.managers.scheduler.PrefillAdder", return_value=adder):
            self.assertIsNone(scheduler.get_new_batch_prefill())
        adder.add_chunked_req.assert_called_once_with(long)
        long.init_next_round_input.assert_called_once_with()
        self.assertIs(scheduler.chunked_reqs[0], long)
        self.assertEqual(scheduler.parked_chunked_reqs, [[]])


if __name__ == "__main__":
    unittest.main()
