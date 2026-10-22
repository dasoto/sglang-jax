"""Recurrent-state (KDA/GDN/GLA) component for UnifiedRadixCache.

State is one RecurrentStatePool slot per snapshot node. Leaves hold snapshots;
interior nodes keep theirs only when they are branch points or prompt-tail
checkpoints (extra-buffer path), and are LRU-evicted through
``aux_evictable_device_nodes``. Unary-chain nodes drop their slot so a single
long prefill does not keep one slot per chunk boundary.

The base path (page_size=1): finished-request commit + prefix-hit
copy-on-write clone. The extra-buffer path (page_size>=128) adds a
page-aligned ping-pong extra buffer and unfinished/fork donation."""

from __future__ import annotations

import heapq
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


def _node_dp_rank(node: UnifiedTreeNode) -> int:
    return node.key.dp_rank if node.key and node.key.dp_rank is not None else 0


class RecurrentComponent(TreeComponent):
    component_type = ComponentType.RECURRENT

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams | None = None):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(cache.req_to_token_pool, HybridReqToTokenPool), (
            f"RecurrentComponent requires HybridReqToTokenPool, "
            f"got {type(cache.req_to_token_pool)}"
        )
        if not getattr(cache, "enable_recurrent_extra_buffer", False):
            assert cache.page_size == 1, (
                f"RecurrentComponent requires page_size=1 when the extra buffer "
                f"is off, got {cache.page_size}"
            )
        super().__init__(cache, params)
        self.req_to_token_pool = cache.req_to_token_pool
        self.recurrent_state_pool = cache.req_to_token_pool.recurrent_state_pool
        self.enable_recurrent_extra_buffer = getattr(cache, "enable_recurrent_extra_buffer", False)
        # Prompt-tail checkpoints (extra-buffer path only): keep the snapshots at
        # the last N track boundaries of each request's prompt even after the
        # request's own decode extends the tree below them. Multi-turn clients
        # rarely resend the exact generated tokens (agentic trace replay, stripped
        # reasoning, re-rendered tool calls) and sometimes drop the tail of the
        # previous prompt, so the next turn diverges at or slightly before the
        # previous prompt end, never at the decode leaf. 0 restores the old
        # behavior (only leaves and branch points keep snapshots).
        self.prompt_tail_checkpoints = max(
            0, int(os.environ.get("SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS", "2"))
        )
        interval = getattr(cache, "recurrent_track_interval", None) or 0
        self.prompt_tail_window = max(0, self.prompt_tail_checkpoints - 1) * interval

    # ---- matching -----------------------------------------------------------

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        return lambda node: node.component_data[ct].value is not None

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list,
        best_value_len: int,
    ) -> MatchResult:
        if not params.cow_recurrent:
            return result
        req = params.req
        node = result.best_match_node
        if req is None or node is None:
            return result
        cd = node.component_data[self.component_type]
        if cd.value is not None:
            req.recurrent_cow_src_index = int(cd.value[0])
        return result

    # ---- insert / commit ----------------------------------------------------

    def _retain_checkpoint(self, node: UnifiedTreeNode | None) -> None:
        """Keep ``node``'s snapshot when it goes internal, like a branch point.

        The node stays an LRU eviction candidate (aux_evictable_device_nodes), so
        retention never blocks admission under recurrent-slot pressure.
        """
        if node is None or node is self.cache.root_node:
            return
        if node.component_data[self.component_type].value is None:
            return
        node._recurrent_branch_retained = True

    def _mark_prompt_tail_checkpoint(self, req: Req, is_finished: bool) -> None:
        """Retain ``req.last_node`` if it is one of the prompt-tail boundaries.

        ``req.last_node`` is the deepest snapshot node the request holds a lock on
        before this caching call. On the finished call it is the prompt-end
        checkpoint (the last prefill boundary, or the matched node when the
        prefill crossed none), which the decode insert is about to turn into a
        unary-chain node. On unfinished (chunk) calls it is the previous chunk
        boundary, kept when it lies within ``prompt_tail_window`` of the prompt
        end so a next turn that drops the tail of this prompt still hits.
        """
        if self.prompt_tail_checkpoints <= 0:
            return
        last_node = getattr(req, "last_node", None)
        if is_finished:
            self._retain_checkpoint(last_node)
            return
        if self.prompt_tail_window <= 0 or last_node is None:
            return
        fill_ids = getattr(req, "fill_ids", None)
        origin_input_ids = getattr(req, "origin_input_ids", None)
        if fill_ids is None or origin_input_ids is None:
            return
        target_len = len(origin_input_ids) + len(getattr(req, "output_ids", None) or ())
        if target_len - len(fill_ids) < self.prompt_tail_window:
            self._retain_checkpoint(last_node)

    def redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode):
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0

    def on_parent_gains_child(self, node: UnifiedTreeNode) -> None:
        # Unary-chain invariant: a single-child internal node drops its slot so
        # a single long request does not accumulate one slot per chunk boundary.
        # However, if the node is a shared branch point (>=2 children or
        # explicitly checkpointed on an existing internal node), retain its
        # snapshot so subsequent branches can reuse it.
        cd = node.component_data[self.component_type]
        if len(node.children) >= 2 or getattr(node, "_recurrent_branch_retained", False):
            node._recurrent_branch_retained = True
            return
        if cd.value is None or cd.lock_ref > 0:
            return
        self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), _node_dp_rank(node))
        self.cache.component_evictable_size_[self.component_type][_node_dp_rank(node)] -= 1
        cd.value = None
        self.cache._update_aux_evictable_node_sets(node)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        ct = self.component_type
        if params.recurrent_value is None:
            return
        cd = node.component_data[ct]
        if cd.value is not None:
            result.recurrent_exist = True
            result.recurrent_committed = False
            return
        if node.children:
            node._recurrent_branch_retained = True
        cd.value = params.recurrent_value
        self.cache.component_evictable_size_[ct][_node_dp_rank(node)] += 1
        self.cache._update_evictable_leaf_sets(node)
        self.cache._update_aux_evictable_node_sets(node)
        result.recurrent_committed = True

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int | None:
        if not self.enable_recurrent_extra_buffer:
            # Base path (page_size=1), finished-only donation: the running slot
            # already holds the final state (no copy); donating unfinished would
            # publish state not yet materialized.
            if not is_finished or req.recurrent_pool_idx is None:
                return None
            insert_params.recurrent_value = self.req_to_token_pool.recurrent_value_from_slot(
                req.recurrent_pool_idx
            )
            return None

        # Extra-buffer (page>=128): donate the materialized page-boundary snapshot
        # held by the KEEP ping-pong slot, capping the tree key at that boundary.
        self._mark_prompt_tail_checkpoint(req, is_finished)
        cache_len = req.recurrent_last_track_seqlen
        if cache_len is None:
            # No materialized boundary: nothing to donate; cleanup frees owned slots.
            return 0

        keep_idx = self.req_to_token_pool.get_recurrent_ping_pong_keep_idx(req)
        keep_slot = req.recurrent_ping_pong_track_buffer[keep_idx]
        if is_finished:
            # The keep slot IS the final boundary state; donate it directly. Cleanup
            # transfers ownership (do NOT remove it from the buffer here).
            insert_params.recurrent_value = self.req_to_token_pool.recurrent_value_from_slot(
                keep_slot
            )
            return cache_len

        # Unfinished: secure a fresh REPLACEMENT track slot FIRST, then donate the
        # keep slot. A mid-flight request must always retain two owned track slots,
        # so never donate without a secured replacement.
        dp = req.dp_rank if req.dp_rank is not None else 0
        replacement_slot = self.req_to_token_pool.alloc_recurrent_slot(dp)
        if replacement_slot is None:
            self.cache.evict(EvictParams(recurrent_num=1, dp_rank=dp))
            replacement_slot = self.req_to_token_pool.alloc_recurrent_slot(dp)
        if replacement_slot is None:
            # All candidates locked: skip donation (buffer + watermark intact).
            insert_params.recurrent_value = None
            return 0
        insert_params.recurrent_value = self.req_to_token_pool.donate_recurrent_ping_pong_slot(
            req, replacement_slot
        )
        return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: InsertResult | None = None,
        insert_params: InsertParams | None = None,
    ) -> None:
        committed = insert_result.recurrent_committed if insert_result is not None else False
        if not self.enable_recurrent_extra_buffer:
            # Base path: sole owner of the finished donate-vs-free decision.
            if not is_finished:
                return
            if committed:
                # Tree owns the running slot now (its content is the final state).
                self.req_to_token_pool.commit_to_tree(req)
            # else: leave req.recurrent_pool_idx set so release frees it.
            return

        # Extra-buffer: the donated value is the KEEP ping-pong slot, not the
        # running slot. Do NOT use commit_to_tree (page=1 running-slot donation).
        dp = req.dp_rank if req.dp_rank is not None else 0
        if is_finished:
            if committed:
                # Tree now owns the keep slot. Zero the keep position so
                # free_recurrent_cache skips it, then free running + the non-keep
                # track slot. Net: 3 req-owned → 1 tree-owned + 2 free.
                keep_idx = self.req_to_token_pool.get_recurrent_ping_pong_keep_idx(req)
                self.req_to_token_pool.set_recurrent_ping_pong_slot(req, keep_idx, 0)
                self.req_to_token_pool.free_recurrent_cache(req)
            else:
                # Nothing donated to the tree (duplicate/internal/no-insert): free
                # running + BOTH track slots.
                self.req_to_token_pool.free_recurrent_cache(req)
            return

        # Unfinished: the request stays live, keeping running + the other track
        # slot in the buffer (free nothing here).
        if committed:
            # Tree owns the donated slot; the buffer already holds the replacement.
            # No request-owned track slot represents that boundary anymore.
            req.recurrent_last_track_seqlen = None
        elif insert_params is not None and insert_params.recurrent_value is not None:
            # A donation was attempted but the tree rejected it: the donated slot
            # was swapped OUT of the buffer (replacement took its place) and is now
            # orphaned. Free it and invalidate the watermark.
            self.req_to_token_pool.free_recurrent_slot(int(insert_params.recurrent_value[0]), dp)
            req.recurrent_last_track_seqlen = None
        # else: donation SKIPPED (replacement alloc failed) → buffer + watermark
        # intact; the request keeps both track slots and may publish later.

    # ---- eviction -----------------------------------------------------------

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), _node_dp_rank(node))
            self.cache.component_evictable_size_[ct][_node_dp_rank(node)] -= 1
            cd.value = None
            node._recurrent_branch_retained = False
            freed = 1
        return freed, 0

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]) -> None:
        request = params.recurrent_num
        if request <= 0:
            return
        ct = self.component_type
        dp_rank = params.dp_rank
        candidates = self.cache.aux_evictable_device_nodes.get(
            ct, self.cache.evictable_device_leaves
        )
        heap = [
            x
            for x in candidates
            if self.node_has_component_data(x, EvictLayer.DEVICE)
        ]
        heapq.heapify(heap)
        while tracker[ct] < request and heap:
            x = heapq.heappop(heap)
            if not self.node_has_component_data(x, EvictLayer.DEVICE):
                continue
            if x.component_data[ct].lock_ref > 0:
                continue
            if dp_rank is not None and _node_dp_rank(x) != dp_rank:
                continue
            if x in self.cache.evictable_device_leaves:
                parent = x.parent
                self.cache._evict_device_leaf(x, tracker)
                if (
                    parent is not None
                    and parent is not self.cache.root_node
                    and parent in self.cache.evictable_device_leaves
                    and self.node_has_component_data(parent, EvictLayer.DEVICE)
                ):
                    heapq.heappush(heap, parent)
            else:
                freed, _ = self.evict_component(x, EvictLayer.DEVICE)
                tracker[ct] += freed
                self.cache._update_aux_evictable_node_sets(x)

    # ---- locking ------------------------------------------------------------

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.cache.root_node:
            return result
        cd = node.component_data[ct]
        if cd.value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result
        if cd.lock_ref == 0:
            node_dp_rank = _node_dp_rank(node)
            self.cache.component_evictable_size_[ct][node_dp_rank] -= 1
            self.cache.component_protected_size_[ct][node_dp_rank] += 1
        cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams | None,
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.cache.root_node:
            return
        skip = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip:
            return
        cd = node.component_data[ct]
        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                node_dp_rank = _node_dp_rank(node)
                is_branch = len(node.children) >= 2 or getattr(
                    node, "_recurrent_branch_retained", False
                )
                if cd.value is not None and node.children and not is_branch:
                    # Unary-chain node went internal while locked: deferred free
                    # from on_parent_gains_child.
                    self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), node_dp_rank)
                    self.cache.component_protected_size_[ct][node_dp_rank] -= 1
                    cd.value = None
                    cd.lock_ref = 0
                    return
                if is_branch:
                    node._recurrent_branch_retained = True
                self.cache.component_evictable_size_[ct][node_dp_rank] += 1
                self.cache.component_protected_size_[ct][node_dp_rank] -= 1
            cd.lock_ref -= 1
