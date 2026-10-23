"""Per-intake DP routing load, independent of batch admission and device execution."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from sgl_jax.srt.managers.dp_schedule_policy import (
    pick_cache_aware_dp,
    pick_force_cache_aware_dp,
    pick_shape_aware_dp,
    req_prefix_match_key,
)


@dataclass(frozen=True)
class DpLoadSnapshot:
    """Assigned unfinished work and per-rank routing limits for one intake.

    A zero limit marks a rank full. Otherwise the limit retains the scheduler's
    per-DP request cap, applied to backlog plus assignments earlier this round.
    """

    request_counts: tuple[int, ...]
    input_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    request_limits: tuple[int, ...]

    @classmethod
    def collect(
        cls,
        dp_size: int,
        requests,
        estimate_io,
        request_limits: tuple[int, ...],
    ) -> DpLoadSnapshot:
        counts = [0] * dp_size
        inputs = [0] * dp_size
        outputs = [0] * dp_size
        # Batch copies retain the same Req objects. Use object identity, not a
        # user-supplied rid, to avoid conflating distinct live requests.
        seen: dict[int, int] = {}
        for rank, req in requests:
            if not 0 <= rank < dp_size:
                raise ValueError(f"Invalid DP rank {rank}")
            identity = id(req)
            if identity in seen:
                if seen[identity] != rank:
                    raise ValueError("One request is owned by multiple DP ranks")
                continue
            seen[identity] = rank
            if req.finished():
                continue
            in_tokens, out_tokens = estimate_io(req)
            counts[rank] += 1
            inputs[rank] += in_tokens
            outputs[rank] += out_tokens
        return cls(tuple(counts), tuple(inputs), tuple(outputs), request_limits)


logger = logging.getLogger(__name__)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


# Sticky cache affinity for the ``cache_aware`` policy (off unless set). A request is
# deferred instead of routed to an available rank when a full rank caches more than
# SGLANG_JAX_DP_MAX_SPILL_TOKENS more of its prompt (see ``pick_cache_aware_dp``). After
# SGLANG_JAX_DP_MAX_DEFER_ROUNDS deferrals it may spill again. A round is one intake call,
# made once per scheduler iteration in lockstep on every node, so the count is identical
# across nodes and routing stays deterministic.
_DEFAULT_MAX_DEFER_ROUNDS = 500


class DpRouter:
    """Assign requests using mutable load counters private to one intake round.

    The initial snapshot stays immutable. Each successful assignment updates
    the counters before the next request is considered. Execution admission
    remains the scheduler's responsibility; this object never reads its queues.
    """

    def __init__(
        self,
        snapshot: DpLoadSnapshot,
        policy: str,
        estimate_io,
        lookup_prefix,
        round_robin_start: int = 0,
    ):
        self.request_counts = list(snapshot.request_counts)
        self.input_tokens = list(snapshot.input_tokens)
        self.output_tokens = list(snapshot.output_tokens)
        self.request_limits = snapshot.request_limits
        self.dp_size = len(self.request_counts)
        self.policy = policy
        self._estimate_io = estimate_io
        self._lookup_prefix = lookup_prefix
        self.round_robin_counter = round_robin_start
        self.max_spill_tokens = (
            _env_int("SGLANG_JAX_DP_MAX_SPILL_TOKENS") if policy == "cache_aware" else None
        )
        max_defer = _env_int("SGLANG_JAX_DP_MAX_DEFER_ROUNDS")
        self.max_defer_rounds = _DEFAULT_MAX_DEFER_ROUNDS if max_defer is None else max_defer

    def assign(self, req) -> int | None:
        """Choose or preserve a DP rank, then record its load exactly once.

        Single-DP and round-robin routing do not need load estimates. A failed
        selection leaves counters untouched so the caller can defer the request.
        """
        if self.dp_size == 1:
            req.dp_rank = 0
            return 0

        rank = req.dp_rank
        if rank is not None and not 0 <= rank < self.dp_size:
            logger.warning(
                "Ignoring invalid dp_rank=%s for request %s; reassigning with %s policy",
                rank,
                req.rid,
                self.policy,
            )
            rank = None
            req.dp_rank = None

        if self.policy == "round_robin":
            if rank is None:
                rank = self.round_robin_counter % self.dp_size
                self.round_robin_counter += 1
            req.dp_rank = rank
            return rank

        eligible = self._eligible_ranks() if rank is None else []
        if rank is None and not eligible:
            return None

        input_tokens, output_tokens = self._estimate_io(req)
        if rank is None:
            if self.policy in ("cache_aware", "force_cache_aware"):
                rank = self._pick_cache_aware(req, eligible, input_tokens, output_tokens)
            elif self.policy == "shape_aware":
                rank = pick_shape_aware_dp(
                    eligible,
                    self.input_tokens,
                    self.output_tokens,
                    input_tokens,
                    output_tokens,
                )
            else:
                # Keep the existing min_running_queue CLI spelling compatible.
                rank = self._pick_least_loaded(eligible)

        if rank is not None:
            req.dp_rank = rank
            self.request_counts[rank] += 1
            self.input_tokens[rank] += input_tokens
            self.output_tokens[rank] += output_tokens
        return rank

    def _eligible_ranks(self) -> list[int]:
        return [
            rank
            for rank in range(self.dp_size)
            if self.request_counts[rank] < self.request_limits[rank]
        ]

    def _pick_least_loaded(self, eligible: list[int]) -> int:
        return min(
            eligible,
            key=lambda rank: (
                self.request_counts[rank],
                self.input_tokens[rank] + self.output_tokens[rank],
                rank,
            ),
        )

    def _pick_cache_aware(
        self, req, eligible: list[int], input_tokens: int, output_tokens: int
    ) -> int | None:
        token_ids, extra_key = req_prefix_match_key(req)
        deferrals = getattr(req, "_dp_defer_rounds", 0)
        sticky = self.max_spill_tokens is not None and deferrals < self.max_defer_rounds
        matches = {}
        if token_ids:
            # Force-cache-aware must see full holders too: defer instead of
            # silently choosing a smaller cache hit on an available rank.
            # Sticky cache-aware probes them to measure the cost of spilling.
            probe_ranks = (
                range(self.dp_size) if self.policy == "force_cache_aware" or sticky else eligible
            )
            for rank in probe_ranks:
                matches[rank] = self._lookup_prefix(token_ids, extra_key, rank)
        prompt_len = len(token_ids) if token_ids else 0
        loads = [i + o for i, o in zip(self.input_tokens, self.output_tokens)]
        if self.policy == "force_cache_aware":
            return pick_force_cache_aware_dp(
                eligible,
                self.request_counts,
                loads,
                matches,
                prompt_len,
                self.input_tokens,
                self.output_tokens,
                input_tokens,
                output_tokens,
            )
        rank = pick_cache_aware_dp(
            eligible,
            self.request_counts,
            loads,
            matches,
            prompt_len,
            self.input_tokens,
            self.output_tokens,
            input_tokens,
            output_tokens,
            max_spill_tokens=self.max_spill_tokens if sticky else None,
        )
        if rank is not None:
            # Diagnostics: longest cached prefix on any probed rank, on the chosen rank, deferrals.
            req._dp_route_info = (max(matches.values(), default=0), matches.get(rank, 0), rank, deferrals)
        if rank is None and sticky:
            req._dp_defer_rounds = deferrals + 1
            if req._dp_defer_rounds == self.max_defer_rounds:
                logger.info(
                    "DP router: request %s deferred %d rounds for its cache holder; "
                    "it may now spill to another rank",
                    req.rid,
                    req._dp_defer_rounds,
                )
        return rank
