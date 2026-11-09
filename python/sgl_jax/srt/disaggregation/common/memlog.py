"""Periodic device-memory log for PD servers, to tell leaked buffers from piling-up
programs: every SGLANG_JAX_PD_MEMLOG_S seconds (0 = off) process 0 logs device 0's
memory stats, the live arrays resident on it grouped by (shape, dtype), and the
number of loaded executables. SGLANG_JAX_MEMPEAK_SAMPLE_MS > 0 also polls device 0's
bytes_in_use from a thread at that period and logs the window's maximum (and the overall
maximum since the first log), which, unlike peak_bytes_in_use, excludes model loading.
SGLANG_JAX_PD_MEMLOG_GC=0 skips the gc.collect() before each log, and
SGLANG_JAX_PD_MEMLOG_TOP sets how many (shape, dtype) groups are listed (default 5)."""

from __future__ import annotations

import collections
import logging
import math
import os
import threading
import time

import jax

logger = logging.getLogger(__name__)

_PERIOD_S = float(os.environ.get("SGLANG_JAX_PD_MEMLOG_S", "0"))
_SAMPLE_S = float(os.environ.get("SGLANG_JAX_MEMPEAK_SAMPLE_MS", "0")) / 1000
_GC = os.environ.get("SGLANG_JAX_PD_MEMLOG_GC", "1") != "0"
_TOP = int(os.environ.get("SGLANG_JAX_PD_MEMLOG_TOP", "5"))
_last_log = 0.0
_prev_count: collections.Counter = collections.Counter()
# Sampler state: window max of bytes_in_use, window min of largest_free_block_bytes, overall max.
_win = {"max": 0, "min_free": None, "all": 0}
_sampler: threading.Thread | None = None


def _sample_loop(dev) -> None:
    while True:
        try:
            stats = dev.memory_stats() or {}
            used = stats.get("bytes_in_use", 0)
            if used > _win["max"]:
                _win["max"] = used
            if used > _win["all"]:
                _win["all"] = used
            free = stats.get("largest_free_block_bytes")
            if free is not None and (_win["min_free"] is None or free < _win["min_free"]):
                _win["min_free"] = free
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_SAMPLE_S)


def maybe_log_device_memory(tag: str) -> None:
    global _last_log, _prev_count, _sampler
    if _PERIOD_S <= 0 or jax.process_index() != 0:
        return
    if _SAMPLE_S > 0 and _sampler is None:
        _sampler = threading.Thread(
            target=_sample_loop, args=(jax.local_devices()[0],), daemon=True, name="mempeak"
        )
        _sampler.start()
    now = time.monotonic()
    if now - _last_log < _PERIOD_S:
        return
    _last_log = now
    try:
        import gc

        collected = gc.collect() if _GC else -1  # unreachable cycles holding device arrays
        dev = jax.local_devices()[0]
        stats = dev.memory_stats() or {}
        by_key: collections.Counter = collections.Counter()
        count: collections.Counter = collections.Counter()
        total = 0
        for arr in jax.live_arrays():
            if dev not in arr.devices():
                continue
            nbytes = math.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize
            key = (tuple(arr.shape), str(arr.dtype))
            by_key[key] += nbytes
            count[key] += 1
            total += nbytes
        try:
            from jax.extend import backend as _backend

            n_exec = len(_backend.get_backend().live_executables())
        except Exception:  # noqa: BLE001
            n_exec = -1
        top = ", ".join(
            f"{list(k[0])}{k[1]} x{count[k]}={v / 2**20:.0f}M" for k, v in by_key.most_common(_TOP)
        )
        grown = sorted(
            ((count[k] - _prev_count.get(k, 0), k) for k in count),
            key=lambda t: -t[0] * (by_key[t[1]] / max(count[t[1]], 1)),
        )
        growth = ", ".join(
            f"{list(k[0])}{k[1]} {d:+d} (x{count[k]})" for d, k in grown[:5] if d > 0
        )
        _prev_count = count
        mib = lambda key: stats.get(key, 0) / 2**20  # noqa: E731
        if _sampler is not None:
            min_free = _win["min_free"]
            logger.info(
                "[mempeak %s] window max in_use=%.0fM min largest_free=%.0fM; max since start=%.0fM",
                tag,
                _win["max"] / 2**20,
                (min_free if min_free is not None else -(2**20)) / 2**20,
                _win["all"] / 2**20,
            )
            _win["max"], _win["min_free"] = stats.get("bytes_in_use", 0), None
        logger.info(
            "[memlog %s] in_use=%.0fM peak=%.0fM limit=%.0fM largest_free=%.0fM allocs=%s "
            "live_arrays=%d (%.0fM on dev0) executables=%d gc=%d top: %s | grew: %s",
            tag,
            mib("bytes_in_use"),
            mib("peak_bytes_in_use"),
            mib("bytes_limit"),
            mib("largest_free_block_bytes"),
            stats.get("num_allocs"),
            sum(count.values()),
            total / 2**20,
            n_exec,
            collected,
            top,
            growth or "-",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("memlog failed: %s", e)
