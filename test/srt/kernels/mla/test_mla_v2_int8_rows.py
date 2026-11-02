"""int8 latent cache rows (SGLANG_JAX_DSA_KV_INT8) for the MLA v2 kernel's DCP mode.

Checks the row format (quantize / dequantize), and that attention over an int8
cache equals attention over a cache holding the dequantized rows, both through the
CPU reference path and through the kernel body (``pallas_call`` emulated on CPU
by the harness of ``test_mla_v2_bitexact``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import sgl_jax.srt.kernels.mla.v2.kernel as K
from sgl_jax.test.test_mla_v2_bitexact import (
    _cpu_bitcast,
    _cpu_execution_context,
    _cpu_reciprocal,
    _CpuRef,
    _FakeAsyncCopy,
    _py_fori_loop,
    _py_when,
)

LKV, ROPE, ROW = 512, 64, 640


def _rows(rng, t):
    """Latent / rope rows with per-row magnitudes spanning several octaves."""
    kvc = rng.standard_normal((t, LKV)) * rng.choice([1e-3, 0.3, 40.0], size=(t, 1))
    kpe = rng.standard_normal((t, ROPE)) * rng.choice([2e-2, 1.0, 300.0], size=(t, 1))
    return jnp.asarray(kvc, jnp.bfloat16), jnp.asarray(kpe, jnp.bfloat16)


def test_int8_rows_round_trip():
    rng = np.random.default_rng(0)
    kvc, kpe = _rows(rng, 257)
    kvc = kvc.at[3].set(0)
    kpe = kpe.at[4].set(0)
    rows = K.quantize_latent_rows_int8(kvc, kpe, ROW)
    assert rows.shape == (257, ROW) and rows.dtype == jnp.int8
    deq = np.asarray(K.dequantize_latent_rows_int8(rows, LKV))
    x = np.concatenate([np.asarray(kvc, np.float32), np.asarray(kpe, np.float32)], axis=1)
    for part in (slice(0, LKV), slice(LKV, LKV + ROPE)):
        err = np.abs(deq[:, part] - x[:, part]).max(axis=1)
        amax = np.abs(x[:, part]).max(axis=1)
        # power-of-two step <= 2 * amax / 127, rounding error <= step / 2
        assert (err <= amax / 127 * (1 + 1e-6)).all()
    assert (deq[:, LKV + ROPE :] == 0).all(), "scale lanes must dequantize to zero"
    np.testing.assert_array_equal(np.asarray(jnp.asarray(deq, jnp.bfloat16), np.float32), deq)
    r = np.asarray(rows).astype(np.int32)
    assert np.abs(r[:, : LKV + ROPE]).max() <= 127
    lane = LKV + K.INT8_SCALE_LANE
    assert (r[3, :LKV] == 0).all() and (r[4, LKV : LKV + ROPE] == 0).all()
    assert (r[:, lane + 2 :] == 0).all() and (r[:, LKV + ROPE : lane] == 0).all()
    # biased exponents in [1, 254]
    b = r[:, lane : lane + 2] & 0xFF
    assert (b >= 1).all() and (b <= 254).all()


def _dcp_case(rng, *, page_size=32, num_tokens=3, kp=4, q_dtype=jnp.float32, heads=8):
    """Attend-only DCP decode inputs over a cache whose pages hold quantized rows;
    returns (int8 args, dequantized-cache args, kwargs)."""
    num_pages = 1 + num_tokens * kp
    kvc, kpe = _rows(rng, num_pages * page_size)
    rows = K.quantize_latent_rows_int8(kvc, kpe, ROW)
    deq = K.dequantize_latent_rows_int8(rows, LKV, jnp.bfloat16)
    cache8 = rows.reshape(num_pages, page_size // 4, 4, ROW)
    cache_bf = deq.reshape(num_pages, page_size // 2, 2, ROW)
    pages = np.zeros((num_tokens, kp), np.int32)
    pos = np.full((num_tokens, kp), 0x3FFFFFFF, np.int32)
    n_owned = np.array([kp, 2, 1])[:num_tokens]
    global_len = np.array([kp * page_size * 3, 5 * page_size + 7, page_size // 2])[:num_tokens]
    perm = rng.permutation(np.arange(1, num_pages))
    for t in range(num_tokens):
        cand = np.sort(rng.choice(-(-global_len[t] // page_size), n_owned[t], replace=False))
        cand[-1] = (global_len[t] - 1) // page_size  # the new-token page
        cand = np.unique(cand)
        n = len(cand)
        n_owned[t] = n
        pages[t, :n] = perm[t * kp : t * kp + n]
        pos[t, :n] = cand * page_size
    ql = jnp.asarray(rng.standard_normal((num_tokens, heads, LKV)) * 0.2, q_dtype)
    qpe = jnp.asarray(rng.standard_normal((num_tokens, heads, ROPE)) * 0.2, q_dtype)
    meta = (
        jnp.asarray(np.maximum(n_owned, 1) * page_size, jnp.int32),
        jnp.asarray(pages.reshape(-1)),
        jnp.arange(num_tokens + 1, dtype=jnp.int32),
        jnp.arange(num_tokens + 1, dtype=jnp.int32) * (kp * page_size),
        jnp.array([num_tokens] * 3, jnp.int32),
    )
    kw = dict(
        page_seq_positions=jnp.asarray(pos.reshape(-1)),
        global_kv_lens=jnp.asarray(global_len, jnp.int32),
        return_lse=True,
        sm_scale=0.07,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
    )
    z8 = (jnp.zeros((num_tokens, LKV), jnp.int8), jnp.zeros((num_tokens, ROPE), jnp.int8))
    zb = (jnp.zeros((num_tokens, LKV), jnp.bfloat16), jnp.zeros((num_tokens, ROPE), jnp.bfloat16))
    return (ql, qpe, *z8, cache8, *meta), (ql, qpe, *zb, cache_bf, *meta), kw


def _run_kernel_emulated(*args, **kwargs):
    """mla_ragged_paged_attention with its pallas_calls emulated on CPU (kernel body
    executed eagerly; the CPU reference dispatch bypassed)."""

    def fake_pallas_call(kernel_fn, *, grid_spec, out_shape, input_output_aliases=None, **kw):
        del kw

        def runner(*call_args):
            num_sp = grid_spec.num_scalar_prefetch
            sp_args = [_CpuRef(np.array(a, copy=True)) for a in call_args[:num_sp]]
            in_bufs = [np.array(a, copy=True) for a in call_args[num_sp:]]
            out_bufs = [np.zeros(s.shape, dtype=s.dtype) for s in out_shape]
            for in_idx, out_idx in (input_output_aliases or {}).items():
                out_bufs[out_idx] = in_bufs[in_idx - num_sp].copy()
            scratch = [
                _CpuRef(
                    np.zeros(
                        ss.shape,
                        dtype=(
                            ss.dtype
                            if str(getattr(ss, "dtype", "dma_sem")) != "dma_sem"
                            else np.int32
                        ),
                    )
                )
                for ss in grid_spec.scratch_shapes
            ]
            refs = [_CpuRef(b) for b in in_bufs] + [_CpuRef(b) for b in out_bufs]
            for pid in range(int(grid_spec.grid[0])):
                with mock.patch.object(pl, "program_id", side_effect=lambda axis, p=pid: p):
                    kernel_fn(*sp_args, *refs, *scratch)
            return [jnp.asarray(b) for b in out_bufs]

        return runner

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with (
            _cpu_execution_context(),
            mock.patch.object(jax, "default_backend", return_value="tpu"),
            mock.patch.object(pl, "pallas_call", side_effect=fake_pallas_call),
            mock.patch.object(pl, "when", side_effect=_py_when),
            mock.patch.object(pl, "reciprocal", side_effect=_cpu_reciprocal),
            mock.patch.object(pltpu, "make_async_copy", side_effect=_FakeAsyncCopy),
            mock.patch.object(pltpu, "bitcast", side_effect=_cpu_bitcast),
            mock.patch.object(lax, "fori_loop", side_effect=_py_fori_loop),
        ):
            return K.mla_ragged_paged_attention.__wrapped__(*args, **kwargs)


def test_int8_cpu_reference_matches_dequantized_cache():
    a8, abf, kw = _dcp_case(np.random.default_rng(1))
    cache_in = np.asarray(a8[4])  # the kernel donates its cache argument
    o8, cache8, lse8 = K.mla_ragged_paged_attention(*a8, **kw)
    ob, _, lseb = K.mla_ragged_paged_attention(*abf, **kw)
    np.testing.assert_array_equal(np.asarray(o8), np.asarray(ob))
    np.testing.assert_array_equal(np.asarray(lse8), np.asarray(lseb))
    np.testing.assert_array_equal(np.asarray(cache8), cache_in)


@pytest.mark.parametrize("q_dtype", [jnp.float32, jnp.bfloat16])
def test_int8_kernel_body_matches_dequantized_cache(q_dtype):
    a8, abf, kw = _dcp_case(np.random.default_rng(2), q_dtype=q_dtype)
    cache_in = np.asarray(a8[4])
    o8, cache8, lse8 = _run_kernel_emulated(*a8, **kw)
    ob, _, lseb = _run_kernel_emulated(*abf, **kw)
    o_ref, _, lse_ref = K.mla_ragged_paged_attention(*abf, **kw)  # CPU reference
    o8, ob, o_ref = (np.asarray(x, np.float32) for x in (o8, ob, o_ref))
    # f32 queries: only the score's summation order differs (nope + rope dots vs one);
    # bf16 queries add one output rounding.
    tol = 2e-5 if q_dtype == jnp.float32 else 1e-2
    scale = np.abs(ob).max()
    assert np.abs(o8 - ob).max() <= tol * scale, np.abs(o8 - ob).max()
    assert np.abs(o8 - o_ref).max() <= max(tol, 1e-2 if q_dtype == jnp.bfloat16 else 1e-4) * scale
    np.testing.assert_allclose(np.asarray(lse8), np.asarray(lseb), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(lse8), np.asarray(lse_ref), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(cache8), cache_in)


def test_int8_needs_attend_only_mode():
    a8, _, kw = _dcp_case(np.random.default_rng(3))
    kw = {k: v for k, v in kw.items() if k not in ("page_seq_positions", "global_kv_lens")}
    with pytest.raises(NotImplementedError, match="attend-only"):
        K.mla_ragged_paged_attention(*a8, **kw)
