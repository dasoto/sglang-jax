"""Bit-exactness of the elementwise MXFP4 dequantization (CPU).

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_mxfp4_dequant.py -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.quantization.mxfp4 import (
    E2M1_MAGNITUDES,
    dequant_mxfp4,
    dequant_mxfp4_numpy,
    e2m1_to_bf16,
    quantize_mxfp4_numpy,
)


def _bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def test_e2m1_decode_all_codes():
    codes = jnp.arange(16, dtype=jnp.uint8)
    got = np.asarray(e2m1_to_bf16(codes).astype(jnp.float32))
    want = np.concatenate([E2M1_MAGNITUDES, -E2M1_MAGNITUDES]).astype(np.float32)
    np.testing.assert_array_equal(_bits(got), _bits(want))  # includes -0.0 for code 8


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("out_dtype", [jnp.bfloat16, jnp.float32])
def test_dequant_bit_exact_all_bytes_and_scales(axis, out_dtype):
    rng = np.random.default_rng(0)
    # Every byte value appears, with scales spanning the useful E8M0 range.
    out_dim, in_dim = 64, 512
    packed = np.tile(np.arange(256, dtype=np.uint8), out_dim * in_dim // 2 // 256)
    packed = rng.permutation(packed).reshape(out_dim, in_dim // 2)
    scale = rng.integers(100, 140, size=(out_dim, in_dim // 32), dtype=np.uint8)
    scale[0, :4] = (2, 127, 200, 252)  # extremes of the exact (normal, finite) range
    want = dequant_mxfp4_numpy(packed, scale)
    if axis == -2:  # k-major orientation used by the MoE layer
        packed, scale, want = packed.T, scale.T, want.T
    got = jax.jit(lambda p, s: dequant_mxfp4(p, s, axis=axis, out_dtype=out_dtype))(
        jnp.asarray(packed), jnp.asarray(scale)
    )
    assert got.dtype == out_dtype and got.shape == want.shape
    np.testing.assert_array_equal(_bits(np.asarray(got.astype(jnp.float32))), _bits(want))


def test_dequant_batched_experts():
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(3, 64, 48), dtype=np.uint8)  # [E, k//2, n]
    scale = rng.integers(110, 135, size=(3, 4, 48), dtype=np.uint8)  # [E, k//32, n]
    got = dequant_mxfp4(jnp.asarray(packed), jnp.asarray(scale), axis=-2, out_dtype=jnp.float32)
    want = dequant_mxfp4_numpy(packed, scale, axis=-2)
    np.testing.assert_array_equal(_bits(np.asarray(got)), _bits(want))


def test_quantize_round_trip_is_nearest_grid_point():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((16, 128)).astype(np.float32) * 0.05
    packed, scale = quantize_mxfp4_numpy(weight)
    deq = dequant_mxfp4_numpy(packed, scale)
    # Re-quantizing the dequantized weights is lossless.
    packed2, scale2 = quantize_mxfp4_numpy(deq)
    np.testing.assert_array_equal(dequant_mxfp4_numpy(packed2, scale2), deq)
    # Error <= half the grid spacing (1 * scale) inside the grid, <= 2 * scale
    # when amax/scale in (6, 8) saturates to 6.
    groups = np.ldexp(1.0, scale.astype(np.int32) - 127).repeat(32, axis=-1)
    assert np.all(np.abs(deq - weight) <= 2 * groups + 1e-12)
