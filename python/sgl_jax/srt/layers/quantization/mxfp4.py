"""MXFP4 (OCP microscaling FP4) helpers for compressed-tensors checkpoints.

Format (compressed-tensors ``mxfp4-pack-quantized``, as shipped by Kimi-K3):

* ``weight_packed``: uint8, two E2M1 codes per byte along the input
  (contraction) axis; the LOW nibble holds the EVEN element.
* ``weight_scale``: uint8 E8M0, one shared power-of-two scale ``2**(s - 127)``
  per 32 consecutive input elements.
* E2M1 code ``n``: sign ``s = n >> 3``, exponent ``e = (n >> 1) & 3`` (bias 1),
  mantissa ``m = n & 1``; value ``(-1)**s * (m / 2 if e == 0 else
  2**(e - 1) * (1 + m / 2))``, i.e. the magnitudes ``[0, .5, 1, 1.5, 2, 3, 4, 6]``.

``dequant_mxfp4`` is pure elementwise integer/bit arithmetic plus reshapes (no
table lookup or gather, which lowers very poorly on TPU) so XLA can fuse it
into the consumer. Every dequantized value is a <= 2-significant-bit number
times a power of two, hence exactly representable in bf16: the result is
bit-exact whenever it is a finite normal number, i.e. for E8M0 scales in
``[2, 252]`` (real checkpoints sit around ``2**-10``; ``0``/``255`` are
``2**-127``/NaN in E8M0 and are not supported).

The NumPy helpers (``dequant_mxfp4_numpy``, ``quantize_mxfp4_numpy``) are
independent references for tests and offline tooling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MXFP4_GROUP_SIZE = 32
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_to_bf16(codes: jax.Array) -> jax.Array:
    """Decode 4-bit E2M1 codes (any unsigned int dtype, values 0..15) to bf16.

    bf16 bits: sign << 15 | exponent(8, bias 127) << 7 | mantissa(7). A normal
    E2M1 value ``2**(e-1) * (1 + m/2)`` has bf16 exponent ``e + 126`` and top
    mantissa bit ``m``; the only subnormal, 0.5, is exponent 126.
    """
    n = codes.astype(jnp.uint16)
    sign = (n >> 3) & 1
    exponent = (n >> 1) & 3
    mantissa = n & 1
    normal = ((exponent + 126) << 7) | (mantissa << 6)
    subnormal = jnp.where(mantissa == 1, jnp.uint16(126 << 7), jnp.uint16(0))
    magnitude = jnp.where(exponent == 0, subnormal, normal)
    return jax.lax.bitcast_convert_type((sign << 15) | magnitude, jnp.bfloat16)


def e8m0_to_bf16(scale: jax.Array) -> jax.Array:
    """``2**(s - 127)`` as bf16 by writing ``s`` into the exponent field."""
    return jax.lax.bitcast_convert_type(scale.astype(jnp.uint16) << 7, jnp.bfloat16)


def dequant_mxfp4(
    packed: jax.Array,
    scale: jax.Array,
    *,
    axis: int = -1,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Dequantize packed MXFP4 along ``axis`` (the contraction axis).

    Args:
      packed: uint8 with ``K // 2`` entries along ``axis`` (low nibble = even).
      scale: uint8 E8M0 with ``K // 32`` entries along ``axis``; every other
        dimension matches ``packed``.
      axis: packed axis. ``-1`` is the checkpoint orientation ``[..., out, in//2]``;
        ``-2`` is the k-major orientation ``[..., in//2, out]`` used by the
        MoE layer (the result is then directly a GMM ``[E, k, n]`` operand and
        the interleave / scale broadcast happen along sublanes on TPU).
      out_dtype: result dtype (bf16 values are exact in any wider float dtype).

    Returns:
      Array like ``packed`` with ``K`` entries along ``axis``.
    """
    if packed.dtype != jnp.uint8 or scale.dtype != jnp.uint8:
        raise ValueError(f"MXFP4 expects uint8 packed/scale, got {packed.dtype}/{scale.dtype}")
    ndim = packed.ndim
    axis = axis % ndim
    k = packed.shape[axis] * 2
    expected_scale = (*packed.shape[:axis], k // MXFP4_GROUP_SIZE, *packed.shape[axis + 1 :])
    if k % MXFP4_GROUP_SIZE or scale.shape != expected_scale:
        raise ValueError(
            f"MXFP4 scale shape {scale.shape} does not match packed shape {packed.shape} "
            f"along axis {axis} (expected {expected_scale})"
        )
    lo = e2m1_to_bf16(packed & 0x0F)
    hi = e2m1_to_bf16(packed >> 4)
    shape_k = (*packed.shape[:axis], k, *packed.shape[axis + 1 :])
    values = jnp.stack([lo, hi], axis=axis + 1).reshape(shape_k)
    grouped = (
        *packed.shape[:axis],
        k // MXFP4_GROUP_SIZE,
        MXFP4_GROUP_SIZE,
        *packed.shape[axis + 1 :],
    )
    weight = values.reshape(grouped) * jnp.expand_dims(e8m0_to_bf16(scale), axis + 1)
    return weight.reshape(shape_k).astype(out_dtype)


def dequant_mxfp4_numpy(packed: np.ndarray, scale: np.ndarray, axis: int = -1) -> np.ndarray:
    """Independent float32 NumPy reference (table lookup + ``ldexp``)."""
    packed = np.moveaxis(np.asarray(packed, dtype=np.uint8), axis, -1)
    scale = np.moveaxis(np.asarray(scale, dtype=np.uint8), axis, -1)
    codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    values = E2M1_MAGNITUDES[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    exponents = np.repeat(scale.astype(np.int32) - 127, MXFP4_GROUP_SIZE, axis=-1)
    weight = np.ldexp(values, exponents).astype(np.float32)
    return np.moveaxis(weight, -1, axis)


def quantize_mxfp4_numpy(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``[..., out, in]`` floats to (packed ``[..., out, in//2]``, scale).

    OCP MX rule: per 32-group ``scale = 2**(floor(log2(amax)) - 2)`` (E2M1
    emax = 2), values rounded to the nearest E2M1 magnitude and clamped to 6.
    Test / tooling helper; the round trip is checked against
    ``dequant_mxfp4_numpy``.
    """
    weight = np.asarray(weight, dtype=np.float32)
    *lead, out_dim, in_dim = weight.shape
    if in_dim % MXFP4_GROUP_SIZE:
        raise ValueError(f"in_dim={in_dim} must be a multiple of {MXFP4_GROUP_SIZE}")
    groups = weight.reshape(*lead, out_dim, in_dim // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE)
    amax = np.abs(groups).max(axis=-1)
    with np.errstate(divide="ignore"):
        exp = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 2, 0)
    exp = np.clip(exp, -126, 127).astype(np.int32)
    scaled = groups / np.ldexp(np.float32(1.0), exp)[..., None]
    magnitude = np.minimum(np.abs(scaled), 6.0)
    idx = np.abs(magnitude[..., None] - E2M1_MAGNITUDES).argmin(axis=-1).astype(np.uint8)
    codes = (idx | np.where(scaled < 0, 8, 0).astype(np.uint8)).reshape(*lead, out_dim, in_dim)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    scale = (exp + 127).astype(np.uint8)
    return packed, scale
