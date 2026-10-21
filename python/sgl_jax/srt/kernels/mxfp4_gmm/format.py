"""Load-time weight formats for the fused MXFP4 grouped matmul kernel.

The checkpoint stores routed-expert weights k-major as

* ``packed``: uint8 ``[E, K/2, N]``, byte ``j`` holds input rows ``2j`` (low
  nibble) and ``2j+1`` (high nibble), E2M1 codes ``sign<<3 | magnitude``;
* ``scale``: uint8 ``[E, K/32, N]`` E8M0 (``2**(s-127)`` per 32 rows).

On TPU v5e the VPU has 32-bit lanes and no bf16 ALU, so the kernel decodes two
weights per 32-bit lane (SWAR, one bf16 per 16-bit half) and feeds the MXU with
bf16.  To make that cheap the 4-bit codes are re-encoded and re-arranged once at
load time (same 4 bits per weight, same 61 MB per layer):

``mu`` code (3 bits) -- a linear exponent/mantissa index
    bf16 bits of ``|w|`` are ``G + (mu << 6)`` with ``G = (s - 1) << 7``:
    ``mu = 0`` is 0.5, ``2..7`` are 1, 1.5, 2, 3, 4, 6 (``mu == t`` for
    ``t >= 2``).  ``mu = 1`` (0.75, not an E2M1 value) encodes ZERO; the kernel
    gates ``G`` off for it, leaving ``0x0040`` (a bf16 subnormal, 2**-127), so
    zeros contribute nothing (the MXU result is identical to an exact 0).

``l24`` word layout -- ``uint32 [E, K/8, N]``
    Every 64-row block of K is stored as 8 consecutive words per column (one
    (8, 128) vreg row).  Word ``i`` holds the 8 weights of rows
    ``16*j + 2*i + h`` (``j = 0..3`` output vreg, ``h = 0/1`` half): in each
    16-bit half the magnitude fields sit at bits ``[0:3], [3:6], [6:9], [9:12]``
    and the sign bits at ``12..15``, assigned so that output words ``j = 0, 1``
    need just ``shift, and, add`` (sign included) and ``j = 2, 3`` a separate
    sign extraction.  Bit-casting each decoded ``(8, 128)`` uint32 word block to
    bf16 ``(16, 128)`` puts row ``2i + h`` of the block at half ``h`` of
    sublane ``i``, so the 4 decoded blocks concatenate to the 64 rows in
    natural order (no activation permutation needed).

Scales stay uint8 ``[E, K/32, N]`` (the kernel derives ``G`` itself).
"""

from __future__ import annotations

import numpy as np

MXFP4_GROUP = 32
BLOCK_K = 64  # rows per l24 word block (8 words x 8 nibbles)

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# magnitude t (= code & 7) -> mu
_T_TO_MU = np.array([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
# mu -> magnitude t
_MU_TO_T = np.array([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.uint32)

# l24 field placement inside one 16-bit half, per output word j:
#   (magnitude field start bit, sign bit)
L24_FIELDS = ((6, 15), (3, 12), (0, 13), (9, 14))


def unpack_codes_kmajor(packed: np.ndarray) -> np.ndarray:
    """uint8 ``[..., K/2, N]`` -> 4-bit codes ``[..., K, N]`` (natural row order)."""
    packed = np.asarray(packed, dtype=np.uint8)
    *lead, k2, n = packed.shape
    codes = np.empty((*lead, 2 * k2, n), dtype=np.uint8)
    codes[..., 0::2, :] = packed & 0x0F
    codes[..., 1::2, :] = packed >> 4
    return codes


def pack_codes_kmajor(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_codes_kmajor`."""
    codes = np.asarray(codes, dtype=np.uint8)
    return (codes[..., 0::2, :] | (codes[..., 1::2, :] << 4)).astype(np.uint8)


def encode_l24(packed: np.ndarray) -> np.ndarray:
    """Checkpoint k-major packed uint8 ``[..., K/2, N]`` -> l24 words ``uint32 [..., K/8, N]``."""
    codes = unpack_codes_kmajor(packed)
    *lead, k, n = codes.shape
    if k % BLOCK_K:
        raise ValueError(f"K={k} must be a multiple of {BLOCK_K}")
    sign = (codes >> 3).astype(np.uint32)
    mu = _T_TO_MU[codes & 7]
    # k = 64*b + 16*j + 2*i + h
    sign = sign.reshape(*lead, k // BLOCK_K, 4, 8, 2, n)
    mu = mu.reshape(*lead, k // BLOCK_K, 4, 8, 2, n)
    words = np.zeros((*lead, k // BLOCK_K, 8, n), dtype=np.uint32)
    for j, (fpos, spos) in enumerate(L24_FIELDS):
        for h in range(2):
            words |= mu[..., j, :, h, :] << np.uint32(fpos + 16 * h)
            words |= sign[..., j, :, h, :] << np.uint32(spos + 16 * h)
    return words.reshape(*lead, k // 8, n)


def decode_l24_codes(words: np.ndarray) -> np.ndarray:
    """l24 words ``uint32 [..., K/8, N]`` -> original E2M1 codes ``uint8 [..., K, N]``.

    (Zero is returned as +0, code 0.)
    """
    words = np.asarray(words, dtype=np.uint32)
    *lead, k8, n = words.shape
    w = words.reshape(*lead, k8 // 8, 8, n)
    codes = np.zeros((*lead, k8 // 8, 4, 8, 2, n), dtype=np.uint8)
    for j, (fpos, spos) in enumerate(L24_FIELDS):
        for h in range(2):
            mu = (w >> np.uint32(fpos + 16 * h)) & 7
            s = (w >> np.uint32(spos + 16 * h)) & 1
            t = _MU_TO_T[mu]
            codes[..., j, :, h, :] = (t | np.where(t == 0, 0, s << 3)).astype(np.uint8)
    return codes.reshape(*lead, k8 * 8, n)


def encode_l24_jax(packed):
    """jnp version of :func:`encode_l24` (jit-able; use it to convert loaded weights on device).

    Works blockwise along K (64 rows), so a K-sharded operand (e.g. the down projection under
    expert-TP) converts shard-locally as long as every shard holds a multiple of 64 rows.
    """
    import jax.numpy as jnp

    if packed.dtype != jnp.uint8:
        raise ValueError(f"expected uint8 packed weights, got {packed.dtype}")
    *lead, k2, n = packed.shape
    k = 2 * k2
    if k % BLOCK_K:
        raise ValueError(f"K={k} must be a multiple of {BLOCK_K}")
    codes = jnp.stack([packed & 0x0F, packed >> 4], axis=-2).reshape(*lead, k, n).astype(jnp.uint32)
    t = codes & 7
    sign = (codes >> 3).reshape(*lead, k // BLOCK_K, 4, 8, 2, n)
    mu = jnp.where(t == 0, 1, jnp.where(t == 1, 0, t)).astype(jnp.uint32)
    mu = mu.reshape(*lead, k // BLOCK_K, 4, 8, 2, n)
    words = jnp.zeros((*lead, k // BLOCK_K, 8, n), jnp.uint32)
    for j, (fpos, spos) in enumerate(L24_FIELDS):
        for h in range(2):
            words = words | (mu[..., j, :, h, :] << (fpos + 16 * h)) | (sign[..., j, :, h, :] << (spos + 16 * h))
    return words.reshape(*lead, k // 8, n)


def canonical_codes(packed: np.ndarray) -> np.ndarray:
    """Checkpoint codes with -0 mapped to +0 (what the l24 round trip returns)."""
    codes = unpack_codes_kmajor(packed)
    return np.where((codes & 7) == 0, 0, codes).astype(np.uint8)


def dequant_reference(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float32 ``[..., K, N]`` reference from checkpoint k-major packed/scale."""
    codes = unpack_codes_kmajor(packed)
    vals = E2M1_MAGNITUDES[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    exps = np.repeat(np.asarray(scale, dtype=np.int32) - 127, MXFP4_GROUP, axis=-2)
    return np.ldexp(vals, exps).astype(np.float32)


def bf16_bits_l24_reference(words: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """What the kernel's SWAR decode produces, as uint16 bf16 bit patterns ``[..., K, N]``.

    Nonzero weights are exact; zeros come out as ``0x0040`` / ``0x8040``
    (bf16 subnormal 2**-127), which the kernel relies on being negligible.
    """
    words = np.asarray(words, dtype=np.uint32)
    *lead, k8, n = words.shape
    g = (np.asarray(scale, dtype=np.uint32) - 1) << 7  # [..., K/32, N]
    w = words.reshape(*lead, k8 // 8, 8, n)
    out = np.zeros((*lead, k8 // 8, 4, 8, 2, n), dtype=np.uint32)
    for j, (fpos, spos) in enumerate(L24_FIELDS):
        grp = g.reshape(*lead, k8 // 8, 2, n)[..., j // 2, :][..., None, :]
        for h in range(2):
            mu = (w >> np.uint32(fpos + 16 * h)) & 7
            s = (w >> np.uint32(spos + 16 * h)) & 1
            gate = np.where(mu == 1, 0, grp)
            out[..., j, :, h, :] = (gate + (mu << 6)) | (s << 15)
    return out.reshape(*lead, k8 * 8, n).astype(np.uint16)


# ----------------------------------------------------------------------------------------
# f8l: decode to float8_e5m2 (4 weights per 32-bit lane) with a per-column power-of-two
# post-scale.  Every 64-row block is 8 words per column; word i, byte b holds rows
# 4*i + b (j = 0) and 32 + 4*i + b (j = 1): magnitude field bits [1:4] / sign 7 for j = 0,
# magnitude [4:7] / sign 0 for j = 1.  e5m2 bits of |w| are (Eg << 2) + (mu << 1) with the
# per-group exponent Eg = F8_ETOP - (smax - s) (smax = max scale of the column), so all
# groups of a column share one f32 post-scale 2**(smax - 113 - F8_ETOP).
# ----------------------------------------------------------------------------------------
F8_ETOP = 27  # e5m2 exponent of the 0.5-magnitude of the largest group (max code -> 30)
F8L_FIELDS = ((1, 7), (4, 0))


def encode_f8l(packed: np.ndarray, scale: np.ndarray):
    """-> (words uint32 [..., K/8, N], eg uint8 [..., K/32, N], colscale float32 [..., 1, N])."""
    codes = unpack_codes_kmajor(packed)
    *lead, k, n = codes.shape
    if k % BLOCK_K:
        raise ValueError(f"K={k} must be a multiple of {BLOCK_K}")
    scale = np.asarray(scale, dtype=np.int32)
    smax = scale.max(axis=-2, keepdims=True)
    eg = F8_ETOP - (smax - scale)
    if eg.min() < 1:
        raise ValueError(f"scale span per column {int((smax - scale).max())} binades exceeds e5m2 range")
    colscale = np.ldexp(np.float32(1.0), smax - 113 - F8_ETOP).astype(np.float32)
    sign = (codes >> 3).astype(np.uint32)
    mu = _T_TO_MU[codes & 7]
    # k = 64*blk + 32*j + 4*i + b
    sign = sign.reshape(*lead, k // BLOCK_K, 2, 8, 4, n)
    mu = mu.reshape(*lead, k // BLOCK_K, 2, 8, 4, n)
    words = np.zeros((*lead, k // BLOCK_K, 8, n), dtype=np.uint32)
    for j, (fpos, spos) in enumerate(F8L_FIELDS):
        for b in range(4):
            words |= mu[..., j, :, b, :] << np.uint32(fpos + 8 * b)
            words |= sign[..., j, :, b, :] << np.uint32(spos + 8 * b)
    return words.reshape(*lead, k // 8, n), eg.astype(np.uint8), colscale


def random_mxfp4(shape_kn, n_experts, *, seed=0, scale_lo=112, scale_hi=124, zero_frac=None):
    """Random checkpoint-format MXFP4 expert weights: (packed u8 [E,K/2,N], scale u8 [E,K/32,N])."""
    k, n = shape_kn
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n_experts, k, n), dtype=np.uint8)
    if zero_frac is not None:
        z = rng.random(size=codes.shape) < zero_frac
        codes = np.where(z, codes & 8, codes).astype(np.uint8)
    packed = pack_codes_kmajor(codes)
    scale = rng.integers(scale_lo, scale_hi, size=(n_experts, k // MXFP4_GROUP, n), dtype=np.uint8)
    return packed, scale
