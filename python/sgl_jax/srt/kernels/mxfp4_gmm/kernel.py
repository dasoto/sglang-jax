"""Fused MXFP4 dequant + matmul Pallas kernels for TPU v5e (see format.py).

Decode cost per 8 weights (one uint32 word -> 4 bf16-pair words), in 32-bit VPU ops:
  l24, zero_mode="none" ......... 17   (numerically wrong for zero codes; cost floor)
  l24, zero_mode="mul" .......... 33   (gates the exponent offset G with an integer multiply)
  l24, zero_mode="mask" ......... 37   (gates G with a spread mask: shifts/sub/and only)
The f8l (float8_e5m2) variant decodes 4 weights per 32-bit op (18 / 8 / 20 ops per 8 weights).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

U32 = jnp.uint32

_M_JA = 0x81C081C0  # magnitude bits 6..8 + sign bit 15, both halves
_M_MU = 0x01C001C0  # magnitude bits only
_M_SG = 0x80008000  # sign bits
_M_NZ = 0x00800080  # nz flag position (bit 7 of each half)
_FLIP_MU0 = 0x02490249  # lowest magnitude bit of the 4 fields, both halves


def _c(v):
    return jnp.asarray(v, dtype=U32)


ZERO_MODES = ("mul", "mask", "none")


def _gate(nzf, g, zero_mode, spread_shift):
    """G for nonzero weights, 0 for zero codes. nzf: flags (1 bit per lane field)."""
    if zero_mode == "mul":
        return nzf * g  # g = s - 1 (bf16) / Eg (f8): the flag sits at the field's LSB position
    return g & ((nzf << spread_shift) - nzf)  # g = G replicated; mask covers the exponent bits


def decode_l24_block(w, ga, gb, *, zero_mode: str = "mul"):
    """Decode one l24 word block.

    Args:
      w: uint32 [8, bn] -- 64 weight rows x bn columns.
      ga, gb: uint32 [8, bn] (rows broadcast) for the two 32-row groups of the block, from
        :func:`scale_operand` (``s - 1`` for zero_mode="mul", else ``(s - 1) << 7`` in both halves).
    Returns:
      bf16 [64, bn] in natural row order.
    """
    if zero_mode != "none":
        t = w ^ _c(_FLIP_MU0)
        nz = t | (t >> 1) | (t << 1)  # bit 3f+1 of each half = (mu_f != 1)
        m = _c(_M_NZ)
        a0 = _gate(nz & m, ga, zero_mode, 8)
        a1 = _gate((nz << 3) & m, ga, zero_mode, 8)
        a2 = _gate((nz << 6) & m, gb, zero_mode, 8)
        a3 = _gate((nz >> 3) & m, gb, zero_mode, 8)
    else:
        a0 = a1 = ga
        a2 = a3 = gb
    o0 = (w & _c(_M_JA)) + a0
    o1 = ((w << 3) & _c(_M_JA)) + a1
    o2 = (((w << 6) & _c(_M_MU)) + a2) | ((w << 2) & _c(_M_SG))
    o3 = (((w >> 3) & _c(_M_MU)) + a3) | ((w << 1) & _c(_M_SG))
    return jnp.concatenate([pltpu.bitcast(o, jnp.bfloat16) for o in (o0, o1, o2, o3)], axis=0)


def decode_f8l_block(w, ea, eb, *, zero_mode: str = "mul"):
    """f8l word block uint32 [8, bn] -> float8_e5m2 [64, bn] (natural row order).

    ea, eb: uint32 [8, bn] from :func:`f8_scale_operand` (``Eg`` for "mul", else ``Eg << 2`` in
    all four bytes).
    """
    if zero_mode != "none":
        t = w ^ _c(0x12121212)
        nz = t | (t >> 1) | (t << 1)  # bits 2 / 5 of each byte = (mu != 1)
        m = _c(0x04040404)
        a0 = _gate(nz & m, ea, zero_mode, 5)
        a1 = _gate((nz >> 3) & m, eb, zero_mode, 5)
    else:
        a0, a1 = ea, eb
    o0 = (w & _c(0x8E8E8E8E)) + a0
    o1 = (((w >> 3) & _c(0x0E0E0E0E)) + a1) | ((w << 7) & _c(0x80808080))
    return jnp.concatenate([pltpu.bitcast(o, jnp.float8_e5m2) for o in (o0, o1)], axis=0)


def decode_f8l_chunk(w_ref, g_ref, c, *, nblk: int, zero_mode: str, lanes=None):
    ln = _lane_slice(lanes)
    wq = w_ref[pl.ds(pl.multiple_of(c * 8 * nblk, 8 * nblk), 8 * nblk), ln]
    ngrp = 2 * nblk
    gq = g_ref[pl.ds(pl.multiple_of(c * ngrp, ngrp), ngrp), ln] if ngrp % 8 == 0 else g_ref[pl.ds(c * ngrp, ngrp), ln]
    parts = [
        decode_f8l_block(wq[8 * b : 8 * b + 8, :], _bcast_row(gq, 2 * b), _bcast_row(gq, 2 * b + 1),
                         zero_mode=zero_mode)
        for b in range(nblk)
    ]
    return parts[0] if nblk == 1 else jnp.concatenate(parts, axis=0)


def f8_scale_operand(eg_u32, *, zero_mode: str):
    if zero_mode == "mul":
        return eg_u32
    e = eg_u32 << 2
    e = e | (e << 8)
    return e | (e << 16)


def scale_operand(s_u32, *, zero_mode: str):
    """Per-group operand from E8M0 scales (uint32): ``s-1`` ("mul") or ``(s-1)<<7`` in both halves."""
    sm1 = s_u32 - _c(1)
    if zero_mode == "mul":
        return sm1
    g = sm1 << 7
    return g | (g << 16)


def _bcast_row(x, r, rows=8):
    return jnp.broadcast_to(x[r : r + 1, :], (rows, x.shape[1]))


def _lane_slice(lanes):
    return slice(None) if lanes is None else lanes


def decode_l24_chunk(w_ref, g_ref, c, *, nblk: int, zero_mode: str, lanes=None):
    """Decode ``nblk`` consecutive 64-row blocks starting at block ``c * nblk``.

    w_ref: uint32 [K/8, bn] ref; g_ref: uint32 [K/32, bn] ref holding scale_operand();
    lanes: optional column slice (pl.ds) to decode only part of the tile.
    Returns bf16 [64 * nblk, bn or lanes].
    """
    ln = _lane_slice(lanes)
    wq = w_ref[pl.ds(pl.multiple_of(c * 8 * nblk, 8 * nblk), 8 * nblk), ln]
    ngrp = 2 * nblk
    if ngrp % 8 == 0:
        gq = g_ref[pl.ds(pl.multiple_of(c * ngrp, ngrp), ngrp), ln]
    else:
        gq = g_ref[pl.ds(c * ngrp, ngrp), ln]
    parts = []
    for b in range(nblk):
        ga = _bcast_row(gq, 2 * b)
        gb = _bcast_row(gq, 2 * b + 1)
        parts.append(decode_l24_block(wq[8 * b : 8 * b + 8, :], ga, gb, zero_mode=zero_mode))
    return parts[0] if nblk == 1 else jnp.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------------------
# Dense per-expert kernel: out[e] = x[e] @ W[e] (x padded to T rows per expert).
# Used for benchmarking decode / tiling variants; the ragged GMM lives in gmm.py.
# ----------------------------------------------------------------------------------------


def _dense_kernel(x_ref, w_ref, s_ref, *rest, kc: int, zero_mode: str, unroll, fmt: str, nl: int, bigdot: bool):
    # x_ref [T, K] bf16; w_ref [K/8, bn] u32; s_ref [K/32, bn] u8|u32; o_ref [T, bn]
    wbuf = rest[-1] if bigdot else None
    rest = rest[:-1] if bigdot else rest
    if fmt == "f8l":
        cs_ref, o_ref, g_scr = rest
        g_scr[...] = f8_scale_operand(s_ref[...].astype(U32), zero_mode=zero_mode)
        dec = decode_f8l_chunk
    else:
        o_ref, g_scr = rest
        g_scr[...] = scale_operand(s_ref[...].astype(U32), zero_mode=zero_mode)
        dec = decode_l24_chunk
    if bigdot:  # decode the whole tile into VMEM, then one MXU dot over the full K
        def dbody(c, carry):
            wbuf[pl.ds(pl.multiple_of(c * kc, kc), kc), :] = dec(w_ref, g_scr, c, nblk=kc // 64, zero_mode=zero_mode)
            return carry

        lax.fori_loop(0, x_ref.shape[1] // kc, dbody, 0, unroll=unroll)
        acc = jnp.dot(x_ref[...], wbuf[...], preferred_element_type=jnp.float32)
        if fmt == "f8l":
            acc = acc * cs_ref[...]
        o_ref[...] = acc.astype(o_ref.dtype)
        return
    nblk = kc // 64
    nk = x_ref.shape[1] // kc
    bn = w_ref.shape[1]
    nls = bn // nl

    def body(c, acc):
        xk = x_ref[:, pl.ds(pl.multiple_of(c * kc, kc), kc)]
        if nls == 1:
            wd = dec(w_ref, g_scr, c, nblk=nblk, zero_mode=zero_mode)
            return acc + jnp.dot(xk, wd, preferred_element_type=jnp.float32)
        parts = []
        for l in range(nls):
            wd = dec(w_ref, g_scr, c, nblk=nblk, zero_mode=zero_mode, lanes=pl.ds(l * nl, nl))
            parts.append(jnp.dot(xk, wd, preferred_element_type=jnp.float32))
        return acc + jnp.concatenate(parts, axis=1)

    acc = lax.fori_loop(0, nk, body, jnp.zeros(o_ref.shape, jnp.float32), unroll=unroll)
    if fmt == "f8l":
        acc = acc * cs_ref[...]
    o_ref[...] = acc.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("bn", "kc", "nl", "zero_mode", "out_dtype", "resident", "unroll", "bigdot", "interpret"),
)
def dense_mxfp4_matmul(
    x,
    w,
    s,
    colscale=None,
    *,
    bn: int = 512,
    kc: int = 256,
    nl: int | None = None,
    zero_mode: str = "mul",
    out_dtype=jnp.bfloat16,
    resident: bool = False,
    unroll=1,
    bigdot: bool = False,
    interpret: bool = False,
):
    """x [E, T, K] bf16; w l24 uint32 [E, K/8, N]; s uint8|uint32 [E, K/32, N] -> [E, T, N].

    resident=True pins every grid step to weight block (0, 0) so no weight DMA happens
    after the first step (measures compute throughput only).
    """
    E, T, K = x.shape
    N = w.shape[-1]
    assert w.shape == (E, K // 8, N) and s.shape == (E, K // 32, N), (x.shape, w.shape, s.shape)
    assert N % bn == 0 and K % kc == 0 and kc % 64 == 0

    if resident:
        widx = lambda e, n: (0, 0, 0)
    else:
        widx = lambda e, n: (e, 0, n)
    fmt = "l24" if colscale is None else "f8l"
    in_specs = [
        pl.BlockSpec((None, T, K), lambda e, n: (e, 0, 0)),
        pl.BlockSpec((None, K // 8, bn), widx),
        pl.BlockSpec((None, K // 32, bn), widx),
    ]
    args = [x, w, s]
    if colscale is not None:
        in_specs.append(pl.BlockSpec((None, 1, bn), widx))
        args.append(colscale)
    return pl.pallas_call(
        functools.partial(_dense_kernel, kc=kc, zero_mode=zero_mode, unroll=unroll, fmt=fmt, nl=nl or bn,
                          bigdot=bigdot),
        out_shape=jax.ShapeDtypeStruct((E, T, N), out_dtype),
        grid=(E, N // bn),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((None, T, bn), lambda e, n: (e, 0, n)),
        scratch_shapes=[pltpu.VMEM((K // 32, bn), U32)]
        + ([pltpu.VMEM((K, bn), jnp.bfloat16 if colscale is None else jnp.float8_e5m2)] if bigdot else []),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=100 * 2**20,
        ),
        interpret=interpret,
    )(*args)
