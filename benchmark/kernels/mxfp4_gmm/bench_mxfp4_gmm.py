"""TPU v5e benchmark for the fused MXFP4 grouped matmul (single chip).

Run on one chip (other hosts of a multi-host slice just wait at a barrier):
    python benchmark/kernels/mxfp4_gmm/bench_mxfp4_gmm.py --sections micro,dense,layer

Prints one line per measurement and a final ``RESULTS_JSON {...}`` line.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
import traceback

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

RESULTS: dict = {}
E_LOCAL, HID, INTER = 7, 3584, 1536  # Kimi-K3 per-chip MoE layer at EP128 x expert-TP2
LAYER_WEIGHTS = E_LOCAL * (HID * 2 * INTER + INTER * HID)
HBM_BW = 820e9


_LOG_LINES: list = []
_LOG_STATE = {"gcs": None, "last_upload": 0.0, "t0": time.time(), "budget": 1e9}


def _upload_log(force=False):
    url = _LOG_STATE["gcs"]
    if not url or (not force and time.time() - _LOG_STATE["last_upload"] < 3):
        return
    _LOG_STATE["last_upload"] = time.time()
    try:
        from google.cloud import storage

        bucket, blob = url[5:].split("/", 1)
        storage.Client().bucket(bucket).blob(blob).upload_from_string("\n".join(_LOG_LINES) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"(log upload failed: {e})", flush=True)


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _LOG_LINES.append(line)
    _upload_log()


class BudgetExceeded(Exception):
    pass


def check_budget():
    if time.time() - _LOG_STATE["t0"] > _LOG_STATE["budget"]:
        raise BudgetExceeded()


def timeit(f, *args, n=10, reps=2):
    check_budget()
    t0 = time.perf_counter()
    out = f(*args)
    jax.block_until_ready(out)
    RESULTS.setdefault("first_call_s", []).append(round(time.perf_counter() - t0, 2))
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            out = f(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def guarded(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:400]}"
                log(f"  [{name}] FAILED {msg}")
                RESULTS.setdefault("failures", {})[f"{name}{a}{k}"[:200]] = msg[:300]
                return None

        return wrapper

    return deco


# ------------------------------------------------------------------------------ micro
@guarded("mxu_rate")
def mxu_rate(T, K, N, xdt, wdt, acc_dt, rep):
    def kern(x_ref, w_ref, o_ref):
        def body(r, acc):
            off = pl.multiple_of((r % 2) * K, K)
            return acc + jnp.dot(x_ref[...], w_ref[pl.ds(off, K), :], preferred_element_type=acc_dt)

        o_ref[...] = lax.fori_loop(0, rep, body, jnp.zeros((T, N), acc_dt))

    rng = np.random.default_rng(0)
    if jnp.issubdtype(xdt, jnp.integer):
        x = jnp.asarray(rng.integers(-3, 4, (T, K)), xdt)
    else:
        x = jnp.asarray(rng.standard_normal((T, K)) * 0.1, xdt)
    if jnp.issubdtype(wdt, jnp.integer):
        w = jnp.asarray(rng.integers(-3, 4, (2 * K, N)), wdt)
    else:
        w = jnp.asarray(rng.standard_normal((2 * K, N)) * 0.1, jnp.float32).astype(wdt)
    f = jax.jit(
        pl.pallas_call(
            kern,
            out_shape=jax.ShapeDtypeStruct((T, N), acc_dt),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 2**20),
        )
    )
    t = timeit(f, x, w, n=5)
    rate = rep * K * N / t
    key = f"mxu_{jnp.dtype(xdt).name}x{jnp.dtype(wdt).name}_T{T}"
    RESULTS[key] = rate
    log(f"  MXU {jnp.dtype(xdt).name:>14} x {jnp.dtype(wdt).name:<16} T={T:>4}: {rate/1e12:6.2f} T weights/s"
        f"  ({2*rate*T/1e12:6.1f} TFLOP/s)")
    return rate


@guarded("vpu_rate")
def vpu_rate(kind, nops=48, rows=512, cols=1024):
    def kern(x_ref, o_ref):
        def body(i, c):
            v = x_ref[pl.ds(pl.multiple_of(i * 8, 8), 8), :]
            for _ in range(nops // 3):
                if kind == "add":
                    v = (v + jnp.uint32(0x9E3779B9)) ^ (v >> 3)
                elif kind == "mul":
                    v = v * jnp.uint32(0x9E3779B1) + (v >> 3)
                elif kind == "shift":
                    v = (v << 3) ^ (v >> 5) ^ v
            o_ref[pl.ds(pl.multiple_of(i * 8, 8), 8), :] = v
            return c

        lax.fori_loop(0, rows // 8, body, 0, unroll=2)

    x = jnp.asarray(np.random.default_rng(0).integers(0, 2**32, (rows, cols), dtype=np.uint64).astype(np.uint32))
    f = jax.jit(pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((rows, cols), jnp.uint32)))
    t = timeit(f, x, n=10)
    vreg_ops = rows * cols / 1024 * (nops // 3) * 3
    rate = vreg_ops / t
    RESULTS[f"vpu_{kind}_vregops_per_s"] = rate
    log(f"  VPU {kind:>5}: {rate/1e9:6.2f} G vreg-ops/s = {rate/1.5e9:4.2f} per cycle @1.5GHz")


@guarded("denormal")
def denormal_test():
    # W[k, n] = bf16 bits k (k < 128 are subnormals) on the diagonal; x = 2**100.
    bits = np.zeros((128, 128), np.uint16)
    bits[np.arange(128), np.arange(128)] = np.arange(128, dtype=np.uint16)
    w = jnp.asarray(bits).view(jnp.bfloat16)
    x = jnp.full((8, 128), 2.0**100, jnp.bfloat16)

    def kern(x_ref, w_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    out = jax.jit(pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32)))(x, w)
    got = np.asarray(out)[0]
    want = np.arange(128) * 2.0**-133 * 2.0**100
    flushed = int(np.sum((got == 0) & (want != 0)))
    exact = int(np.sum(got == want))
    RESULTS["denormal_flushed"] = flushed
    log(f"  bf16 subnormal RHS in MXU: {exact}/128 exact, {flushed} flushed to zero")


@guarded("convert_rate")
def convert_rate(src_dtype, dst_dtype=jnp.bfloat16, K=2048, N=1024, rep=50):
    """In-kernel dtype conversion throughput (e.g. int8 -> bf16 for an int8-intermediate decode)."""

    def kern(w_ref, o_ref):
        def body(r, c):
            off = pl.multiple_of((r % 2) * K, K)
            o_ref[...] = w_ref[pl.ds(off, K), :].astype(dst_dtype)
            return c

        lax.fori_loop(0, rep, body, 0)

    rng = np.random.default_rng(0)
    if jnp.issubdtype(src_dtype, jnp.integer):
        w = jnp.asarray(rng.integers(-96, 97, (2 * K, N)), src_dtype)
    else:
        w = jnp.asarray(rng.standard_normal((2 * K, N)), jnp.float32).astype(src_dtype)
    f = jax.jit(
        pl.pallas_call(
            kern,
            out_shape=jax.ShapeDtypeStruct((K, N), dst_dtype),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 2**20),
        )
    )
    t = timeit(f, w, n=5)
    rate = rep * K * N / t
    RESULTS[f"convert_{jnp.dtype(src_dtype).name}_to_{jnp.dtype(dst_dtype).name}"] = rate
    log(f"  convert {jnp.dtype(src_dtype).name} -> {jnp.dtype(dst_dtype).name}: {rate/1e12:6.2f} T elem/s")


@guarded("hbm_stream")
def hbm_stream(bm, bn, rows=3136, cols=3072):
    def kern(w_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _():
            o_ref[...] = jnp.zeros_like(o_ref)

        o_ref[...] += w_ref[0:8, 0:128]

    nbn = cols // bn
    w = jnp.zeros((rows, cols), jnp.uint32)
    f = jax.jit(
        pl.pallas_call(
            kern,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
            grid=((rows // bm) * nbn,),
            in_specs=[pl.BlockSpec((bm, bn), lambda i: (i // nbn, i % nbn))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 2**20),
        )
    )
    t = timeit(f, w, n=10)
    bw = rows * cols * 4 / t
    RESULTS[f"hbm_stream_{bm}x{bn}"] = bw
    log(f"  HBM stream block {bm}x{bn} u32: {bw/1e9:6.1f} GB/s ({t*1e6:.1f} us for {rows*cols*4/1e6:.1f} MB)")


def run_micro():
    log("== micro: MXU rates (VMEM-resident operands) ==")
    for T in (8, 32, 128, 256):
        mxu_rate(T, 2048, 1024, jnp.bfloat16, jnp.bfloat16, jnp.float32, 400)
    for wdt in (jnp.float8_e5m2, jnp.float8_e4m3b11fnuz):
        for T in (8, 128):
            mxu_rate(T, 2048, 1024, jnp.bfloat16, wdt, jnp.float32, 400)
    for T in (8, 128):
        mxu_rate(T, 2048, 1024, jnp.int8, jnp.int8, jnp.int32, 400)
    log("== micro: VPU ==")
    for kind in ("add", "mul"):
        vpu_rate(kind)
    convert_rate(jnp.int8)
    convert_rate(jnp.float8_e5m2)
    log("== micro: denormals ==")
    denormal_test()
    log("== micro: HBM streaming ==")
    for bm, bn in ((448, 512), (448, 3072)):
        hbm_stream(bm, bn)


# ------------------------------------------------------------------------------ dense fused
def make_weights(K, N, seed):
    from sgl_jax.srt.kernels.mxfp4_gmm import format as F

    packed, scale = F.random_mxfp4((K, N), E_LOCAL, seed=seed, zero_frac=0.12)
    return packed, scale, F.encode_l24(packed)


@guarded("dense")
def dense_case(x, w, s, *, bn, kc, zero_mode, resident, unroll=1, tag="", want=None, colscale=None, nl=None,
               bigdot=False):
    from sgl_jax.srt.kernels.mxfp4_gmm.kernel import dense_mxfp4_matmul

    f = functools.partial(dense_mxfp4_matmul, bn=bn, kc=kc, zero_mode=zero_mode, resident=resident,
                          unroll=unroll, nl=nl, bigdot=bigdot)
    tag = tag + ("_bigdot" if bigdot else "")
    if colscale is not None:
        f = functools.partial(f, colscale=colscale)
    t = timeit(f, x, w, s)
    E, T, K = x.shape
    N = w.shape[-1]
    rate = E * K * N / t
    key = f"dense{tag}_T{T}_bn{bn}_nl{nl or bn}_kc{kc}_z{zero_mode}_res{int(resident)}_u{unroll}"
    RESULTS[key] = t * 1e6
    err = ""
    if want is not None:
        got = np.asarray(f(x, w, s).astype(jnp.float32))
        err = f" relerr={np.abs(got - want).max() / np.abs(want).max():.2e}"
    log(f"  dense{tag} T={T:>4} bn={bn:>4} nl={nl or bn:>4} kc={kc:>3} zero={zero_mode:>4} "
        f"resident={int(resident)} unroll={unroll}: {t*1e6:8.1f} us  {rate/1e12:5.2f} T w/s{err}")
    return t


def run_dense():
    log("== dense fused kernel (one matmul: 7 experts x 3584 -> 3072) ==")
    K, N = HID, 2 * INTER
    packed, scale, words = make_weights(K, N, 0)
    w = jnp.asarray(words)
    s8 = jnp.asarray(scale)
    rng = np.random.default_rng(1)
    f8_ok = RESULTS.get("mxu_bfloat16xfloat8_e5m2_T8") is not None or "micro" not in SECTIONS
    if f8_ok:
        from sgl_jax.srt.kernels.mxfp4_gmm import format as F

        w8, eg8, cs8 = map(jnp.asarray, F.encode_f8l(packed, scale))
    for T in (8, 256):
        x = jnp.asarray(rng.standard_normal((E_LOCAL, T, K)) * 0.5, jnp.bfloat16)
        want = None
        if T == 8:
            from sgl_jax.srt.kernels.mxfp4_gmm import format as F

            ref = F.dequant_reference(packed, scale)
            want = np.einsum("etk,ekn->etn", np.asarray(x, np.float32), ref)
        for bn in (512, 1024):
            for ez in ("mul", "none"):
                for res in (False, True):
                    dense_case(x, w, s8, bn=bn, kc=256, zero_mode=ez, resident=res,
                               want=want if (ez != "none" and not res) else None)
            if f8_ok and bn == 512:
                for ez in ("mul", "none"):
                    for res in (False, True):
                        dense_case(x, w8, eg8, bn=bn, kc=256, zero_mode=ez, resident=res, colscale=cs8,
                                   tag="_f8", want=want if (ez != "none" and not res) else None)
        for bn, nl in ((1024, 256), (512, 128), (512, 256)):
            for res in (False, True):
                dense_case(x, w, s8, bn=bn, nl=nl, kc=256, zero_mode="mul", resident=res)
        dense_case(x, w, s8, bn=512, kc=512, zero_mode="mul", resident=False)
        dense_case(x, w, s8, bn=512, kc=128, zero_mode="mul", resident=False)
        dense_case(x, w, s8, bn=512, kc=256, zero_mode="mul", resident=False, unroll=2)
        dense_case(x, w, s8, bn=1536, kc=256, zero_mode="mul", resident=False)
    # scales as uint32 instead of uint8
    x = jnp.asarray(rng.standard_normal((E_LOCAL, 8, K)) * 0.5, jnp.bfloat16)
    dense_case(x, w, s8.astype(jnp.uint32), bn=512, kc=256, zero_mode="mul", resident=False, tag="_s32")


def run_dense8():
    """Decode-regime kernel variants at 8 tokens/expert (7 experts, 3584 -> 3072)."""
    from sgl_jax.srt.kernels.mxfp4_gmm import format as F

    log("== dense fused kernel @ T=8 (7 experts x 3584 -> 3072, 77M weights) ==")
    K, N = HID, 2 * INTER
    packed, scale, words = make_weights(K, N, 0)
    w, s8 = jnp.asarray(words), jnp.asarray(scale)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((E_LOCAL, 8, K)) * 0.5, jnp.bfloat16)
    want = np.einsum("etk,ekn->etn", np.asarray(x, np.float32), F.dequant_reference(packed, scale))
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mul", resident=False, want=want)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mask", resident=False, want=want)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mul", resident=True)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mask", resident=True)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="none", resident=True)
    dense_case(x, w, s8, bn=1024, nl=256, kc=256, zero_mode="mul", resident=False)
    dense_case(x, w, s8, bn=1024, nl=256, kc=256, zero_mode="mul", resident=True)
    dense_case(x, w, s8, bn=512, nl=128, kc=256, zero_mode="mul", resident=True)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mul", resident=True, bigdot=True)
    dense_case(x, w, s8, bn=512, kc=256, zero_mode="mul", resident=False, bigdot=True, want=want)
    if RESULTS.get("mxu_bfloat16xfloat8_e5m2_T8") is not None:
        w8, eg8, cs8 = map(jnp.asarray, F.encode_f8l(packed, scale))
        dense_case(x, w8, eg8, bn=512, kc=256, zero_mode="mul", resident=False, colscale=cs8, tag="_f8", want=want)
        dense_case(x, w8, eg8, bn=512, kc=256, zero_mode="mul", resident=True, colscale=cs8, tag="_f8")
        dense_case(x, w8, eg8, bn=1024, nl=256, kc=256, zero_mode="mul", resident=True, colscale=cs8, tag="_f8")


# ------------------------------------------------------------------------------ ragged gmm
def _sizes(T, G, goff, M_total):
    sizes = np.zeros((G,), np.int32)
    sizes[goff : goff + E_LOCAL] = T
    rest = M_total - E_LOCAL * T
    if rest > 0:
        others = [g for g in range(G) if not (goff <= g < goff + E_LOCAL)]
        rng = np.random.default_rng(T)
        np.add.at(sizes, rng.choice(others, rest), 1)
    return jnp.asarray(sizes)


@guarded("gmm")
def gmm_case(x, ws, gs, goff, *, tm, bn, situ, cache=True, tag="", nl=None):
    from sgl_jax.srt.kernels.mxfp4_gmm.gmm import mxfp4_gmm

    if situ:
        w, s, wu, su = ws
        f = jax.jit(lambda x, w, s, wu, su, gs, goff: mxfp4_gmm(
            x, w, s, gs, goff, w_up=wu, scale_up=su, activation="situ_glu",
            tm=tm, bn=bn, nl=nl, cache=cache))
        args = (x, w, s, wu, su, gs, goff)
        nweights = 2 * w.shape[0] * x.shape[1] * w.shape[2]
    else:
        w, s = ws
        f = jax.jit(lambda x, w, s, gs, goff: mxfp4_gmm(
            x, w, s, gs, goff, tm=tm, bn=bn, nl=nl, cache=cache))
        args = (x, w, s, gs, goff)
        nweights = w.shape[0] * x.shape[1] * w.shape[2]
    t = timeit(f, *args)
    key = f"gmm{tag}_situ{int(situ)}_M{x.shape[0]}_tm{tm}_bn{bn}_nl{nl or bn}_c{int(cache)}"
    RESULTS[key] = t * 1e6
    log(f"  gmm{tag} situ={int(situ)} M={x.shape[0]:>5} tm={tm:>4} bn={bn:>4} nl={nl or bn:>4} cache={int(cache)}: "
        f"{t*1e6:8.1f} us "
        f"{nweights/t/1e12:5.2f} T w/s")
    return t


def run_gmm():
    log("== ragged GMM (megablox-style), w13 gate/up fused with SiTU: 7 local experts 3584 -> 2x1536 ==")
    K = HID
    pg, sg, wg = make_weights(K, INTER, 0)
    pu, su, wu = make_weights(K, INTER, 1)
    ws13 = tuple(map(jnp.asarray, (wg, sg, wu, su)))
    p2, s2, w2 = make_weights(INTER, HID, 2)
    ws2 = tuple(map(jnp.asarray, (w2, s2)))
    rng = np.random.default_rng(3)
    # all-local groups (G = 7)
    for T, tms in ((1, (16,)), (4, (32,)), (16, (16, 128)), (64, (64, 128)), (256, (128, 256, 512)),
                   (1024, (256, 512))):
        M = E_LOCAL * T
        gs = _sizes(T, E_LOCAL, 0, M)
        x = jnp.asarray(rng.standard_normal((M, K)) * 0.5, jnp.bfloat16)
        for tm in tms:
            for bn in (512, 768):
                gmm_case(x, ws13, gs, jnp.int32(0), tm=tm, bn=bn, situ=True)
            gmm_case(x, ws13, gs, jnp.int32(0), tm=tm, bn=768, nl=256, situ=True)
        x2 = jnp.asarray(rng.standard_normal((M, INTER)) * 0.5, jnp.bfloat16)
        for tm in tms:
            for bn in (512, 896):
                gmm_case(x2, ws2, gs, jnp.int32(0), tm=tm, bn=bn, situ=False)
            gmm_case(x2, ws2, gs, jnp.int32(0), tm=tm, bn=1792, nl=256, situ=False)
    # cache off at prefill (decode again on every m-tile) for comparison
    T = 256
    gs = _sizes(T, E_LOCAL, 0, E_LOCAL * T)
    x = jnp.asarray(rng.standard_normal((E_LOCAL * T, K)) * 0.5, jnp.bfloat16)
    gmm_case(x, ws13, gs, jnp.int32(0), tm=128, bn=512, situ=True, cache=False)
    # expert-parallel decode: B=256 tokens x top-8 = 2048 rows over 896 experts, 7 local
    for T in (2, 8):
        gs = _sizes(T, 896, 448, 2048)
        x = jnp.asarray(rng.standard_normal((2048, K)) * 0.5, jnp.bfloat16)
        for tm in (16, 64):
            gmm_case(x, ws13, gs, jnp.int32(448), tm=tm, bn=768, situ=True, tag="_ep")


# ------------------------------------------------------------------------------ layer
def situ(h):
    g, u = jnp.split(h.astype(jnp.float32), 2, axis=-1)
    return (4 * jnp.tanh(g / 4) * jax.nn.sigmoid(g) * 25 * jnp.tanh(u / 25)).astype(jnp.bfloat16)


def run_layer():
    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm as megablox_gmm
    from sgl_jax.srt.kernels.mxfp4_gmm.gmm import mxfp4_gmm
    from sgl_jax.srt.layers.quantization.mxfp4 import dequant_mxfp4

    log("== full MoE layer per chip (ragged rows): gate/up 3584->2x1536 + SiTU + down 1536->3584, 7 experts ==")
    pg, sg, wg = make_weights(HID, INTER, 0)
    pu, su, wu = make_weights(HID, INTER, 1)
    p2, s2, w2 = make_weights(INTER, HID, 2)
    wg, sg_d, wu, su_d, w2, s2_d = map(jnp.asarray, (wg, sg, wu, su, w2, s2))
    deq = jax.jit(lambda p, s: dequant_mxfp4(p, s, axis=-2))
    bg, bu, b2 = deq(jnp.asarray(pg), jnp.asarray(sg)), deq(jnp.asarray(pu), jnp.asarray(su)), deq(
        jnp.asarray(p2), jnp.asarray(s2))
    pg, sg, pu, su, p2, s2 = map(jnp.asarray, (pg, sg, pu, su, p2, s2))
    b13 = jnp.concatenate([bg, bu], axis=-1)

    def f_bf16_dense(x, b13, b2):  # x [E, T, K]: XLA einsum with resident bf16 weights (the 0.54 ms baseline)
        h = jnp.einsum("etk,ekn->etn", x, b13)
        return jnp.einsum("etn,enk->etk", situ(h), b2)

    def f_megablox_bf16(x, gs, bg, bu, b2):  # current MXFP4EPMoE path minus the dequant: bf16 megablox gmm
        g = megablox_gmm(x, bg, gs, preferred_element_type=jnp.bfloat16)
        u = megablox_gmm(x, bu, gs, preferred_element_type=jnp.bfloat16)
        h = situ(jnp.concatenate([g, u], axis=-1))
        return megablox_gmm(h, b2, gs, preferred_element_type=jnp.bfloat16)

    def f_current(x, gs, pg, sg, pu, su, p2, s2):  # XLA dequant + bf16 megablox (what moe_mxfp4.py does today)
        dg, du, d2 = deq(pg, sg), deq(pu, su), deq(p2, s2)
        g = megablox_gmm(x, dg, gs, preferred_element_type=jnp.bfloat16)
        u = megablox_gmm(x, du, gs, preferred_element_type=jnp.bfloat16)
        h = situ(jnp.concatenate([g, u], axis=-1))
        return megablox_gmm(h, d2, gs, preferred_element_type=jnp.bfloat16)

    def f_mx(x, gs, wg, sg_d, wu, su_d, w2, s2_d, *, tm, bn13, bn2, nl, zm):
        use_cache = x.shape[0] > tm * wg.shape[0]
        h = mxfp4_gmm(x, wg, sg_d, gs, w_up=wu, scale_up=su_d, activation="situ_glu", tm=tm, bn=bn13,
                      nl=nl if nl and bn13 % nl == 0 else None, zero_mode=zm, act_input_dtype=jnp.bfloat16,
                      cache=use_cache, zero_unvisited=False)
        return mxfp4_gmm(h, w2, s2_d, gs, tm=tm, bn=bn2, nl=nl if nl and bn2 % nl == 0 else None,
                         zero_mode=zm, cache=use_cache)

    f_dense = jax.jit(f_bf16_dense)
    f_mb = jax.jit(f_megablox_bf16)
    f_cur = jax.jit(f_current)
    rng = np.random.default_rng(2)
    for T in LAYER_TS:
        M = E_LOCAL * T
        gs = jnp.full((E_LOCAL,), T, jnp.int32)
        x3 = jnp.asarray(rng.standard_normal((E_LOCAL, T, HID)) * 0.5, jnp.bfloat16)
        x = x3.reshape(M, HID)
        line = f"  T/expert={T:>5}:"
        ref = np.asarray(f_dense(x3, b13, b2).astype(jnp.float32)).reshape(M, HID)
        for name, fn, args in (("bf16-einsum", f_dense, (x3, b13, b2)),
                               ("bf16-megablox", f_mb, (x, gs, bg, bu, b2)),
                               ("dequant+megablox", f_cur, (x, gs, pg, sg, pu, su, p2, s2))):
            try:
                t = timeit(fn, *args)
                RESULTS[f"layer_{name}_T{T}"] = t * 1e3
                line += f" {name} {t*1e3:7.3f} ms |"
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                line += f" {name} FAILED {type(e).__name__} |"
        tm = 16 if T <= 8 else (64 if T <= 64 else (256 if T <= 256 else 512))
        best = None
        for bn13, bn2, nl, zm in LAYER_CFGS:
            if True:
                try:
                    check_budget()
                    f = jax.jit(functools.partial(f_mx, tm=tm, bn13=bn13, bn2=bn2, nl=nl, zm=zm))
                    t = timeit(f, x, gs, wg, sg_d, wu, su_d, w2, s2_d)
                    RESULTS[f"layer_mx_T{T}_tm{tm}_bn{bn13}-{bn2}_nl{nl}_{zm}"] = t * 1e3
                    log(f"    mx tm={tm} bn={bn13}/{bn2} nl={nl} zero={zm}: {t*1e3:.3f} ms")
                    if best is None or t < best[0]:
                        got = np.asarray(f(x, gs, wg, sg_d, wu, su_d, w2, s2_d).astype(jnp.float32))
                        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
                        best = (t, tm, f"{bn13}/{bn2} nl={nl} {zm}", "", err)
                except BudgetExceeded:
                    raise
                except Exception as e:  # noqa: BLE001
                    log(f"    mx tm={tm} bn={bn13}/{bn2} zero={zm} FAILED {type(e).__name__}: {str(e)[:300]}")
        if best:
            t, tm, bn13, bn2, err = best
            RESULTS[f"layer_mx_best_T{T}"] = t * 1e3
            line += (f" MXFP4-fused {t*1e3:7.3f} ms ({LAYER_WEIGHTS/t/1e12:5.2f} T w/s, tm={tm} bn={bn13},"
                     f" rel.err vs bf16 {err:.1e})")
        log(line)
    log(f"  (HBM ideal: MXFP4 {LAYER_WEIGHTS*0.53125/HBM_BW*1e3:.3f} ms, bf16 {LAYER_WEIGHTS*2/HBM_BW*1e3:.3f} ms)")


SECTIONS: list = []
LAYER_TS = (1, 8, 64, 256, 1024)
LAYER_CFGS = ((512, 512, None, "mul"), (512, 896, None, "mul"), (512, 512, None, "mask"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="micro,dense8,layer,dense,gmm")
    ap.add_argument("--gcs-log", default=os.environ.get("BENCH_GCS_LOG", ""))
    ap.add_argument("--budget-s", type=float, default=float(os.environ.get("BENCH_BUDGET_S", "1e9")))
    a = ap.parse_args()
    SECTIONS[:] = a.sections.split(",")
    _LOG_STATE["budget"] = a.budget_s
    rank = int(os.environ.get("NODE_RANK", "0"))
    nh = int(os.environ.get("NUM_HOSTS", "1"))
    if nh > 1:
        jax.distributed.initialize(coordinator_address=os.environ["COORD"] + ":8476", num_processes=nh,
                                   process_id=rank)
    if rank == 0:
        _LOG_STATE["gcs"] = a.gcs_log or None
        _LOG_STATE["t0"] = time.time()
        d = jax.local_devices()[0]
        log(f"device: {d.device_kind}, jax {jax.__version__}, local devices {len(jax.local_devices())}")
        with jax.default_device(d):
            for sec in a.sections.split(","):
                t0 = time.time()
                try:
                    {"micro": run_micro, "dense8": run_dense8, "dense": run_dense, "gmm": run_gmm,
                     "layer": run_layer}[sec]()
                except BudgetExceeded:
                    log(f"-- time budget exhausted during section {sec}")
                    break
                except Exception:  # noqa: BLE001
                    log(traceback.format_exc()[-2000:])
                log(f"-- section {sec} took {time.time()-t0:.0f}s")
        log("RESULTS_JSON " + json.dumps(RESULTS))
        _upload_log(force=True)
    if nh > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("bench_done")
    log(f"rank {rank} done")


if __name__ == "__main__":
    main()
