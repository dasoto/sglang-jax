"""Ragged (megablox-style) grouped matmul over l24-encoded MXFP4 expert weights.

    out[rows of group g] = lhs[rows of group g] @ W[g - group_offset]

``lhs`` rows are sorted by expert (``group_sizes`` like megablox ``gmm``); only the
groups ``[group_offset, group_offset + E_local)`` held by this shard are computed.

Grid ``(visit, n_tile)``: a *visit* is one (m-tile, group) pair from
``make_group_metadata`` (an m-tile that straddles a group boundary is visited once
per group).  On the first visit of a group the kernel decodes each weight column
tile (K x bn) once -- fused with the matmul so VPU decode and MXU work overlap --
and keeps the decoded bf16 expert in a VMEM cache (``K x N``); later visits of the
same group (prefill: many m-tiles per expert) reuse it.  The weight index map pins
non-first visits to the last fetched block, so packed weights are read from HBM
exactly once per call and decoded exactly once.

``activation="situ_glu"`` takes separate gate and up weights and returns
``SiTU(lhs @ Wg) * (25 tanh((lhs @ Wu) / 25))`` with SiTU(g) = 4 tanh(g/4) sigmoid(g),
computed in f32 on the accumulators (Kimi-K3's expert activation).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import make_group_metadata
from sgl_jax.srt.kernels.mxfp4_gmm.kernel import U32, decode_l24_chunk, scale_operand


def situ_glu_f32(g, u, beta: float = 4.0, linear_beta: float | None = 25.0):
    """SiTU-GLU in f32: beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)."""
    g = g.astype(jnp.float32)
    u = u.astype(jnp.float32)
    act = beta * jnp.tanh(g * (1.0 / beta)) * jax.nn.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * jnp.tanh(u * (1.0 / linear_beta))
    return act * u


def _gmm_kernel(
    # scalar prefetch (SMEM)
    goffs_ref,  # [G + 1] group row offsets
    gids_ref,  # [V] group id per visit
    mids_ref,  # [V] m-tile id per visit
    flags_ref,  # [V] bit0: first visit of its group, bit1: first visit of its m-tile,
    #             bit2: the next visit has the same group (decoded tile must be cached)
    goff_ref,  # [1] group_offset
    *refs,
    n_w: int,
    tm: int,
    bn: int,
    kc: int,
    nl: int,
    zero_mode: str,
    activation: str | None,
    cache: bool,
    situ_beta: float,
    situ_linear_beta: float | None,
    act_input_dtype,
):
    lhs_ref = refs[0]
    w_refs = refs[1 : 1 + 2 * n_w : 2]
    s_refs = refs[2 : 2 + 2 * n_w : 2]
    out_ref = refs[1 + 2 * n_w]
    scratch = refs[2 + 2 * n_w :]
    g_scrs = scratch[:n_w]
    acc_refs = scratch[n_w : 2 * n_w]
    wc_refs = scratch[2 * n_w : 3 * n_w] if cache else (None,) * n_w

    v = pl.program_id(0)
    n = pl.program_id(1)
    gid = gids_ref[v]
    flags = flags_ref[v]
    first_g = (flags & 1) == 1
    first_m = (flags & 2) == 2
    keep = (flags & 4) == 4
    K = lhs_ref.shape[1]
    nk = K // kc
    col = pl.ds(pl.multiple_of(n * bn, bn), bn)

    # Small accumulators ride in vregs through the K loop; big (prefill) tiles accumulate in VMEM.
    carry_in_vregs = tm * bn * 4 <= 128 * 1024

    def decode_mm(w_ref, s_ref, g_scr, wc_ref, acc_ref):
        g_scr[...] = scale_operand(s_ref[...].astype(U32), zero_mode=zero_mode)

        def chunk(c):
            rows = pl.ds(pl.multiple_of(c * kc, kc), kc)
            xk = lhs_ref[:, rows]
            parts = []
            for l in range(bn // nl):
                lanes = None if nl == bn else pl.ds(l * nl, nl)
                wd = decode_l24_chunk(w_ref, g_scr, c, nblk=kc // 64, zero_mode=zero_mode, lanes=lanes)
                if wc_ref is not None:
                    wc_ref[rows, pl.ds(pl.multiple_of(n * bn + l * nl, nl), nl)] = wd
                parts.append(jnp.dot(xk, wd, preferred_element_type=jnp.float32))
            return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)

        if carry_in_vregs:
            acc_ref[...] = lax.fori_loop(0, nk, lambda c, a: a + chunk(c), jnp.zeros((tm, bn), jnp.float32))
        else:
            acc_ref[...] = jnp.zeros_like(acc_ref)

            def body(c, carry):
                acc_ref[...] += chunk(c)
                return carry

            lax.fori_loop(0, nk, body, 0)

    def cached_mm(wc_ref, acc_ref):
        acc_ref[...] = jnp.dot(lhs_ref[...], wc_ref[:, col], preferred_element_type=jnp.float32)

    for i in range(n_w):
        if cache:
            branch = jnp.where(first_g, jnp.where(keep, 0, 1), 2)
            lax.switch(
                branch,
                [
                    functools.partial(decode_mm, w_refs[i], s_refs[i], g_scrs[i], wc_refs[i], acc_refs[i]),
                    functools.partial(decode_mm, w_refs[i], s_refs[i], g_scrs[i], None, acc_refs[i]),
                    functools.partial(cached_mm, wc_refs[i], acc_refs[i]),
                ],
            )
        else:
            decode_mm(w_refs[i], s_refs[i], g_scrs[i], None, acc_refs[i])

    # Epilogue (activation + masked store), in row chunks to bound code size for big tiles.
    start = goffs_ref[gid]
    end = goffs_ref[gid + 1]
    ec = min(tm, 64)

    def epilogue(r, carry):
        rs = pl.ds(pl.multiple_of(r * ec, ec), ec)
        if activation == "situ_glu":
            g = acc_refs[0][rs, :]
            u = acc_refs[1][rs, :]
            if act_input_dtype is not None:  # match a reference that rounds the GMM outputs first
                g = g.astype(act_input_dtype)
                u = u.astype(act_input_dtype)
            y = situ_glu_f32(g, u, situ_beta, situ_linear_beta)
        else:
            y = acc_refs[0][rs, :]
        rows = mids_ref[v] * tm + r * ec + lax.broadcasted_iota(jnp.int32, (ec, bn), 0)
        mask = (rows >= start) & (rows < end)
        old = out_ref[rs, col]
        old = jnp.where(first_m, jnp.zeros_like(old), old)
        out_ref[rs, col] = jnp.where(mask, y.astype(out_ref.dtype), old)
        return carry

    lax.fori_loop(0, tm // ec, epilogue, 0)


def _visit_flags(group_ids, m_tile_ids, num_visits):
    v = jnp.arange(group_ids.shape[0], dtype=jnp.int32)
    prev_g = jnp.concatenate([jnp.full((1,), -1, jnp.int32), group_ids[:-1]])
    prev_m = jnp.concatenate([jnp.full((1,), -1, jnp.int32), m_tile_ids[:-1]])
    next_g = jnp.concatenate([group_ids[1:], jnp.full((1,), -1, jnp.int32)])
    keep = (group_ids == next_g) & (v + 1 < num_visits)
    return (
        (group_ids != prev_g).astype(jnp.int32)
        | (2 * (m_tile_ids != prev_m).astype(jnp.int32))
        | (4 * keep.astype(jnp.int32))
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "tm",
        "bn",
        "kc",
        "nl",
        "activation",
        "out_dtype",
        "zero_mode",
        "cache",
        "zero_unvisited",
        "situ_beta",
        "situ_linear_beta",
        "act_input_dtype",
        "vmem_limit_bytes",
        "interpret",
    ),
)
def mxfp4_gmm(
    lhs: jax.Array,
    w: jax.Array,
    scale: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array | None = None,
    *,
    w_up: jax.Array | None = None,
    scale_up: jax.Array | None = None,
    activation: str | None = None,
    tm: int = 128,
    bn: int = 512,
    kc: int = 256,
    nl: int | None = None,
    out_dtype=jnp.bfloat16,
    zero_mode: str = "mul",
    cache: bool = True,
    zero_unvisited: bool = True,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    act_input_dtype=None,
    vmem_limit_bytes: int = 100 * 2**20,
    interpret: bool = False,
) -> jax.Array:
    """Grouped matmul ``lhs @ W[group]`` with l24-encoded MXFP4 weights (see format.py).

    Args:
      lhs: [M, K] bf16, rows sorted by group.
      w: uint32 [E_local, K/8, N] (``format.encode_l24`` of the k-major packed weights).
      scale: uint8 [E_local, K/32, N] E8M0 scales (checkpoint k-major layout).
      group_sizes: int32 [G] rows per group (G >= E_local, global expert count).
      group_offset: scalar int32, first global group held in ``w`` (expert sharding).
      w_up, scale_up: second weight (up projection) for ``activation="situ_glu"``.
      activation: None or "situ_glu".
      tm, bn, kc: row tile, output column tile, K chunk (multiple of 64).
      nl: decode/matmul lane sub-tile inside a column tile (bn % nl == 0); None -> bn.
      zero_mode: "mul" | "mask" (exact zeros; integer-multiply or shift/and gating) or "none"
        (benchmark only: zero codes decode as 0.75 * scale).
      zero_unvisited: zero rows outside the local groups (megablox gmm semantics).
      situ_beta, situ_linear_beta: SiTU-GLU constants (Kimi-K3: 4 and 25).
      act_input_dtype: round the gate/up accumulators to this dtype before SiTU
        (jnp.bfloat16 reproduces EPMoE, whose GMMs emit bf16); None keeps f32.
    Returns:
      [M, N] in ``out_dtype``.
    """
    M, K = lhs.shape
    E_l, K8, N = w.shape
    if K8 * 8 != K:
        raise ValueError(f"w {w.shape} does not match lhs K={K}")
    if scale.shape != (E_l, K // 32, N):
        raise ValueError(f"scale {scale.shape} != {(E_l, K // 32, N)}")
    nl = nl or bn
    if K % kc:  # largest multiple-of-64 chunk <= kc that divides K
        kc = next((c for c in range(kc - kc % 64, 63, -64) if K % c == 0), kc)
    if N % bn or K % kc or kc % 64 or bn % nl:
        raise ValueError(f"bad tiling N={N} bn={bn} K={K} kc={kc} nl={nl}")
    n_w = 1
    weights = [w, scale]
    if activation == "situ_glu":
        if w_up is None or scale_up is None or w_up.shape != w.shape or scale_up.shape != scale.shape:
            raise ValueError("situ_glu needs w_up/scale_up shaped like w/scale")
        n_w = 2
        weights += [w_up, scale_up]
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation}")
    if group_offset is None:
        group_offset = jnp.zeros((), jnp.int32)
    group_offset = jnp.asarray(group_offset, jnp.int32).reshape(())
    group_sizes = group_sizes.astype(jnp.int32)
    if M == 0:
        return jnp.zeros((0, N), out_dtype)

    pad = (-M) % tm
    if pad:
        lhs = jnp.pad(lhs, ((0, pad), (0, 0)))
        group_sizes = group_sizes.at[-1].add(pad)
    Mp = M + pad
    meta, num_visits = make_group_metadata(
        group_sizes=group_sizes,
        m=Mp,
        tm=tm,
        start_group=group_offset,
        num_nonzero_groups=E_l,
        visit_empty_groups=False,
    )
    flags = _visit_flags(meta.group_ids, meta.m_tile_ids, num_visits)
    tiles_n = N // bn

    def lhs_map(v, n, goffs, gids, mids, flg, goff):
        return mids[v], 0

    def w_map(v, n, goffs, gids, mids, flg, goff):
        nn = jnp.where((flg[v] & 1) == 1, n, tiles_n - 1) if cache else n
        return gids[v] - goff[0], 0, nn

    def out_map(v, n, goffs, gids, mids, flg, goff):
        return mids[v], 0

    in_specs = [pl.BlockSpec((tm, K), lhs_map)]
    for _ in range(n_w):
        in_specs += [
            pl.BlockSpec((None, K // 8, bn), w_map),
            pl.BlockSpec((None, K // 32, bn), w_map),
        ]
    scratch = [pltpu.VMEM((K // 32, bn), U32) for _ in range(n_w)]
    scratch += [pltpu.VMEM((tm, bn), jnp.float32) for _ in range(n_w)]
    if cache:
        scratch += [pltpu.VMEM((K, N), jnp.bfloat16) for _ in range(n_w)]

    kernel = functools.partial(
        _gmm_kernel,
        n_w=n_w,
        tm=tm,
        bn=bn,
        kc=kc,
        nl=nl,
        zero_mode=zero_mode,
        activation=activation,
        cache=cache,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        act_input_dtype=act_input_dtype,
    )
    flops = 2 * Mp * K * N * n_w
    bytes_accessed = Mp * K * 2 + E_l * n_w * (K * N // 2 + K * N // 32) + Mp * N * jnp.dtype(out_dtype).itemsize
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((Mp, N), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(num_visits, tiles_n),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((tm, N), out_map),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
        interpret=interpret,
        name=f"mxfp4_gmm_{activation or 'none'}_k{K}_n{N}_tm{tm}_bn{bn}",
    )(
        meta.group_offsets,
        meta.group_ids,
        meta.m_tile_ids,
        flags,
        group_offset[None],
        lhs,
        *weights,
    )
    if zero_unvisited:
        start = meta.group_offsets[group_offset]
        end = meta.group_offsets[group_offset + E_l]
        r = lax.broadcasted_iota(jnp.int32, (Mp, 1), 0)
        out = jnp.where((r >= start) & (r < end), out, jnp.zeros((), out.dtype))
    return out[:M] if pad else out


def mxfp4_gmm_reference(
    lhs,
    w_deq,
    group_sizes,
    group_offset=0,
    *,
    w_up_deq=None,
    activation=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    act_input_dtype=None,
):
    """Plain-JAX reference: w_deq [E_local, K, N] float weights; rows outside local groups -> 0."""
    M = lhs.shape[0]
    E_l = w_deq.shape[0]
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    row = jnp.arange(M)
    gid = jnp.sum(row[:, None] >= ends[None, :], axis=1)  # global group of each row
    local = gid - group_offset
    valid = (local >= 0) & (local < E_l)
    li = jnp.clip(local, 0, E_l - 1)
    x = lhs.astype(jnp.float32)
    hp = jax.lax.Precision.HIGHEST
    g = jnp.einsum("mk,mkn->mn", x, w_deq.astype(jnp.float32)[li], precision=hp)
    if activation == "situ_glu":
        u = jnp.einsum("mk,mkn->mn", x, w_up_deq.astype(jnp.float32)[li], precision=hp)
        if act_input_dtype is not None:
            g, u = g.astype(act_input_dtype), u.astype(act_input_dtype)
        g = situ_glu_f32(g, u, situ_beta, situ_linear_beta)
    del starts
    return jnp.where(valid[:, None], g, 0.0)
