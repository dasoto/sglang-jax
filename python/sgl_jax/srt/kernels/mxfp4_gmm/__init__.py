"""Fused MXFP4 grouped matmul for TPU v5e (Kimi-K3 routed experts).

Public API:
  * :func:`prepare_l24` / ``format.encode_l24`` -- load-time re-encoding of the k-major
    compressed-tensors packed weights ``uint8 [E, K/2, N]`` into the kernel layout
    ``uint32 [E, K/8, N]`` (same bytes). Scales stay ``uint8 [E, K/32, N]``.
  * :func:`mxfp4_gmm` -- megablox-style ragged grouped matmul (optionally with fused
    SiTU-GLU over separate gate/up weights).
  * :func:`mxfp4_moe_mlp` -- gate/up (+SiTU) and down projections for EPMoE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.mxfp4_gmm.format import encode_l24, encode_l24_jax
from sgl_jax.srt.kernels.mxfp4_gmm.gmm import mxfp4_gmm, mxfp4_gmm_reference, situ_glu_f32

__all__ = [
    "encode_l24",
    "encode_l24_jax",
    "mxfp4_gmm",
    "mxfp4_gmm_reference",
    "mxfp4_moe_mlp",
    "pick_row_tile",
    "prepare_l24",
    "situ_glu_f32",
]


def prepare_l24(packed: jax.Array) -> jax.Array:
    """uint8 ``[..., K/2, N]`` (checkpoint k-major) -> uint32 ``[..., K/8, N]`` kernel layout.

    jit-able and blockwise along K (64 rows): call it once after loading, e.g.
    ``jax.jit(prepare_l24, out_shardings=sharding)(packed)`` with the same PartitionSpec as the
    packed weight; for K-sharded tensors (down projection under expert-TP) each shard must hold
    a multiple of 64 rows of K (Kimi-K3: 1536 per shard).
    """
    return encode_l24_jax(packed)


def pick_row_tile(m_rows: int, num_groups: int, *, lo: int = 16, hi: int = 512) -> int:
    """Row tile for ``m_rows`` routed rows spread over ``num_groups`` (global) experts."""
    avg = max(1, m_rows // max(1, num_groups))
    tm = lo
    while tm < hi and tm < avg:
        tm *= 2
    return tm


def mxfp4_moe_mlp(
    x: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array | None,
    w_gate: jax.Array,
    s_gate: jax.Array,
    w_up: jax.Array,
    s_up: jax.Array,
    w_down: jax.Array,
    s_down: jax.Array,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    act_input_dtype=jnp.bfloat16,
    out_dtype=jnp.bfloat16,
    tm: int | None = None,
    bn_up: int = 512,
    bn_down: int = 512,
    nl: int | None = None,
    zero_mode: str = "mul",
    zero_unvisited: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Routed-expert MLP over sorted rows: ``down(SiTU(x @ gate) * 25 tanh((x @ up) / 25))``.

    x: [M, H] bf16 sorted by expert; group_sizes: int32 [G]; group_offset: first local expert.
    w_gate/w_up: l24 ``[E_l, H/8, I_l]``; w_down: l24 ``[E_l, I_l/8, H]``; scales uint8 E8M0.
    ``act_input_dtype=bf16`` reproduces EPMoE (its GMMs emit bf16 before the activation).
    """
    if tm is None:
        tm = pick_row_tile(x.shape[0], group_sizes.shape[0])
    n_up, n_down = w_gate.shape[-1], w_down.shape[-1]
    bn_up = bn_up if n_up % bn_up == 0 else n_up
    bn_down = bn_down if n_down % bn_down == 0 else n_down
    # VMEM decoded-weight cache is only useful when the input has more m-tiles than local experts.
    use_cache = x.shape[0] > tm * w_gate.shape[0]
    h = mxfp4_gmm(
        x,
        w_gate,
        s_gate,
        group_sizes,
        group_offset,
        w_up=w_up,
        scale_up=s_up,
        activation="situ_glu",
        tm=tm,
        bn=bn_up,
        nl=nl if nl and bn_up % nl == 0 else None,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        act_input_dtype=act_input_dtype,
        out_dtype=x.dtype,
        zero_mode=zero_mode,
        cache=use_cache,
        zero_unvisited=False,  # the down GMM only reads local rows
        interpret=interpret,
    )
    return mxfp4_gmm(
        h,
        w_down,
        s_down,
        group_sizes,
        group_offset,
        tm=tm,
        bn=bn_down,
        nl=nl if nl and bn_down % nl == 0 else None,
        out_dtype=out_dtype,
        zero_mode=zero_mode,
        cache=use_cache,
        zero_unvisited=zero_unvisited,
        interpret=interpret,
    )
