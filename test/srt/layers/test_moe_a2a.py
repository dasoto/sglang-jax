"""MXFP4EPMoE all-to-all prefill path (SGLANG_JAX_MOE_A2A) vs the dense mesh-mode path.

The a2a path sends each chip's tokens only to the chips of the experts they route to (ragged
all-to-alls over the model mesh) and returns the expert outputs the same way, so it must give
the dense path's output and row sums of squares for every token, with padding rows zero. A
layer whose routing would overflow the receive buffers falls back to the dense path.

XLA:CPU has no ragged-all-to-all, so these tests run the layer's all-gather emulation of it;
``test_ragged_emulation_contract`` checks the emulation against the op's documented semantics.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_a2a.py -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.mxfp4_gmm import format as F
from sgl_jax.srt.layers.moe_mxfp4 import MXFP4EPMoE, _ragged_all_to_all

# 8 devices: data 2 x tensor 4; MoE expert 4 x tensor 2 (2 local experts per rank).
H, I, E, TOPK, EP = 256, 256, 8, 3, 4

needs8 = pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")


def _mesh():
    return jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _layers(mesh, seed=7, emit_sumsq=False):
    common = dict(
        hidden_size=H,
        num_experts=E,
        num_experts_per_tok=TOPK,
        ep_size=EP,
        mesh=mesh,
        intermediate_dim=I,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    dense = MXFP4EPMoE(use_gmm_kernel=True, **common)
    a2a = MXFP4EPMoE(use_gmm_kernel=True, **common)
    for layer in (dense, a2a):
        layer.rs_combine = layer.rs_combine_mesh = True
        layer.hier_combine = True
        layer.local_window_min_rows = 1
        layer.local_window_capacity = 16
        layer.local_window_align = True
        layer.compact_divisors = ()
        layer.emit_sumsq = emit_sumsq
    a2a.a2a = True
    for name, (k, n), s in (("w1", (H, I), seed), ("w3", (H, I), seed + 1), ("w2", (I, H), seed + 2)):
        packed, scale = F.random_mxfp4((k, n), E, seed=s, zero_frac=0.1)
        for layer in (dense, a2a):
            for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                param = getattr(layer, f"{name}_{suffix}")
                param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
    assert dense.prepare_gmm_kernel_weights() and a2a.prepare_gmm_kernel_weights()
    return dense, a2a


def _inputs(mesh, T, mask_np, seed, hot=None):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, E))
    if hot is not None:  # every token's first choice is expert `hot`
        logits[:, hot] += 100.0
    ids = np.argsort(-logits, axis=1)[:, :TOPK].astype(np.int32)
    w = np.take_along_axis(logits, ids, 1)
    w = w - w.max(1, keepdims=True)
    w = (np.exp(w) / np.exp(w).sum(1, keepdims=True)).astype(np.float32)
    route = NamedSharding(mesh, P("data", None))
    hs = jax.device_put(
        jnp.asarray(rng.standard_normal((T, H)), jnp.bfloat16),
        NamedSharding(mesh, P("data", "tensor")),
    )
    tw = jax.device_put(jnp.asarray(w), route)
    ti = jax.device_put(jnp.asarray(ids), route)
    mask = jax.device_put(jnp.asarray(mask_np), NamedSharding(mesh, P("data")))
    return hs, tw, ti, mask


def _run(layer, hs, tw, ti, mask):
    out = jax.jit(lambda a, b, c, m: layer(a, b, c, token_mask=m))(hs, tw, ti, mask)
    if isinstance(out, tuple):
        assert out[0].sharding.spec == P("data", "tensor") and out[1].sharding.spec == P("data")
        return np.asarray(out[0].astype(jnp.float32)), np.asarray(out[1])
    assert out.sharding.spec == P("data", "tensor")
    return np.asarray(out.astype(jnp.float32)), None


def _direct_a2a(layer, hs, tw, ti, mask):
    """_a2a_forward without the overflow cond: proves the a2a branch itself is correct."""
    blocks = layer._a2a_blocks(hs, ti)
    assert blocks == (2, 4)

    def f(a, b, c, m):
        ids, ids_all, cnt = layer._a2a_routing(c, m, blocks)
        r_cap = layer._a2a_capacity(c.shape[0] * c.shape[1])
        return layer._a2a_forward(a, b, ids, ids_all, cnt, blocks=blocks, r_cap=r_cap)

    out = jax.jit(f)(hs, tw, ti, mask)
    out = out[0] if isinstance(out, tuple) else out
    return np.asarray(out.astype(jnp.float32))


def _assert_close(out, ref, mask_np):
    assert np.all(out[~mask_np] == 0)
    if not mask_np.any():
        return
    scale = np.abs(ref).max()
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * scale)
    assert np.abs(out - ref).mean() < 3e-3 * np.abs(ref[mask_np]).mean()


@needs8
def test_ragged_emulation_contract():
    """Chip c sends sizes[c, j] rows from in_off[c, j] to chip j, landing at out_off[c, j]."""
    mesh = _mesh()
    n, S, R, W = 8, 24, 40, 4
    rng = np.random.default_rng(0)
    sizes = rng.integers(0, 4, size=(n, n)).astype(np.int32)
    in_off = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(sizes, 1)[:, :-1]], 1)
    out_off = np.concatenate([np.zeros((1, n), np.int32), np.cumsum(sizes, 0)[:-1]], 0)
    x = rng.standard_normal((n, S, W)).astype(np.float32)
    spec = P(("data", "tensor"), None)

    def f(xs, io, ss, oo, rs):
        y = _ragged_all_to_all(
            xs[0], jnp.full((R, W), -1.0), io[0], ss[0], oo[0], rs[0],
            axis_name=("data", "tensor"), emulate=True,
        )
        return y[None]

    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))  # noqa: E731
    with jax.set_mesh(mesh):
        got = jax.jit(
            shard_map(f, mesh=mesh, in_specs=(P(("data", "tensor"), None, None),) + (spec,) * 4,
                      out_specs=P(("data", "tensor"), None, None), check_vma=False)
        )(put(x), put(in_off), put(sizes), put(out_off), put(sizes.T.copy()))
    got = np.asarray(got)
    want = np.full((n, R, W), -1.0, np.float32)
    for c in range(n):
        for j in range(n):
            m = sizes[c, j]
            want[j, out_off[c, j] : out_off[c, j] + m] = x[c, in_off[c, j] : in_off[c, j] + m]
    np.testing.assert_array_equal(got, want)


@needs8
def test_a2a_blocks():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        dense, a2a = _layers(mesh)
        hs, tw, ti, _ = _inputs(mesh, 64, np.ones((64,), bool), seed=1)
        assert a2a._a2a_blocks(hs, ti) == (2, 4)
        assert dense._a2a_blocks(hs, ti) is None  # SGLANG_JAX_MOE_A2A off
        a2a.local_window_min_rows = 64 * TOPK + 1  # decode-sized: dense path
        assert a2a._a2a_blocks(hs, ti) is None
        a2a.local_window_min_rows = 1
        hs60, _, ti60, _ = _inputs(mesh, 60, np.ones((60,), bool), seed=1)  # 60 % 8 != 0
        assert a2a._a2a_blocks(hs60, ti60) is None
        rep = jax.device_put(hs, NamedSharding(mesh, P(None, None)))
        assert a2a._a2a_blocks(rep, ti) is None


@needs8
@pytest.mark.parametrize("emit_sumsq", [False, True])
@pytest.mark.parametrize(
    "T, per_group",
    [
        (64, None),  # every token real
        (64, (5, 0)),  # one busy DP group
        (64, (0, 0)),  # nothing real: zeros
        (128, (64, 17)),
        (256, None),
    ],
)
def test_a2a_matches_dense(emit_sumsq, T, per_group):
    mesh = _mesh()
    if per_group is None:
        mask_np = np.ones((T,), bool)
    else:
        mask_np = np.zeros((T,), bool)
        for g, n in enumerate(per_group):
            mask_np[g * (T // 2) : g * (T // 2) + n] = True
    with jax.set_mesh(mesh):
        dense, a2a = _layers(mesh, seed=T + 11, emit_sumsq=emit_sumsq)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=T + (per_group or (0,))[0])
        ref, ref_ss = _run(dense, hs, tw, ti, mask)
        out, out_ss = _run(a2a, hs, tw, ti, mask)
        direct = _direct_a2a(a2a, hs, tw, ti, mask)
    _assert_close(out, ref, mask_np)
    np.testing.assert_array_equal(out, direct)  # the cond took the a2a branch
    if emit_sumsq:
        # The sums come from the bf16-rounded rows of each path's own output.
        np.testing.assert_allclose(out_ss, np.sum(out * out, axis=1), rtol=2e-2, atol=1e-3)
        np.testing.assert_allclose(out_ss, ref_ss, rtol=5e-2, atol=1e-2 * ref_ss.max() + 1e-6)
        assert np.all(out_ss[~mask_np] == 0)


@needs8
def test_a2a_scattered_mask():
    T = 128
    mesh = _mesh()
    rng = np.random.default_rng(5)
    mask_np = np.zeros((T,), bool)
    mask_np[rng.choice(T, size=45, replace=False)] = True
    with jax.set_mesh(mesh):
        dense, a2a = _layers(mesh, seed=23)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=6)
        ref, _ = _run(dense, hs, tw, ti, mask)
        out, _ = _run(a2a, hs, tw, ti, mask)
    _assert_close(out, ref, mask_np)


@needs8
def test_a2a_skewed_routing():
    """One hot expert takes every token: its rank gets T rows, spanning several GMM tiles."""
    T = 128
    mesh = _mesh()
    mask_np = np.ones((T,), bool)
    with jax.set_mesh(mesh):
        dense, a2a = _layers(mesh, seed=31)
        a2a.a2a_capacity = T * TOPK  # holds any routing
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=9, hot=5)
        ref, _ = _run(dense, hs, tw, ti, mask)
        out, _ = _run(a2a, hs, tw, ti, mask)
        direct = _direct_a2a(a2a, hs, tw, ti, mask)
    _assert_close(out, ref, mask_np)
    np.testing.assert_array_equal(out, direct)


@needs8
def test_a2a_overflow_falls_back_to_dense():
    """A rank load above the receive buffer takes the dense path (same output)."""
    T = 128
    mesh = _mesh()
    mask_np = np.ones((T,), bool)
    with jax.set_mesh(mesh):
        dense, a2a = _layers(mesh, seed=37)
        a2a.a2a_capacity = 128  # the hot rank gets >= T rows plus others' picks: overflow
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=10, hot=2)
        ids = np.asarray(ti)
        loads = np.bincount(ids.ravel() // (E // EP), minlength=EP)
        assert loads.max() > 128
        ref, _ = _run(dense, hs, tw, ti, mask)
        out, _ = _run(a2a, hs, tw, ti, mask)
        direct = _direct_a2a(a2a, hs, tw, ti, mask)  # what an unchecked a2a would give
    np.testing.assert_array_equal(out, ref)  # the dense branch, bit for bit
    assert np.abs(direct - ref).max() > 1e-2 * np.abs(ref).max()  # truncated rows were lost
