"""MXFP4EPMoE compact prefill path (SGLANG_JAX_MOE_COMPACT) vs the full mesh-mode path.

With few busy DP groups most of a prefill step's tokens are padding. The compact path packs
the real tokens into a static number of rows before the token gather, runs the experts on
them and scatters the combined rows back, so it must reproduce the full path's output for
every real token and leave padding rows zero. A switch on the real-token count picks the
smallest compact size that holds them, or the full path.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_compact.py -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.mxfp4_gmm import format as F
from sgl_jax.srt.layers.moe_mxfp4 import MXFP4EPMoE, _parse_compact_divisors

H, I, E, TOPK, EP = 256, 256, 8, 3, 4  # 8 devices: data 2 x tensor 4; MoE expert 4 x tensor 2


def _mesh():
    return jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _layers(mesh, seed=7):
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
    full = MXFP4EPMoE(use_gmm_kernel=True, **common)
    compact = MXFP4EPMoE(use_gmm_kernel=True, **common)
    for layer in (full, compact):
        layer.rs_combine = layer.rs_combine_mesh = True
        layer.local_window_min_rows = 1
        layer.local_window_capacity = 16  # several windows: exercises the window loop
    full.compact_divisors = ()
    compact.compact_divisors = (8, 4, 2)
    compact.compact_min_tokens = 1
    for name, (k, n), s in (("w1", (H, I), seed), ("w3", (H, I), seed + 1), ("w2", (I, H), seed + 2)):
        packed, scale = F.random_mxfp4((k, n), E, seed=s, zero_frac=0.1)
        for layer in (full, compact):
            for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                param = getattr(layer, f"{name}_{suffix}")
                param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
    assert full.prepare_gmm_kernel_weights() and compact.prepare_gmm_kernel_weights()
    return full, compact


def _inputs(mesh, T, mask_np, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, E))
    ids = np.argsort(-logits, axis=1)[:, :TOPK].astype(np.int32)
    w = np.take_along_axis(logits, ids, 1)
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


def _prefix_mask(T, per_group):
    """DP-group-major layout: group g's real tokens first in its T/dp slots."""
    mask = np.zeros((T,), bool)
    tl = T // len(per_group)
    for g, n in enumerate(per_group):
        mask[g * tl : g * tl + n] = True
    return mask


def test_parse_compact_divisors():
    assert _parse_compact_divisors("") == ()
    assert _parse_compact_divisors("0") == ()
    assert _parse_compact_divisors("8,4,2") == (2, 4, 8)
    assert _parse_compact_divisors("4, 1, 4") == (4,)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
def test_compact_sizes():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        full, compact = _layers(mesh)
        T = 64
        hs, tw, ti, _ = _inputs(mesh, T, np.ones((T,), bool), seed=1)
        blocks = compact._rs_combine_blocks(hs)
        assert blocks == (2, 4)
        assert compact._compact_sizes(hs, ti, blocks) == (8, 16, 32)
        assert full._compact_sizes(hs, ti, blocks) == ()
        compact.compact_min_tokens = 16
        assert compact._compact_sizes(hs, ti, blocks) == (16, 32)
        # Needs the P("data", "tensor") input of mesh mode.
        rep = jax.device_put(hs, NamedSharding(mesh, P(None, None)))
        assert compact._compact_sizes(rep, ti, blocks) == ()
        compact.rs_combine_mesh = False
        assert compact._compact_sizes(hs, ti, blocks) == ()


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize(
    "per_group",
    [
        (5, 0),  # one busy group: smallest compact size (8)
        (0, 13),  # the other group: 16
        (9, 14),  # both groups: 32 (T / 2)
        (32, 7),  # more than T / 2 real tokens: the full path
        (0, 0),  # nothing real: zeros
    ],
)
def test_compact_matches_full(per_group):
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, per_group)
    with jax.set_mesh(mesh):
        full, compact = _layers(mesh, seed=17)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=sum(per_group) + 3)
        f_full = jax.jit(lambda a, b, c, m: full(a, b, c, token_mask=m))
        f_compact = jax.jit(lambda a, b, c, m: compact(a, b, c, token_mask=m))
        out_full = f_full(hs, tw, ti, mask)
        out_compact = f_compact(hs, tw, ti, mask)
        assert out_compact.sharding.spec == P("data", "tensor")
        out_full = np.asarray(out_full.astype(jnp.float32))
        out_compact = np.asarray(out_compact.astype(jnp.float32))
    assert np.all(out_compact[~mask_np] == 0) and np.all(out_full[~mask_np] == 0)
    if not mask_np.any():
        return
    scale = np.abs(out_full).max()
    # Same per-chip partial sums; the bf16 cross-chip sums may round in another order.
    np.testing.assert_allclose(out_compact, out_full, rtol=0, atol=1e-2 * scale)
    assert np.abs(out_compact - out_full).mean() < 3e-3 * np.abs(out_full[mask_np]).mean()


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("compact", [False, True])
def test_bf16_accumulator_matches_f32(compact):
    """SGLANG_JAX_MOE_BF16_ACC: the local-window accumulator in bf16 instead of f32."""
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, (20, 9))
    with jax.set_mesh(mesh):
        f32_layer, bf16_layer = _layers(mesh, seed=37)
        for layer in (f32_layer, bf16_layer):
            layer.compact_divisors = (2,) if compact else ()
        bf16_layer.local_acc_dtype = jnp.bfloat16
        assert f32_layer.local_acc_dtype == jnp.float32
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=41)
        out_f32 = jax.jit(lambda a, b, c, m: f32_layer(a, b, c, token_mask=m))(hs, tw, ti, mask)
        out_bf16 = jax.jit(lambda a, b, c, m: bf16_layer(a, b, c, token_mask=m))(hs, tw, ti, mask)
        out_f32 = np.asarray(out_f32.astype(jnp.float32))
        out_bf16 = np.asarray(out_bf16.astype(jnp.float32))
    assert np.all(out_bf16[~mask_np] == 0)
    scale = np.abs(out_f32).max()
    np.testing.assert_allclose(out_bf16, out_f32, rtol=0, atol=1e-2 * scale)
    assert np.abs(out_bf16 - out_f32).mean() < 3e-3 * np.abs(out_f32[mask_np]).mean()


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("c", [16, 32])
def test_compact_forward_scattered_mask(c):
    """Real tokens anywhere in a group's slots (not only a prefix), direct compact call."""
    T = 64
    mesh = _mesh()
    rng = np.random.default_rng(c)
    mask_np = np.zeros((T,), bool)
    mask_np[rng.choice(T, size=c - 3, replace=False)] = True
    with jax.set_mesh(mesh):
        full, compact = _layers(mesh, seed=29)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=c)
        out_full = jax.jit(lambda a, b, d, m: full(a, b, d, token_mask=m))(hs, tw, ti, mask)
        blocks = compact._rs_combine_blocks(hs)
        out_compact = jax.jit(
            lambda a, b, d, m: compact._compact_forward(a, b, d, m, rs_blocks=blocks, c=c)
        )(hs, tw, ti, mask)
        out_full = np.asarray(out_full.astype(jnp.float32))
        out_compact = np.asarray(out_compact.astype(jnp.float32))
    assert np.all(out_compact[~mask_np] == 0)
    scale = np.abs(out_full).max()
    np.testing.assert_allclose(out_compact, out_full, rtol=0, atol=1e-2 * scale)
