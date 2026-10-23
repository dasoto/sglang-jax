"""MXFP4EPMoE reduce-scatter combine (SGLANG_JAX_MOE_RS_COMBINE) vs the all-reduce combine.

The RS combine lands each chip's block of the model mesh's P("data", "tensor") layout with one
reduce-scatter over the whole (expert, tensor) MoE mesh instead of all-reducing [T, H] over
"tensor" and "expert". Values must match the all-reduce path; only the output sharding changes.
Mesh mode (SGLANG_JAX_MOE_RS_COMBINE=2) runs that reduce-scatter and the token all-gather over
the model mesh's axes instead; it is checked the same way.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_rs_combine.py -q
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
from sgl_jax.srt.layers.moe_mxfp4 import MXFP4EPMoE


def _routing(T, num_experts, topk, seed):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, num_experts))
    ids = np.argsort(-logits, axis=1)[:, :topk].astype(np.int32)
    w = np.take_along_axis(logits, ids, 1)
    w = np.exp(w) / np.exp(w).sum(1, keepdims=True)
    return ids, w.astype(np.float32)


def _layers(mesh, H, I, E, topk, ep, seed, mesh_mode=False):
    common = dict(
        hidden_size=H,
        num_experts=E,
        num_experts_per_tok=topk,
        ep_size=ep,
        mesh=mesh,
        intermediate_dim=I,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    ar = MXFP4EPMoE(use_gmm_kernel=True, **common)
    rs = MXFP4EPMoE(use_gmm_kernel=True, **common)
    ar.rs_combine, rs.rs_combine = False, True
    ar.rs_combine_mesh, rs.rs_combine_mesh = False, mesh_mode
    for name, (k, n), s in (
        ("w1", (H, I), seed),
        ("w3", (H, I), seed + 1),
        ("w2", (I, H), seed + 2),
    ):
        packed, scale = F.random_mxfp4((k, n), E, seed=s, zero_frac=0.1)
        for layer in (ar, rs):
            for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                param = getattr(layer, f"{name}_{suffix}")
                param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
    assert ar.prepare_gmm_kernel_weights() and rs.prepare_gmm_kernel_weights()
    return ar, rs


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("T,window", [(8, False), (40, False), (40, True), (64, True)])
@pytest.mark.parametrize("in_spec", [P(None, None), P("data", "tensor")])
@pytest.mark.parametrize("mesh_mode", [False, True])
def test_rs_combine_matches_allreduce(T, window, in_spec, mesh_mode):
    """Generic (decode) and local-window (prefill) paths, replicated or sharded input."""
    H, I, E, topk, ep = 256, 256, 8, 3, 4  # 8 devices: data 2 x tensor 4; MoE expert 4 x tensor 2
    mesh = jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    with jax.set_mesh(mesh):
        ar, rs = _layers(mesh, H, I, E, topk, ep, seed=11, mesh_mode=mesh_mode)
        for layer in (ar, rs):
            layer.local_window_min_rows = 1 if window else 1 << 30
        ids, w = _routing(T, E, topk, seed=T)
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(T).standard_normal((T, H)), jnp.bfloat16),
            NamedSharding(mesh, in_spec),
        )
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        assert rs._rs_combine_blocks(hs) == (2, 4) and ar._rs_combine_blocks(hs) is None
        out_ar = ar(hs, tw, ti)
        out_rs = rs(hs, tw, ti)
        assert out_rs.sharding.spec == P("data", "tensor")
        out_ar = np.asarray(out_ar.astype(jnp.float32))
        out_rs = np.asarray(out_rs.astype(jnp.float32))

    assert out_ar.shape == out_rs.shape == (T, H)
    scale = np.abs(out_ar).max()
    # Same partial sums; the bf16 cross-shard sums are rounded in a different order (two
    # all-reduces vs one reduce-scatter), so elements differ by about one bf16 ulp.
    np.testing.assert_allclose(out_rs, out_ar, rtol=0, atol=1e-2 * scale)
    assert np.abs(out_rs - out_ar).mean() < 3e-3 * np.abs(out_ar).mean()


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("T", [8, 24])
@pytest.mark.parametrize("mesh_mode", [False, True])
def test_rs_combine_exact_in_f32(T, mesh_mode):
    """f32 activations (XLA dequant + GMM path): the two combines agree to f32 rounding."""
    H, I, E, topk, ep = 256, 256, 8, 3, 4
    mesh = jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    with jax.set_mesh(mesh):
        common = dict(
            hidden_size=H,
            num_experts=E,
            num_experts_per_tok=topk,
            ep_size=ep,
            mesh=mesh,
            intermediate_dim=I,
            activation="situ",
            situ_beta=4.0,
            situ_linear_beta=25.0,
            dtype=jnp.float32,
        )
        ar = MXFP4EPMoE(use_gmm_kernel=False, **common)
        rs = MXFP4EPMoE(use_gmm_kernel=False, **common)
        ar.rs_combine, rs.rs_combine = False, True
        ar.rs_combine_mesh, rs.rs_combine_mesh = False, mesh_mode
        for name, (k, n), s in (("w1", (H, I), 41), ("w3", (H, I), 42), ("w2", (I, H), 43)):
            packed, scale = F.random_mxfp4((k, n), E, seed=s, zero_frac=0.1)
            for layer in (ar, rs):
                for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                    param = getattr(layer, f"{name}_{suffix}")
                    param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
        ids, w = _routing(T, E, topk, seed=7 + T)
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(T).standard_normal((T, H)), jnp.float32),
            NamedSharding(mesh, P("data", "tensor")),
        )
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        out_ar = np.asarray(ar(hs, tw, ti))
        out_rs = np.asarray(rs(hs, tw, ti))
    np.testing.assert_allclose(out_rs, out_ar, rtol=1e-5, atol=1e-5 * np.abs(out_ar).max())


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("mesh_mode", [False, True])
def test_rs_combine_with_masked_padding_and_jit(mesh_mode):
    """Under jit, with DP-padding tokens masked out (prefill path): zero rows stay zero."""
    T, H, I, E, topk, ep = 48, 256, 256, 8, 3, 4
    mesh = jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    with jax.set_mesh(mesh):
        ar, rs = _layers(mesh, H, I, E, topk, ep, seed=21, mesh_mode=mesh_mode)
        for layer in (ar, rs):
            layer.local_window_min_rows = 1
            layer.local_window_capacity = 16
        ids, w = _routing(T, E, topk, seed=5)
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(5).standard_normal((T, H)), jnp.bfloat16),
            NamedSharding(mesh, P("data", "tensor")),
        )
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        mask_np = np.ones((T,), bool)
        mask_np[[3, 4, 30, 47]] = False
        mask = jax.device_put(jnp.asarray(mask_np), NamedSharding(mesh, P(None)))
        f_ar = jax.jit(lambda a, b, c, m: ar(a, b, c, token_mask=m))
        f_rs = jax.jit(lambda a, b, c, m: rs(a, b, c, token_mask=m))
        out_ar = np.asarray(f_ar(hs, tw, ti, mask).astype(jnp.float32))
        out_rs = np.asarray(f_rs(hs, tw, ti, mask).astype(jnp.float32))

    assert np.all(out_rs[~mask_np] == 0) and np.all(out_ar[~mask_np] == 0)
    scale = np.abs(out_ar).max()
    np.testing.assert_allclose(out_rs, out_ar, rtol=0, atol=1e-2 * scale)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
def test_rs_combine_falls_back_when_tokens_not_divisible():
    """T not divisible by the data axis keeps the all-reduce combine (replicated output)."""
    H, I, E, topk, ep = 256, 256, 8, 2, 4
    mesh = jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    with jax.set_mesh(mesh):
        _, rs = _layers(mesh, H, I, E, topk, ep, seed=31)
        hs = jax.device_put(jnp.zeros((5, H), jnp.bfloat16), NamedSharding(mesh, P(None, None)))
        assert rs._rs_combine_blocks(hs) is None
