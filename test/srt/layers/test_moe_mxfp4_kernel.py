"""MXFP4EPMoE: fused MXFP4 GMM kernel path vs the dequant + bf16 GMM path (CPU, interpret mode).

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_mxfp4_kernel.py -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

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


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 (CPU) devices")
@pytest.mark.parametrize("T", [5, 40])
def test_kernel_path_matches_dequant_path(T):
    H, I, E, topk, ep = 256, 256, 8, 2, 2  # expert-TP = 2 -> 128 intermediate rows per shard
    mesh = jax.make_mesh((1, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
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
        )
        ref_layer = MXFP4EPMoE(use_gmm_kernel=False, **common)
        ker_layer = MXFP4EPMoE(use_gmm_kernel=True, **common)
        assert ker_layer.gmm_kernel_supported()
        for name, (k, n), seed in (("w1", (H, I), 1), ("w3", (H, I), 2), ("w2", (I, H), 3)):
            packed, scale = F.random_mxfp4((k, n), E, seed=seed, zero_frac=0.1)
            for layer in (ref_layer, ker_layer):
                for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                    param = getattr(layer, f"{name}_{suffix}")
                    param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
        assert ker_layer.prepare_gmm_kernel_weights()
        assert ker_layer.w1_weight_packed.value.dtype == jnp.uint32
        assert ker_layer.w2_weight_packed.value.shape == (E, I // 8, H)
        assert not ref_layer.prepare_gmm_kernel_weights()

        ids, w = _routing(T, E, topk, seed=T)
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(T).standard_normal((T, H)), jnp.bfloat16), rep
        )
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        out_ref = np.asarray(ref_layer(hs, tw, ti).astype(jnp.float32))
        out_ker = np.asarray(ker_layer(hs, tw, ti).astype(jnp.float32))

    assert out_ref.shape == out_ker.shape == (T, H)
    scale = np.abs(out_ref).max()
    # Same math (bf16 GMM outputs, f32 SiTU, bf16 down-proj output); only f32 summation order
    # differs, which can flip a few bf16 roundings.
    np.testing.assert_allclose(out_ker, out_ref, rtol=0, atol=2e-2 * scale)
    assert np.abs(out_ker - out_ref).mean() < 2e-3 * np.abs(out_ref).mean()


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 (CPU) devices")
@pytest.mark.parametrize("T,cap", [(40, None), (40, 16), (64, 24), (7, 8)])
def test_local_window_path_matches_generic_kernel_path(T, cap):
    """Prefill local-window path (incl. multi-window overflow loop) == EPMoE's generic path."""
    H, I, E, topk, ep = 256, 256, 8, 3, 2
    mesh = jax.make_mesh((1, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
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
        )
        generic = MXFP4EPMoE(use_gmm_kernel=True, **common)
        windowed = MXFP4EPMoE(use_gmm_kernel=True, **common)
        generic.local_window_min_rows = 1 << 30  # never
        windowed.local_window_min_rows = 1  # always
        windowed.local_window_capacity = cap
        for name, (k, n), seed in (("w1", (H, I), 4), ("w3", (H, I), 5), ("w2", (I, H), 6)):
            packed, scale = F.random_mxfp4((k, n), E, seed=seed, zero_frac=0.1)
            for layer in (generic, windowed):
                for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                    param = getattr(layer, f"{name}_{suffix}")
                    param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
        assert generic.prepare_gmm_kernel_weights() and windowed.prepare_gmm_kernel_weights()

        ids, w = _routing(T, E, topk, seed=100 + T)
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(T).standard_normal((T, H)), jnp.bfloat16), rep
        )
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        out_gen = np.asarray(generic(hs, tw, ti).astype(jnp.float32))
        out_win = np.asarray(windowed(hs, tw, ti).astype(jnp.float32))

    assert out_gen.shape == out_win.shape == (T, H)
    scale = np.abs(out_gen).max()
    np.testing.assert_allclose(out_win, out_gen, rtol=0, atol=1e-2 * scale)
    assert np.abs(out_win - out_gen).mean() < 1e-3 * np.abs(out_gen).mean()


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 (CPU) devices")
def test_local_window_path_skips_masked_padding_tokens():
    """DP-padding tokens (token_mask=False) are skipped: zero output, real rows unchanged."""
    T, H, I, E, topk, ep = 48, 256, 256, 8, 3, 2
    mesh = jax.make_mesh((1, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)
    with jax.set_mesh(mesh):
        common = dict(hidden_size=H, num_experts=E, num_experts_per_tok=topk, ep_size=ep, mesh=mesh,
                      intermediate_dim=I, activation="situ", situ_beta=4.0, situ_linear_beta=25.0)
        generic = MXFP4EPMoE(use_gmm_kernel=True, **common)
        windowed = MXFP4EPMoE(use_gmm_kernel=True, **common)
        generic.local_window_min_rows = 1 << 30
        windowed.local_window_min_rows = 1
        windowed.local_window_capacity = 16
        for name, (k, n), seed in (("w1", (H, I), 7), ("w3", (H, I), 8), ("w2", (I, H), 9)):
            packed, scale = F.random_mxfp4((k, n), E, seed=seed, zero_frac=0.1)
            for layer in (generic, windowed):
                for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                    param = getattr(layer, f"{name}_{suffix}")
                    param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
        assert generic.prepare_gmm_kernel_weights() and windowed.prepare_gmm_kernel_weights()

        ids, w = _routing(T, E, topk, seed=7)
        mask = np.zeros(T, bool)
        mask[:12] = True  # first DP rank's 12 real tokens; the rest is padding
        mask[30:33] = True
        rep = NamedSharding(mesh, P(None, None))
        hs = jax.device_put(jnp.asarray(np.random.default_rng(7).standard_normal((T, H)), jnp.bfloat16), rep)
        tw = jax.device_put(jnp.asarray(w), rep)
        ti = jax.device_put(jnp.asarray(ids), rep)
        tm = jax.device_put(jnp.asarray(mask), NamedSharding(mesh, P(None)))
        out_gen = np.asarray(generic(hs, tw, ti).astype(jnp.float32))
        out_win = np.asarray(windowed(hs, tw, ti, token_mask=tm).astype(jnp.float32))

    assert np.all(out_win[~mask] == 0)
    scale = np.abs(out_gen[mask]).max()
    np.testing.assert_allclose(out_win[mask], out_gen[mask], rtol=0, atol=1e-2 * scale)
