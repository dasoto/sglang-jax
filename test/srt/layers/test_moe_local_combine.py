"""MXFP4EPMoE local-window combine variants (SGLANG_JAX_MOE_LOCAL_COMBINE).

The local-window path scatter-adds each window's weighted expert rows into the per-chip
[T, H] accumulator. "onehot" does it as a [T, cap] one-hot matmul with f32 accumulation (for
T up to SGLANG_JAX_MOE_ONEHOT_MAX_TOKENS), "sorted" sorts the rows by token first and drops
the masked rows. Results must match the default scatter-add (the f32 sums differ only in
rounding order), for the full and compact paths and with several windows.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_local_combine.py -q
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

H, I, E, TOPK, EP = 256, 256, 8, 3, 4  # 8 devices: data 2 x tensor 4; MoE expert 4 x tensor 2


def _mesh():
    return jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _layers(mesh, mode, compact, capacity, bf16_acc, seed=13):
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
    ref = MXFP4EPMoE(use_gmm_kernel=True, **common)
    new = MXFP4EPMoE(use_gmm_kernel=True, **common)
    for layer in (ref, new):
        layer.rs_combine = layer.rs_combine_mesh = True
        layer.local_window_min_rows = 1
        layer.local_window_capacity = capacity
        layer.compact_divisors = (8, 4, 2) if compact else ()
        layer.compact_min_tokens = 1
        if bf16_acc:
            layer.local_acc_dtype = jnp.bfloat16
    ref.local_combine = "scatter"
    new.local_combine = mode
    new.onehot_max_tokens = 1 << 20
    for name, (k, n), s in (("w1", (H, I), seed), ("w3", (H, I), seed + 1), ("w2", (I, H), seed + 2)):
        packed, scale = F.random_mxfp4((k, n), E, seed=s, zero_frac=0.1)
        for layer in (ref, new):
            for suffix, arr in (("weight_packed", packed), ("weight_scale", scale)):
                param = getattr(layer, f"{name}_{suffix}")
                param.value = jax.device_put(jnp.asarray(arr), param.value.sharding)
    assert ref.prepare_gmm_kernel_weights() and new.prepare_gmm_kernel_weights()
    return ref, new


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
    mask = np.zeros((T,), bool)
    tl = T // len(per_group)
    for g, n in enumerate(per_group):
        mask[g * tl : g * tl + n] = True
    return mask


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("mode", ["onehot", "sorted"])
@pytest.mark.parametrize(
    "per_group,compact,capacity,bf16_acc",
    [
        ((32, 32), False, 16, False),  # full path, several windows, f32 accumulator
        ((32, 32), False, 64, True),  # full path, one window, bf16 accumulator
        ((5, 0), True, 16, True),  # compact, one busy group
        ((9, 14), True, 8, False),  # compact, both groups, many windows
    ],
)
def test_local_combine_matches_scatter(mode, per_group, compact, capacity, bf16_acc):
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, per_group)
    with jax.set_mesh(mesh):
        ref, new = _layers(mesh, mode, compact, capacity, bf16_acc, seed=sum(per_group) + 7)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=sum(per_group) + 11)
        out_ref = jax.jit(lambda a, b, c, m: ref(a, b, c, token_mask=m))(hs, tw, ti, mask)
        out_new = jax.jit(lambda a, b, c, m: new(a, b, c, token_mask=m))(hs, tw, ti, mask)
        assert out_new.sharding.spec == P("data", "tensor")
        out_ref = np.asarray(out_ref.astype(jnp.float32))
        out_new = np.asarray(out_new.astype(jnp.float32))
    assert np.all(out_new[~mask_np] == 0) and np.all(out_ref[~mask_np] == 0)
    scale = np.abs(out_ref).max()
    np.testing.assert_allclose(out_new, out_ref, rtol=0, atol=1e-2 * scale)
    assert np.abs(out_new - out_ref).mean() < 3e-3 * np.abs(out_ref[mask_np]).mean()


def test_mode_parsing(monkeypatch):
    from sgl_jax.srt.layers.moe_mxfp4 import _local_combine_mode

    monkeypatch.setenv("SGLANG_JAX_MOE_LOCAL_COMBINE", "OneHot")
    assert _local_combine_mode() == "onehot"
    monkeypatch.setenv("SGLANG_JAX_MOE_LOCAL_COMBINE", "bogus")
    assert _local_combine_mode() == "scatter"
    monkeypatch.delenv("SGLANG_JAX_MOE_LOCAL_COMBINE")
    assert _local_combine_mode() == "scatter"
