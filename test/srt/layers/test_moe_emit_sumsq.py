"""MXFP4EPMoE.emit_sumsq: the combine also returns each output row's sum of squares.

Kimi-K3's routed_expert_norm (SGLANG_JAX_MOE_NORM_FROM_COMBINE=1) uses these instead of
all-reducing them over "tensor". Outputs must equal the plain path; the sums must equal the
f32 sums of squares of the output rows, for the full (generic and local-window) and compact
mesh-mode paths, with DP padding.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_moe_emit_sumsq.py -q
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
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.moe_mxfp4 import MXFP4EPMoE

H, I, E, TOPK, EP = 256, 256, 8, 3, 4  # 8 devices: data 2 x tensor 4; MoE expert 4 x tensor 2


def _mesh():
    return jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _layers(mesh, compact, window, seed=3):
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
        layer.local_window_min_rows = 1 if window else 1 << 30
        layer.local_window_capacity = 16
        layer.compact_divisors = (8, 4, 2) if compact else ()
        layer.compact_min_tokens = 1
    new.emit_sumsq = True
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
@pytest.mark.parametrize(
    "per_group,compact,window,use_mask",
    [
        ((32, 32), False, False, False),  # decode-like generic path, no mask
        ((32, 32), False, True, True),  # full local-window path
        ((5, 0), True, True, True),  # compact, one busy group
        ((9, 14), True, True, True),  # compact, both groups
        ((32, 7), True, True, True),  # falls back to the full path inside the switch
    ],
)
def test_emit_sumsq(per_group, compact, window, use_mask):
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, per_group)
    with jax.set_mesh(mesh):
        ref, new = _layers(mesh, compact, window, seed=sum(per_group) + 2)
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=sum(per_group) + 5)
        if use_mask:
            f_ref = jax.jit(lambda a, b, c, m: ref(a, b, c, token_mask=m))
            f_new = jax.jit(lambda a, b, c, m: new(a, b, c, token_mask=m))
            out_ref = f_ref(hs, tw, ti, mask)
            out_new, ss = f_new(hs, tw, ti, mask)
        else:
            out_ref = jax.jit(lambda a, b, c: ref(a, b, c))(hs, tw, ti)
            out_new, ss = jax.jit(lambda a, b, c: new(a, b, c))(hs, tw, ti)
        assert out_new.sharding.spec == P("data", "tensor")
        assert ss.shape == (T,) and ss.dtype == jnp.float32
        out_ref = np.asarray(out_ref.astype(jnp.float32))
        out_new = np.asarray(out_new.astype(jnp.float32))
        ss = np.asarray(ss)
    real = mask_np if use_mask else np.ones((T,), bool)
    # The combine is the same all-reduce, so the outputs match exactly.
    np.testing.assert_array_equal(out_new, out_ref)
    expect = np.sum(out_new.astype(np.float64) ** 2, axis=1)
    np.testing.assert_allclose(ss[real], expect[real], rtol=1e-5, atol=1e-6)
    assert np.all(ss[~real] == 0) or not use_mask


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize(
    "per_group,compact,window",
    [
        ((32, 32), False, True),  # full local-window path
        ((32, 32), False, False),  # generic path
        ((5, 0), True, True),  # compact, one busy group
        ((9, 14), True, True),  # compact, both groups
        ((32, 7), True, True),  # falls back to the full path inside the switch
    ],
)
def test_emit_sumsq_hier_combine(per_group, compact, window):
    """SGLANG_JAX_MOE_HIER_COMBINE: same outputs and sums as the one-step all-reduce combine.

    The two reduction stages round the bf16 partial sums in a different order, so the outputs
    agree to bf16 rounding; the sums of squares must match the returned output rows.
    """
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, per_group)
    with jax.set_mesh(mesh):
        ref, new = _layers(mesh, compact, window, seed=sum(per_group) + 7)
        ref.emit_sumsq = True
        new.hier_combine = True
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=sum(per_group) + 9)
        f_ref = jax.jit(lambda a, b, c, m: ref(a, b, c, token_mask=m))
        f_new = jax.jit(lambda a, b, c, m: new(a, b, c, token_mask=m))
        out_ref, ss_ref = f_ref(hs, tw, ti, mask)
        out_new, ss_new = f_new(hs, tw, ti, mask)
        assert out_new.sharding.spec == P("data", "tensor")
        out_ref = np.asarray(out_ref.astype(jnp.float32))
        out_new = np.asarray(out_new.astype(jnp.float32))
        ss_ref, ss_new = np.asarray(ss_ref), np.asarray(ss_new)
    np.testing.assert_allclose(out_new, out_ref, rtol=2e-2, atol=2e-2 * np.abs(out_ref).max())
    expect = np.sum(out_new.astype(np.float64) ** 2, axis=1)
    np.testing.assert_allclose(ss_new[mask_np], expect[mask_np], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ss_new[mask_np], ss_ref[mask_np], rtol=2e-2)
    assert np.all(ss_new[~mask_np] == 0)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("window", [True, False])
def test_fp8_dispatch_close_to_bf16(window):
    """SGLANG_JAX_MOE_FP8_DISPATCH: the two-step token gather moves e4m3 rows with an f32 scale
    per row of each chip's H slice; outputs within e4m3 rounding of the bf16 gather."""
    T = 64
    mesh = _mesh()
    mask_np = _prefix_mask(T, (32, 32))
    with jax.set_mesh(mesh):
        ref, new = _layers(mesh, compact=False, window=window, seed=21)
        for layer in (ref, new):
            layer.emit_sumsq = True
            layer.mesh_gather_two_step_bytes = 0  # the large-input (two-step) gather
        new.fp8_dispatch = True
        hs, tw, ti, mask = _inputs(mesh, T, mask_np, seed=23)
        f_ref = jax.jit(lambda a, b, c, m: ref(a, b, c, token_mask=m))
        f_new = jax.jit(lambda a, b, c, m: new(a, b, c, token_mask=m))
        out_ref = np.asarray(f_ref(hs, tw, ti, mask)[0].astype(jnp.float32))
        out_new = np.asarray(f_new(hs, tw, ti, mask)[0].astype(jnp.float32))
    scale = np.abs(out_ref).max()
    np.testing.assert_allclose(out_new, out_ref, rtol=0, atol=0.1 * scale)
    rel = np.abs(out_new - out_ref).mean() / np.abs(out_ref).mean()
    assert 0 < rel < 0.05, rel  # changed (quantized), but by e4m3 rounding only


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
def test_norm_from_sumsq_matches_rmsnorm():
    from sgl_jax.srt.models.kimi_k3 import _rms_norm_from_sumsq

    mesh = _mesh()
    with jax.set_mesh(mesh):
        rng = np.random.default_rng(0)
        x = jax.device_put(
            jnp.asarray(rng.standard_normal((16, H)) * 3, jnp.bfloat16),
            NamedSharding(mesh, P("data", "tensor")),
        )
        norm = RMSNorm(H, epsilon=1e-5, param_dtype=jnp.bfloat16)
        norm.scale.value = jnp.asarray(rng.standard_normal((H,)), jnp.bfloat16)
        x32 = x.astype(jnp.float32)
        ss = jnp.sum(x32 * x32, axis=-1)
        a = np.asarray(norm(x).astype(jnp.float32))
        b = np.asarray(_rms_norm_from_sumsq(x, ss, norm).astype(jnp.float32))
    assert b.dtype == a.dtype
    np.testing.assert_allclose(b, a, rtol=1e-2, atol=1e-2)
    assert np.mean(np.abs(b - a)) < 2e-3 * np.mean(np.abs(a)) + 1e-6
