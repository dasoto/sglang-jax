"""EPMoE dispatch variants (local window, all-to-all) match the dense dispatch.

The window gathers only the M * min(top_k, experts_per_device) expert-sorted rows
that start at the shard's first local row; outputs must be identical to gathering
all M * top_k rows. CPU, 4 expert shards, the GEMM runs in interpret mode.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers import moe
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _route(rng, m, num_experts, top_k, skew):
    """Distinct experts per token; ``skew`` biases routing toward expert 0 and the last."""
    logits = rng.standard_normal((m, num_experts))
    logits[:, 0] += skew
    logits[:, -1] += skew
    ids = np.argsort(-logits, axis=1)[:, :top_k].astype(np.int32)
    w = rng.random((m, top_k)).astype(np.float32)
    return ids, w / w.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "num_experts, top_k, m, skew",
    [(8, 4, 37, 0.0), (8, 4, 64, 3.0), (4, 2, 50, 0.0), (4, 2, 16, 5.0)],
)
def test_local_window_matches_full_dispatch(monkeypatch, num_experts, top_k, m, skew):
    mesh = create_device_mesh(
        ici_parallelism=[1, 4], dcn_parallelism=[1, 1], devices=jax.devices()[:4]
    )
    with jax.set_mesh(mesh):
        layer = moe.EPMoE(
            hidden_size=32,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            ep_size=4,
            mesh=mesh,
            intermediate_dim=16,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
        )
    rng = np.random.default_rng(num_experts * 1000 + m)
    x = jnp.asarray(rng.standard_normal((m, 32)) * 0.1, jnp.float32)
    ids, w = _route(rng, m, num_experts, top_k, skew)
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(moe, "MOE_LOCAL_WINDOW", flag)
        with jax.set_mesh(mesh):
            outs.append(np.asarray(layer(x, jnp.asarray(w), jnp.asarray(ids))))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-5, atol=1e-5)
    assert np.abs(outs[0]).max() > 0


@pytest.mark.parametrize(
    "top_k, m, skew",
    [(2, 64, 0.0), (4, 256, 0.0), (4, 128, 6.0), (2, 8, 0.0), (2, 60, 0.0)],
)
def test_a2a_matches_dense_dispatch(monkeypatch, top_k, m, skew):
    """All-to-all exchange (SGLANG_JAX_MOE_A2A) vs the dense exchange: 8 shards with one
    expert each; int8 dispatch, so compare against the output scale. skew=6 routes most
    tokens to two experts (large capacity bucket); m=60 is not a multiple of the shard
    count and falls back to the dense path."""
    mesh = create_device_mesh(
        ici_parallelism=[1, 8], dcn_parallelism=[1, 1], devices=jax.devices()[:8]
    )
    with jax.set_mesh(mesh):
        layer = moe.EPMoE(
            hidden_size=32,
            num_experts=8,
            num_experts_per_tok=top_k,
            ep_size=8,
            mesh=mesh,
            intermediate_dim=16,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
        )
    rng = np.random.default_rng(top_k * 100 + m)
    x = jnp.asarray(rng.standard_normal((m, 32)) * 0.1, jnp.float32)
    ids, w = _route(rng, m, 8, top_k, skew)
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(moe, "MOE_A2A", flag)
        with jax.set_mesh(mesh):
            outs.append(np.asarray(layer(x, jnp.asarray(w), jnp.asarray(ids))))
    scale = np.abs(outs[0]).max()
    assert scale > 0
    np.testing.assert_allclose(outs[1], outs[0], rtol=0, atol=2e-2 * scale)
