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


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("window", [False, True])
def test_gather_rs_matches_dense_dispatch(monkeypatch, int8, window):
    """Reduce-scatter combine (SGLANG_JAX_MOE_GATHER_RS) is exact; the int8 token gather
    (SGLANG_JAX_MOE_GATHER_INT8) is within the activation-quantization error."""
    mesh = create_device_mesh(
        ici_parallelism=[1, 8], dcn_parallelism=[1, 1], devices=jax.devices()[:8]
    )
    with jax.set_mesh(mesh):
        layer = moe.EPMoE(
            hidden_size=32,
            num_experts=8,
            num_experts_per_tok=4,
            ep_size=8,
            mesh=mesh,
            intermediate_dim=16,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
        )
    rng = np.random.default_rng(7 + int8)
    x = jnp.asarray(rng.standard_normal((64, 32)) * 0.1, jnp.float32)
    ids, w = _route(rng, 64, 8, 4, 2.0)
    monkeypatch.setattr(moe, "MOE_LOCAL_WINDOW", window)
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(moe, "MOE_GATHER_RS", flag)
        monkeypatch.setattr(moe, "MOE_GATHER_INT8", flag and int8)
        with jax.set_mesh(mesh):
            outs.append(np.asarray(layer(x, jnp.asarray(w), jnp.asarray(ids))))
    scale = np.abs(outs[0]).max()
    tol = 2e-2 * scale if int8 else 1e-5
    np.testing.assert_allclose(outs[1], outs[0], rtol=0, atol=tol)


def test_gather_rs_with_dp_sharded_output(monkeypatch):
    """The model passes its DP token layout as out_sharding so the reduce-scatter
    combine is followed by a gather inside each DP group, not a global all-reduce."""
    mesh = create_device_mesh(
        ici_parallelism=[2, 4], dcn_parallelism=[1, 1], devices=jax.devices()[:8]
    )
    with jax.set_mesh(mesh):
        layer = moe.EPMoE(
            hidden_size=32,
            num_experts=8,
            num_experts_per_tok=4,
            ep_size=8,
            mesh=mesh,
            intermediate_dim=16,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
        )
    rng = np.random.default_rng(11)
    ids, w = _route(rng, 64, 8, 4, 1.0)
    dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    with jax.set_mesh(mesh):
        x = jax.device_put(jnp.asarray(rng.standard_normal((64, 32)) * 0.1, jnp.float32), dp)
        wd = jax.device_put(jnp.asarray(w), dp)
        idd = jax.device_put(jnp.asarray(ids), dp)
    outs = []
    for flag in (False, True):
        monkeypatch.setattr(moe, "MOE_GATHER_RS", flag)
        with jax.set_mesh(mesh):
            y = layer(x, wd, idd, out_sharding=jax.typeof(x).sharding)
        assert y.sharding.spec[0] == "data"
        outs.append(np.asarray(y))
    np.testing.assert_allclose(outs[1], outs[0], rtol=0, atol=1e-5)


@pytest.mark.parametrize("m", [64, 60])
@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("dp", [2, 4])
def test_replicas_match_single_expert_group(monkeypatch, m, window, int8, dp):
    """SGLANG_JAX_MOE_REPLICAS=2 (EP4 x 2, each replica holding all 8 experts and
    exchanging only its own half of the tokens) matches EP8 on the DP token layout.
    m=60 is not a multiple of the device count: every replica computes all tokens.
    dp=4 splits the data axis between replica and expert, as on the TPU slice."""
    mesh = create_device_mesh(
        ici_parallelism=[dp, 8 // dp], dcn_parallelism=[1, 1], devices=jax.devices()[:8]
    )
    monkeypatch.setattr(moe, "MOE_LOCAL_WINDOW", window)
    monkeypatch.setattr(moe, "MOE_GATHER_RS", True)
    monkeypatch.setattr(moe, "MOE_GATHER_INT8", int8)
    rng = np.random.default_rng(17 + m)
    ids, w = _route(rng, m, 8, 4, 1.5)
    x_host = rng.standard_normal((m, 32)) * 0.1
    dp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    outs = []
    for replicas, ep in ((1, 8), (2, 4)):
        monkeypatch.setattr(moe, "MOE_REPLICAS", replicas)
        with jax.set_mesh(mesh):
            layer = moe.EPMoE(
                hidden_size=32,
                num_experts=8,
                num_experts_per_tok=4,
                ep_size=ep,
                mesh=mesh,
                intermediate_dim=16,
                weight_dtype=jnp.float32,
                dtype=jnp.float32,
            )
            assert layer.experts_per_device == 8 // ep
            # The weight loader places experts on epmoe_mesh(); it must be the layer's mesh.
            loader_mesh = moe.epmoe_mesh(mesh, ep)
            assert loader_mesh.axis_names == layer.moe_mesh.axis_names
            assert (loader_mesh.devices == layer.moe_mesh.devices).all()
            if m % 8 == 0:
                x = jax.device_put(jnp.asarray(x_host, jnp.float32), dp)
                wd = jax.device_put(jnp.asarray(w), dp)
                idd = jax.device_put(jnp.asarray(ids), dp)
                y = layer(x, wd, idd, out_sharding=jax.typeof(x).sharding)
                assert y.sharding.spec[0] == "data"
            else:
                y = layer(jnp.asarray(x_host, jnp.float32), jnp.asarray(w), jnp.asarray(ids))
        outs.append(np.asarray(y))
    scale = np.abs(outs[0]).max()
    assert scale > 0
    tol = 2e-2 * scale if int8 else 1e-5
    np.testing.assert_allclose(outs[1], outs[0], rtol=0, atol=tol)


def test_replicas_reject_bad_layout(monkeypatch):
    mesh = create_device_mesh(
        ici_parallelism=[1, 8], dcn_parallelism=[1, 1], devices=jax.devices()[:8]
    )
    monkeypatch.setattr(moe, "MOE_REPLICAS", 2)
    with jax.set_mesh(mesh), pytest.raises(ValueError, match="replicas"):
        moe.EPMoE(
            hidden_size=32,
            num_experts=8,
            num_experts_per_tok=4,
            ep_size=2,
            mesh=mesh,
            intermediate_dim=16,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
        )
