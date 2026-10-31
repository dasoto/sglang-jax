"""Token-sharded MoE routing (Kimi-K3, SGLANG_JAX_MOE_SHARDED_ROUTING).

The router scores, biased top-k and dynamic EPLB replica remap normally run on each DP group's
tokens replicated over "tensor". Run on P(("data", "tensor"), None) scores instead (each chip
routing 1/tp of the group's tokens) and gathered back, they must give identical weights and
physical expert ids.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/layers/test_sharded_routing.py -q
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

from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.models.kimi_k3 import _gather_routing

H, E, TOPK, REDUNDANT = 64, 32, 4, 8


def _mesh():
    return jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _dispatch_info(mesh):
    """Dynamic dispatch: logical experts 0..REDUNDANT-1 have a second physical replica."""
    l2all = np.full((1, E, 2), -1, np.int32)
    l2all[0, :, 0] = np.arange(E)
    l2all[0, :REDUNDANT, 1] = E + np.arange(REDUNDANT)
    num_valid = np.sum(l2all != -1, axis=2, dtype=np.int32)
    slots = np.arange(16, dtype=np.int32)[None, None, :]
    choice = np.take_along_axis(l2all, slots % np.maximum(1, num_valid[:, :, None]), axis=2)
    p2l = np.concatenate([np.arange(E), np.arange(REDUNDANT)])[None].astype(np.int32)
    info = ExpertLocationMetadata(
        ep_dispatch_algorithm="dynamic",
        logical_to_rank_dispatch_physical_map=l2all[:, :, 0],
        logical_to_all_physical_map=l2all,
        logical_to_all_physical_map_num_valid=num_valid,
        physical_to_logical_map=p2l,
        logical_to_choice_physical_map=choice.reshape(1, -1).astype(np.int32),
        num_physical_experts=E + REDUNDANT,
    )
    return info.replicate_on(mesh)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("T", [64, 256])
def test_token_sharded_routing_matches_replicated(T):
    mesh = _mesh()
    with jax.set_mesh(mesh):
        gate = GateLogit(
            input_size=H,
            num_experts=E,
            enable_expert_bias=True,
            weight_dtype=jnp.bfloat16,
            kernel_dtype=jnp.bfloat16,
            compute_dtype=jnp.float32,
            bias_dtype=jnp.float32,
            score_func="sigmoid",
            kernel_axes=(None, "tensor"),
        )
        topk = TopK(
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=1.0,
            layer_id=0,
            mesh=mesh,
        )
        info = _dispatch_info(mesh)
        hs = jax.device_put(
            jnp.asarray(np.random.default_rng(T).standard_normal((T, H)), jnp.bfloat16),
            NamedSharding(mesh, P("data", None)),
        )
        token_sharded = NamedSharding(mesh, P(("data", "tensor"), None))

        @jax.jit
        def replicated(h):
            return topk(gate(h), gate.bias.value, dispatch_info=info)

        @jax.jit
        def sharded(h):
            scores = gate(h, out_sharding=token_sharded)
            assert jax.typeof(scores).sharding.spec == P(("data", "tensor"), None)
            w, ids = topk(
                scores,
                gate.bias.value,
                dispatch_info=info,
                routing_sharding=token_sharded,
            )
            return _gather_routing(mesh, w, ids)

        w_ref, ids_ref = replicated(hs)
        w_new, ids_new = sharded(hs)
        assert ids_new.sharding.spec == ids_ref.sharding.spec == P("data", None)
        assert w_new.dtype == w_ref.dtype and ids_new.dtype == ids_ref.dtype
        w_ref, ids_ref, w_new, ids_new = map(np.asarray, (w_ref, ids_ref, w_new, ids_new))
    np.testing.assert_array_equal(ids_new, ids_ref)
    np.testing.assert_array_equal(w_new, w_ref)
    # The remap used both replicas of the redundant experts.
    assert np.any(ids_ref >= E)


def _eqns(jaxpr):
    """Every equation of ``jaxpr``, nested jaxprs included."""
    for eqn in jaxpr.eqns:
        yield eqn
        for v in eqn.params.values():
            for sub in v if isinstance(v, (list, tuple)) else (v,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _eqns(inner)


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
def test_gather_routing_keeps_ids_integer():
    """The packed router outputs stay integer on their way through the gather.

    Expert ids bitcast to f32 are denormals, and XLA lowers the concatenate to a maximum of
    -inf-padded operands, a float op that flushes denormals to zero on TPU (every id -> 0).
    CPU keeps denormals, so the round trip alone cannot catch that; check the bitcasts.
    """
    mesh = _mesh()
    T = 64
    rng = np.random.default_rng(0)
    with jax.set_mesh(mesh):
        token_sharded = NamedSharding(mesh, P(("data", "tensor"), None))
        w = jax.device_put(jnp.asarray(rng.random((T, TOPK)), jnp.float32), token_sharded)
        ids = jax.device_put(jnp.asarray(rng.integers(1, 896, (T, TOPK)), jnp.int32), token_sharded)
        jaxpr = jax.make_jaxpr(lambda a, b: _gather_routing(mesh, a, b))(w, ids).jaxpr
        packed = [e for e in _eqns(jaxpr) if e.primitive.name in ("concatenate", "reshard")]
        assert {e.primitive.name for e in packed} == {"concatenate", "reshard"}
        for e in packed:
            dtype = e.outvars[0].aval.dtype
            assert jnp.issubdtype(dtype, jnp.integer), f"{e.primitive.name} carries {dtype}"
        w_out, ids_out = jax.jit(lambda a, b: _gather_routing(mesh, a, b))(w, ids)
        assert ids_out.sharding.spec == w_out.sharding.spec == P("data", None)
        assert ids_out.dtype == jnp.int32 and w_out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(ids_out), np.asarray(ids))
    np.testing.assert_array_equal(np.asarray(w_out), np.asarray(w))
