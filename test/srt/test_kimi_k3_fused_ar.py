"""Kimi-K3 LatentMoE output: one all-reduce for the routed up-projection plus the shared experts.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/test_kimi_k3_fused_ar.py -q
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

from sgl_jax.srt.models.kimi_k3 import _row_parallel_sum


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (CPU) devices")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_row_parallel_sum_matches_two_linears(dtype):
    mesh = jax.make_mesh(
        (2, 4), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    rng = np.random.default_rng(0)
    T, k1, k2, n = 16, 64, 96, 32
    x1, x2 = rng.standard_normal((T, k1)), rng.standard_normal((T, k2))
    w1, w2 = rng.standard_normal((k1, n)) * 0.1, rng.standard_normal((k2, n)) * 0.1
    with jax.set_mesh(mesh):
        put = lambda a, spec: jax.device_put(jnp.asarray(a, dtype), NamedSharding(mesh, spec))
        out = _row_parallel_sum(
            mesh,
            put(x1, P("data", "tensor")),
            put(w1, P("tensor", None)),
            put(x2, P("data", "tensor")),
            put(w2, P("tensor", None)),
        )
        assert out.sharding.spec == P("data", None)
        out = np.asarray(out.astype(jnp.float32))
    cast = lambda a: np.asarray(jnp.asarray(a, dtype).astype(jnp.float32))
    ref = cast(x1) @ cast(w1) + cast(x2) @ cast(w2)
    tol = 1e-5 if dtype == jnp.float32 else 3e-2
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol * np.abs(ref).max())
