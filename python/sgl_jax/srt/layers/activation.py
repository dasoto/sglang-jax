from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.utils.profiling_utils import named_scope

# Keep the exact/approximate GELU distinction used by HF checkpoints.
ACT2FN = {
    "gelu": partial(jax.nn.gelu, approximate=False),
    "gelu_new": partial(jax.nn.gelu, approximate=True),
    "gelu_pytorch_tanh": partial(jax.nn.gelu, approximate=True),
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "quick_gelu": lambda x: x * jax.nn.sigmoid(1.702 * x),
    "tanh": jax.nn.tanh,
}


def situ_and_mul(
    gate: jax.Array,
    up: jax.Array,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> jax.Array:
    """SiTU-GLU (Moonshot ``SituAndMul``), computed in float32.

    ``beta * tanh(gate / beta) * sigmoid(gate) * up'`` where
    ``up' = linear_beta * tanh(up / linear_beta)`` if ``linear_beta`` is set,
    else ``up``. Kimi-K3 uses ``beta=4``, ``linear_beta=25``. The result is
    cast back to ``gate.dtype`` like the reference (``.to(x.dtype)``).
    """
    out_dtype = gate.dtype
    g = gate.astype(jnp.float32)
    u = up.astype(jnp.float32)
    act = beta * jnp.tanh(g / beta) * jax.nn.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * jnp.tanh(u / linear_beta)
    return (act * u).astype(out_dtype)


class GeluAndMul(nnx.Module):
    def __init__(self, approximate: str = "tanh"):
        self.approximate = approximate

    @named_scope
    def __call__(self, gate: jax.Array, up: jax.Array):
        if self.approximate == "tanh":
            gelu = jax.nn.gelu(gate, approximate=True)
        else:
            gelu = jax.nn.gelu(gate, approximate=False)
        out = gelu * up
        return out, None
