"""RecurrentStatePool -- buffer pool for linear recurrent layers (KDA/GDN)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

# Module-level cache for jitted zero-allocators (recurrent/conv buffers).
_RECURRENT_ZERO_ALLOCATOR_CACHE: dict = {}


def _get_recurrent_zero_allocator(shape, dtype, sharding):
    """Return a cached jax.jit(jnp.zeros) allocator for recurrent/conv buffers."""
    key = (
        id(sharding.mesh),
        tuple(shape),
        str(jnp.dtype(dtype)),
        repr(sharding.spec),
        getattr(sharding, "memory_kind", None),
    )
    if key not in _RECURRENT_ZERO_ALLOCATOR_CACHE:
        _RECURRENT_ZERO_ALLOCATOR_CACHE[key] = jax.jit(
            partial(jnp.zeros, shape=tuple(shape), dtype=dtype),
            out_shardings=sharding,
        )
    return _RECURRENT_ZERO_ALLOCATOR_CACHE[key]


@partial(jax.jit, static_argnames=("out_sharding",))
def _jit_gather_recurrent_slot(
    buf: jax.Array, global_slot: jax.Array, out_sharding: NamedSharding
) -> jax.Array:
    if out_sharding is not None and any(
        t == jax.sharding.AxisType.Auto for t in out_sharding.mesh.axis_types
    ):
        return jax.lax.with_sharding_constraint(buf.at[global_slot].get(), out_sharding)

    @partial(jax.sharding.auto_axes, out_sharding=out_sharding)
    def _impl(b, idx):
        return b.at[idx].get()

    return _impl(buf, global_slot)


@partial(jax.jit, donate_argnames=("buf",))
def _jit_write_recurrent_slot(
    buf: jax.Array, global_slot: jax.Array, val: jax.Array
) -> jax.Array:
    sharding = jax.typeof(buf).sharding
    if sharding is not None and any(
        t == jax.sharding.AxisType.Explicit for t in sharding.mesh.axis_types
    ):

        @partial(jax.sharding.auto_axes, out_sharding=sharding)
        def _impl(b, idx, v):
            return b.at[idx].set(v)

        return _impl(buf, global_slot, val)
    return buf.at[global_slot].set(val)


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _resolve_dtype(env_var: str, default):
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


@dataclass(frozen=True)
class RecurrentStateDType:
    conv: jnp.dtype
    temporal: jnp.dtype


@dataclass(frozen=True)
class LinearRecurrentStateParams:
    layers: list[int]
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    dtype: RecurrentStateDType
    # GDN has asymmetric K vs V projection widths (e.g.
    # Qwen3.5 GDN: num_k_heads=16/head_k_dim=128 vs num_v_heads=32/head_v_dim=128).
    # When None (KDA / Lightning / Bailing), RecurrentStatePool falls back to
    # treating K dim = V dim.
    num_k_heads: int | None = None
    head_k_dim: int | None = None


def recurrent_state_dtype() -> RecurrentStateDType:
    return RecurrentStateDType(
        conv=_resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16),
        temporal=_resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32),
    )


@register_pytree_node_class
class RecurrentStatePool:

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        mesh: Mesh,
        dp_size: int = 1,
        recurrent_partition_axis: str = "tensor",
        conv_partition_axis: str = "tensor",
        data_partition_axis: str = "data",
        temporal_dtype=None,
        conv_dtype=None,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        abstract: bool = False,
    ):
        """`size` is the **global** number of valid slots across all DP ranks
        (mirrors MHATokenToKVPool.size semantics). Internally we partition by
        DP: each rank gets `size // dp_size` valid slots + 1 dummy slot at
        index 0, so total buffer slots = size + dp_size.
        """
        state_dtype = recurrent_state_dtype()
        if temporal_dtype is None:
            temporal_dtype = state_dtype.temporal
        if conv_dtype is None:
            conv_dtype = state_dtype.conv
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids), (
            f"linear_recurrent_layer_ids must not contain duplicates, "
            f"got {linear_recurrent_layer_ids}"
        )
        self.linear_recurrent_layer_ids: list[int] = list(linear_recurrent_layer_ids)
        self.layers_mapping: dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(self.linear_recurrent_layer_ids)
        }
        self.num_linear_recurrent_layers: int = len(self.linear_recurrent_layer_ids)

        assert (
            size % dp_size == 0
        ), f"RecurrentStatePool size ({size}) must be divisible by dp_size ({dp_size})."

        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size

        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k

        # Each rank reserves slot 0 as a dummy → +1 per rank.
        self.total_slots = size + dp_size

        self.mesh = mesh
        self.recurrent_partition_axis = recurrent_partition_axis
        self.conv_partition_axis = conv_partition_axis
        self.data_partition_axis = data_partition_axis

        recurrent_axis_size = mesh.shape[recurrent_partition_axis]
        conv_axis_size = mesh.shape[conv_partition_axis]
        assert num_heads % recurrent_axis_size == 0, (
            f"num_heads {num_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert num_k_heads % recurrent_axis_size == 0, (
            f"num_k_heads {num_k_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert self.proj_size % conv_axis_size == 0, (
            f"proj_size {self.proj_size} must be divisible by "
            f"'{conv_partition_axis}' size {conv_axis_size}"
        )

        self.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        self.conv_sharding = NamedSharding(mesh, P(data_partition_axis, conv_partition_axis, None))

        self.recurrent_buffers, self.conv_buffers = self._create_buffers(abstract=abstract)

    def _create_buffers(self, *, abstract: bool = False) -> tuple[list, list]:
        recurrent_shape = (self.total_slots, self.num_heads, self.head_dim, self.head_dim)
        conv_shape = (self.total_slots, self.proj_size, self.conv_kernel_size - 1)
        temporal_dtype = self.temporal_dtype
        conv_dtype = self.conv_dtype

        if abstract:
            return (
                [
                    jax.ShapeDtypeStruct(
                        recurrent_shape, temporal_dtype, sharding=self.recurrent_sharding
                    )
                    for _ in range(self.num_linear_recurrent_layers)
                ],
                [
                    [jax.ShapeDtypeStruct(conv_shape, conv_dtype, sharding=self.conv_sharding)]
                    for _ in range(self.num_linear_recurrent_layers)
                ],
            )

        alloc_recurrent = _get_recurrent_zero_allocator(
            recurrent_shape, temporal_dtype, self.recurrent_sharding
        )
        alloc_conv = _get_recurrent_zero_allocator(conv_shape, conv_dtype, self.conv_sharding)
        with jax.set_mesh(self.mesh):
            recurrent_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                recurrent_buffers.append(alloc_recurrent())

            conv_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                inner = []
                inner.append(alloc_conv())
                conv_buffers.append(inner)

        return recurrent_buffers, conv_buffers

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def replace_buffer(self, buffers) -> None:
        new_recurrent, new_conv = buffers

        assert len(new_recurrent) == self.num_linear_recurrent_layers
        assert len(new_conv) == self.num_linear_recurrent_layers

        # tp_size==1 sharding fix: see MHATokenToKVPool.replace_buffer
        tp_degenerate = self.mesh.shape.get("tensor", 1) == 1
        for layer in range(self.num_linear_recurrent_layers):
            buf = new_recurrent[layer]
            if tp_degenerate:
                buf = jax.device_put(buf, self.recurrent_sharding)
            self.recurrent_buffers[layer] = buf

            assert len(new_conv[layer]) == len(self.conv_buffers[layer])
            for i in range(len(new_conv[layer])):
                cbuf = new_conv[layer][i]
                if tp_degenerate:
                    cbuf = jax.device_put(cbuf, self.conv_sharding)
                self.conv_buffers[layer][i] = cbuf

    def clear(self) -> None:
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])

    def global_slot_index(self, dp_rank: int, local_slot: int) -> int:
        slots_per_rank = self.total_slots // self.dp_size
        return int(dp_rank) * slots_per_rank + int(local_slot)

    def _slot_transfer_specs_and_layer_shardings(
        self,
    ) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct, NamedSharding, NamedSharding]:
        tp_size = self.mesh.shape.get(self.recurrent_partition_axis, 1)
        conv_tp_size = self.mesh.shape.get(self.conv_partition_axis, 1)
        dp_size = (
            self.mesh.shape.get(self.data_partition_axis, 1) if self.data_partition_axis else 1
        )
        if dp_size > 1 and self.data_partition_axis and self.head_dim % dp_size == 0:
            t_layer_pspec = P(self.recurrent_partition_axis, self.data_partition_axis, None)
            t_stacked_pspec = P(None, self.recurrent_partition_axis, self.data_partition_axis, None)
        else:
            t_layer_pspec = P(self.recurrent_partition_axis, None, None)
            t_stacked_pspec = P(None, self.recurrent_partition_axis, None, None)

        if dp_size > 1 and self.data_partition_axis and self.proj_size % (dp_size * conv_tp_size) == 0:
            c_layer_pspec = P((self.data_partition_axis, self.conv_partition_axis), None)
            c_stacked_pspec = P(None, (self.data_partition_axis, self.conv_partition_axis), None)
        else:
            c_layer_pspec = P(self.conv_partition_axis, None)
            c_stacked_pspec = P(None, self.conv_partition_axis, None)

        t_spec = jax.ShapeDtypeStruct(
            (self.num_linear_recurrent_layers, self.num_heads, self.head_dim, self.head_dim),
            self.temporal_dtype,
            sharding=NamedSharding(self.mesh, t_stacked_pspec),
        )
        c_spec = jax.ShapeDtypeStruct(
            (self.num_linear_recurrent_layers, self.proj_size, self.conv_kernel_size - 1),
            self.conv_dtype,
            sharding=NamedSharding(self.mesh, c_stacked_pspec),
        )
        return (
            t_spec,
            c_spec,
            NamedSharding(self.mesh, t_layer_pspec),
            NamedSharding(self.mesh, c_layer_pspec),
        )

    def slot_transfer_specs(self) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
        t_spec, c_spec, _, _ = self._slot_transfer_specs_and_layer_shardings()
        return t_spec, c_spec

    def extract_slot(self, dp_rank: int, local_slot: int) -> tuple[jax.Array, jax.Array]:
        gslot = jax.device_put(
            jnp.asarray(self.global_slot_index(dp_rank, local_slot), dtype=jnp.int32),
            NamedSharding(self.mesh, P()),
        )
        _, _, t_layer_sharding, c_layer_sharding = self._slot_transfer_specs_and_layer_shardings()
        t_layers = [
            _jit_gather_recurrent_slot(self.recurrent_buffers[l], gslot, t_layer_sharding)
            for l in range(self.num_linear_recurrent_layers)
        ]
        c_layers = [
            _jit_gather_recurrent_slot(self.conv_buffers[l][0], gslot, c_layer_sharding)
            for l in range(self.num_linear_recurrent_layers)
        ]
        return jnp.stack(t_layers, axis=0), jnp.stack(c_layers, axis=0)

    def install_slot(
        self,
        dp_rank: int,
        local_slot: int,
        temporal_stacked: jax.Array,
        conv_stacked: jax.Array,
    ) -> None:
        gslot = jax.device_put(
            jnp.asarray(self.global_slot_index(dp_rank, local_slot), dtype=jnp.int32),
            NamedSharding(self.mesh, P()),
        )
        for l in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[l] = _jit_write_recurrent_slot(
                self.recurrent_buffers[l], gslot, temporal_stacked[l]
            )
            self.conv_buffers[l][0] = _jit_write_recurrent_slot(
                self.conv_buffers[l][0], gslot, conv_stacked[l]
            )

    def copy_slots(self, src_indices, dst_indices):
        """Clone src->dst slots across all layers; rows with src==0 keep dst.
        Indices are per-DP-rank local; returns new buffers for the donated pool."""
        mesh = self.mesh
        data_axis = self.data_partition_axis

        def _temporal(buf, src, dst):
            # Donated-buffer aliasing barriers: without them the scatter races the
            # gather under multi-host SPMD -> NaN. Value-preserving; do not remove.
            buf = jax.lax.optimization_barrier(buf)
            val = jnp.where((src == 0).reshape(-1, 1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        def _conv(buf, src, dst):
            buf = jax.lax.optimization_barrier(buf)  # see _temporal
            val = jnp.where((src == 0).reshape(-1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        copy_temporal = jax.shard_map(
            _temporal,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.recurrent_partition_axis, None, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.recurrent_partition_axis, None, None),
            check_vma=False,
        )
        copy_conv = jax.shard_map(
            _conv,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.conv_partition_axis, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.conv_partition_axis, None),
            check_vma=False,
        )

        new_recurrent = [
            copy_temporal(buf, src_indices, dst_indices) for buf in self.recurrent_buffers
        ]
        new_conv = [
            [copy_conv(cbuf, src_indices, dst_indices) for cbuf in inner]
            for inner in self.conv_buffers
        ]
        return new_recurrent, new_conv

    # --- pytree ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
            self.total_slots,
            self.num_heads,
            self.head_dim,
            self.num_k_heads,
            self.head_k_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
            self.mesh,
            self.recurrent_partition_axis,
            self.conv_partition_axis,
            self.data_partition_axis,
            self.recurrent_sharding,
            self.conv_sharding,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            size,
            dp_size,
            total_slots,
            num_heads,
            head_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
            mesh,
            recurrent_partition_axis,
            conv_partition_axis,
            data_partition_axis,
            recurrent_sharding,
            conv_sharding,
        ) = aux_data
        obj = cls.__new__(cls)
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.num_k_heads = num_k_heads
        obj.head_k_dim = head_k_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        obj.proj_size = proj_v + 2 * proj_k
        obj.mesh = mesh
        obj.recurrent_partition_axis = recurrent_partition_axis
        obj.conv_partition_axis = conv_partition_axis
        obj.data_partition_axis = data_partition_axis
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj
