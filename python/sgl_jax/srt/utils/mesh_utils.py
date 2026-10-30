import logging
import os
from collections.abc import MutableSequence, Sequence

import jax
import numpy as np
from jax._src import mesh_utils

from sgl_jax.srt.utils.jax_utils import get_device_id_offset

logger = logging.getLogger(__name__)

default_mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
]


def host_block_device_array(devices, dp: int, tp: int) -> np.ndarray | None:
    """[dp, tp] device array whose tensor groups are compact blocks of a 2D torus.

    jax's create_device_mesh lays a tensor group of a v5e 16x16 slice out as an
    open line of chips. A 2 x (tp/2) block puts every chip on the block's
    perimeter, so the group is a closed ring (2x2 is one 4-chip host). On v5e-256
    this made the absorbed-query all-gather and partial-output reduce-scatter of
    an MLA layer up to 2x faster and ~30% lower latency for TP4 and TP8. Falls
    back to the most square block when 2-row blocks do not divide the grid, and
    returns None when the devices do not form a full 2D grid.
    """
    coords = [tuple(getattr(d, "coords", ())) for d in devices]
    if len(devices) != dp * tp or any(len(c) < 2 for c in coords):
        return None
    if any(len(c) > 2 and c[2] != 0 for c in coords):
        return None
    xs = sorted({c[0] for c in coords})
    ys = sorted({c[1] for c in coords})
    width, height = len(xs), len(ys)
    if width * height != len(devices):
        return None
    grid = np.empty((height, width), dtype=object)
    for d, c in zip(devices, coords):
        grid[ys.index(c[1]), xs.index(c[0])] = d
    shapes = [
        (bh, tp // bh)
        for bh in range(1, tp + 1)
        if tp % bh == 0 and height % bh == 0 and width % (tp // bh) == 0
    ]
    if not shapes:
        return None
    # 2-row blocks (closed rings) first, otherwise the most square block.
    bh, bw = min(shapes, key=lambda s: (s[0] != 2, abs(s[0] - s[1]), s[0]))
    blocks = grid.reshape(height // bh, bh, width // bw, bw).transpose(0, 2, 1, 3)
    return blocks.reshape(dp, tp)


def create_device_mesh(
    ici_parallelism: MutableSequence[int],
    dcn_parallelism: MutableSequence[int],
    devices=None,
    device_indexes: list[int] | None = None,
    num_slices: int = 1,
    allow_split_physical_axes: bool = True,
    use_explicit_sharding: bool = True,
    mesh_axes: Sequence[str] = default_mesh_axes,
    is_full_topology: bool = False,
) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        if is_full_topology:
            raise ValueError("is_full_topology requires explicit target devices")
        devices = jax.devices()

    # Explicit devices can describe a remote compilation topology. Do not query
    # the host runtime to decide whether that topology is a device subset.
    topology_devices = devices

    offset = get_device_id_offset(devices)

    if device_indexes is not None:
        device_indexes = [idx + offset for idx in device_indexes]
        devices_dict = {device.id: device for device in devices}
        devices = [devices_dict.get(i) for i in list(set(device_indexes))]
    ici_parallelism = fill_unspecified_parallelism(ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(dcn_parallelism, num_slices)
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        all_devices = topology_devices if is_full_topology else jax.devices()
        is_subset = len(devices) < len(all_devices)
        devices_array = None
        if (
            not is_subset
            and len(ici_parallelism) == 2
            and os.environ.get("SGLANG_JAX_TP_HOST_BLOCKS", "0") == "1"
        ):
            devices_array = host_block_device_array(devices, *ici_parallelism)
            if devices_array is None:
                logger.warning(
                    "SGLANG_JAX_TP_HOST_BLOCKS=1 ignored: devices do not form a 2D grid"
                )
            else:
                logger.info(
                    "Tensor groups as compact torus blocks; group 0 coords: %s",
                    [tuple(d.coords[:2]) for d in devices_array[0]],
                )
        if devices_array is not None:
            pass
        elif is_subset:
            # JAX's create_device_mesh infers the full physical TPU topology
            # and asserts len(devices) == np.prod(dims), which fails when
            # only a subset of devices is used.  Fall back to a simple reshape.
            devices_array = np.array(devices).reshape(ici_parallelism)
        else:
            devices_array = mesh_utils.create_device_mesh(
                ici_parallelism,
                devices=devices,
                contiguous_submeshes=False,
                allow_split_physical_axes=allow_split_physical_axes,
            )

    if use_explicit_sharding:
        axis_types = (jax.sharding.AxisType.Explicit,) * len(mesh_axes)
        mesh = jax.sharding.Mesh(devices_array, mesh_axes, axis_types=axis_types)
    else:
        mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(
    parallelism: MutableSequence[int], num_devices: int
) -> MutableSequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert (
        determined_val >= 1 and determined_val.is_integer()
    ), "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism
