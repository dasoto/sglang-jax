"""Compact torus-block tensor groups (mesh_utils.host_block_device_array)."""

import itertools
import unittest
from dataclasses import dataclass

from sgl_jax.srt.utils.mesh_utils import host_block_device_array


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    coords: tuple


def _grid(width=16, height=16):
    ids = itertools.count()
    # Shuffle the enumeration order so the helper cannot rely on it.
    cells = [(x, y) for y in range(height) for x in range(width)]
    cells = cells[1::2] + cells[::2]
    return [_FakeDevice(next(ids), (x, y, 0)) for x, y in cells]


def _extent(group):
    xs = [d.coords[0] for d in group]
    ys = [d.coords[1] for d in group]
    return max(ys) - min(ys) + 1, max(xs) - min(xs) + 1


class HostBlockTest(unittest.TestCase):

    def test_two_row_blocks_partition_the_slice(self):
        devices = _grid()
        for tp, shape in ((4, (2, 2)), (8, (2, 4)), (16, (2, 8)), (32, (2, 16))):
            arr = host_block_device_array(devices, 256 // tp, tp)
            self.assertEqual(arr.shape, (256 // tp, tp))
            self.assertEqual({d.id for d in arr.flatten()}, {d.id for d in devices})
            for group in arr:
                self.assertEqual(_extent(group), shape, f"tp={tp}")

    def test_falls_back_to_square_block(self):
        # 6 columns: 2 x 4 blocks do not divide it, 4 x 2 blocks do.
        arr = host_block_device_array(_grid(width=6, height=8), 6, 8)
        self.assertIsNotNone(arr)
        for group in arr:
            self.assertEqual(_extent(group), (4, 2))

    def test_rejects_non_grid_devices(self):
        self.assertIsNone(host_block_device_array(_grid()[:-1], 255, 1))
        no_coords = [_FakeDevice(i, ()) for i in range(16)]
        self.assertIsNone(host_block_device_array(no_coords, 4, 4))
        three_d = [_FakeDevice(i, (i % 4, (i // 4) % 2, i // 8)) for i in range(16)]
        self.assertIsNone(host_block_device_array(three_d, 4, 4))


if __name__ == "__main__":
    unittest.main()
