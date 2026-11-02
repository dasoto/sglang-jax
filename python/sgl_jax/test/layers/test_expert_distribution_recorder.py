"""Expert distribution recorder dumps: missing directories are created and a failed write is logged."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from sgl_jax.srt.layers.routed_experts_capturer import _ExpertDistributionRecorder


def _record(recorder, steps):
    ids = np.array([[0, 1], [1, 3], [-1, -1]], dtype=np.int32)
    for _ in range(steps):
        recorder.add_topk_ids([None, ids], real_bs_per_dp=[2], per_dp_bs_size=3)


def test_dump_creates_missing_directory(tmp_path):
    out = tmp_path / "missing" / "dist.npy"
    recorder = _ExpertDistributionRecorder(
        num_layers=2, buffer_size=2, output_file=str(out), physical_expert_counts=4
    )
    _record(recorder, 2)
    files = list((tmp_path / "missing").glob("dist_*.npy"))
    assert len(files) == 1
    counts = np.load(files[0], allow_pickle=True).item()["logical_count"]
    np.testing.assert_array_equal(counts, [[0, 0, 0, 0], [2, 4, 0, 2]])


def test_failed_dump_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    recorder = _ExpertDistributionRecorder(
        num_layers=2, buffer_size=1, output_file=str(blocker / "dist.npy"), physical_expert_counts=4
    )
    _record(recorder, 1)  # logs the failure instead of raising on the sync thread
