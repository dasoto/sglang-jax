"""Regression check for mutable grammar masks with cached sampler inputs."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.sampling import sampling_batch_info
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata


def test_changed_grammar_mask_is_uploaded():
    mesh = Mesh(np.array(jax.devices()), ("data",))
    mask = np.zeros((4, 3), dtype=np.int32)
    metadata = SamplingMetadata(
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        temperatures=jnp.ones((4, 1)),
        top_ps=None,
        top_ks=None,
        min_ps=None,
        sampling_seeds=None,
        positions=None,
    )
    metadata.update_vocab_mask(mask, mesh, 65)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), mask)
    mask[:] = 17
    metadata.update_vocab_mask(mask, mesh, 65)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), mask)
    assert bool(metadata.apply_vocab_mask)
    metadata.update_vocab_mask(None, mesh, 65)
    assert not bool(metadata.apply_vocab_mask)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), np.zeros((4, 3)))


def test_sampler_uploads_avoid_device_put(monkeypatch):
    """jax.device_put of a host value runs a cross-process assert_equal collective on
    multi-host meshes; the scheduler thread must not launch one while the overlap
    thread runs the forward step, so these uploads go through device_array."""

    def fail(*args, **kwargs):
        raise AssertionError("sampler inputs must be uploaded with device_array")

    monkeypatch.setattr(sampling_batch_info.jax, "device_put", fail)
    sampling_batch_info._sampler_bools.cache_clear()
    sampling_batch_info._empty_vocab_mask.cache_clear()
    mesh = Mesh(np.array(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, PartitionSpec())
    false, true = sampling_batch_info._sampler_bools(sharding)
    assert not bool(false) and bool(true)
    np.testing.assert_array_equal(
        np.asarray(sampling_batch_info._empty_vocab_mask(4, 65, sharding)),
        np.zeros((4, 3)),
    )
    metadata = SamplingMetadata(
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        temperatures=jnp.ones((4, 1)),
        top_ps=None,
        top_ks=None,
        min_ps=None,
        sampling_seeds=None,
        positions=None,
    )
    mask = np.full((4, 3), 5, dtype=np.int32)
    metadata.update_vocab_mask(mask, mesh, 65)
    np.testing.assert_array_equal(np.asarray(metadata.vocab_mask), mask)
    assert bool(metadata.apply_vocab_mask)
