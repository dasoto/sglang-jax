"""GLM-5 EPLB wiring: the expert-location hook and per-layer placement in the MoE weight mappings."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.eplb import expert_location
from sgl_jax.srt.models import glm5_moe

NUM_LAYERS, NUM_EXPERTS = 6, 8


@pytest.fixture
def placement():
    p2l = np.tile(np.arange(NUM_EXPERTS), (NUM_LAYERS, 1))
    p2l[4] = np.random.default_rng(0).permutation(NUM_EXPERTS)
    expert_location.set_global_expert_location_metadata(
        SimpleNamespace(physical_to_logical_map=p2l)
    )
    yield p2l
    expert_location.set_global_expert_location_metadata(None)


def _mock_model():
    config = SimpleNamespace(
        n_routed_experts=NUM_EXPERTS, n_shared_experts=0, moe_backend="epmoe", index_topk=4
    )
    return SimpleNamespace(config=config)


def test_expert_location_config_reads_n_routed_experts():
    cfg = SimpleNamespace(n_routed_experts=256, num_hidden_layers=78, n_group=1)
    assert glm5_moe.GlmMoeDsaForCausalLM.get_expert_location_config(cfg) == {
        "num_experts": 256,
        "num_layers": 78,
        "num_groups": 1,
    }


def test_moe_mappings_use_layer_placement(placement):
    mappings = glm5_moe.Glm5ForCausalLM._create_moe_layer_mappings(_mock_model(), 4, 4, False)
    for name in ("wi_0", "wi_1", "wo"):
        np.testing.assert_array_equal(
            mappings[f"model.layers.4.mlp.{name}"].physical_to_logical_map, placement[4]
        )


def test_moe_mappings_without_placement_or_beyond_it(placement):
    # The NextN layer (index num_hidden_layers) is not covered by the placement.
    mappings = glm5_moe.Glm5ForCausalLM._create_moe_layer_mappings(
        _mock_model(), NUM_LAYERS, NUM_LAYERS, False
    )
    assert mappings[f"model.layers.{NUM_LAYERS}.mlp.wi_0"].physical_to_logical_map is None
    expert_location.set_global_expert_location_metadata(None)
    mappings = glm5_moe.Glm5ForCausalLM._create_moe_layer_mappings(_mock_model(), 4, 4, False)
    assert mappings["model.layers.4.mlp.wi_0"].physical_to_logical_map is None
