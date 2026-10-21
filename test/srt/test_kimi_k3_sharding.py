"""Kimi-K3 full-size layout checks on a virtual v5e-256 mesh (CPU, no weights).

* Builds the real 93-layer / 896-expert text model abstractly
  (``nnx.eval_shape``) on a ``(data=8, tensor=32)`` mesh of 256 virtual CPU
  devices with ``ep_size=128`` (EPMoE sub-mesh ``expert=128 x tensor=2``) and
  MXFP4 experts, and checks that every parameter sharding divides its shape.
  Reports the per-chip weight bytes of this layout.
* Checks that the weight mapping consumes exactly the text tensors of the
  release checkpoint: against ``model.safetensors.index.json`` when
  ``KIMI_K3_INDEX_JSON`` points at it, else against a key set reconstructed
  from per-layer templates snapshotted from that index (497,052 text tensors).

Run (the mesh test needs 256 virtual devices):
    JAX_NUM_CPU_DEVICES=256 python -m pytest test/srt/test_kimi_k3_sharding.py -v -s
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import math
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

K3_TEXT_CONFIG = dict(
    vocab_size=163840,
    hidden_size=7168,
    intermediate_size=33792,
    num_hidden_layers=93,
    num_attention_heads=96,
    num_key_value_heads=96,
    rms_norm_eps=1e-5,
    hidden_act="situ",
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    attn_res_block_size=12,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    mla_use_nope=True,
    mla_use_output_gate=True,
    num_experts=896,
    num_experts_per_token=16,
    moe_intermediate_size=3072,
    routed_expert_hidden_size=3584,
    latent_moe_use_norm=True,
    num_shared_experts=2,
    moe_router_activation_func="sigmoid",
    moe_renormalize=True,
    moe_layer_freq=1,
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    num_expert_group=1,
    topk_group=1,
    linear_attn_config=dict(
        full_attn_layers=[4 * i for i in range(1, 24)] + [93],  # 1-indexed
        kda_layers=[i for i in range(1, 93) if i % 4],
        num_heads=96,
        head_dim=128,
        short_conv_kernel_size=4,
        gate_lower_bound=-5.0,
        use_full_rank_gate=True,
    ),
    tie_word_embeddings=False,
    max_position_embeddings=1048576,
)

# Per-layer key templates of the release index (vision/projector excluded).
_LAYER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
)
_KDA_KEYS = tuple(
    f"self_attn.{k}"
    for k in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
        "dt_bias",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "b_proj.weight",
        "o_norm.weight",
    )
)
_MLA_KEYS = tuple(
    f"self_attn.{k}"
    for k in (
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
    )
)
_DENSE_KEYS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
_MOE_KEYS = tuple(
    f"block_sparse_moe.{k}"
    for k in (
        "gate.weight",
        "gate.e_score_correction_bias",
        "routed_expert_down_proj.weight",
        "routed_expert_up_proj.weight",
        "routed_expert_norm.weight",
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    )
)


def _expected_text_keys(cfg: dict) -> set[str]:
    src = "language_model.model"
    keys = {
        "language_model.lm_head.weight",
        f"{src}.embed_tokens.weight",
        f"{src}.norm.weight",
        f"{src}.output_attn_res_norm.weight",
        f"{src}.output_attn_res_proj.weight",
    }
    kda = set(cfg["linear_attn_config"]["kda_layers"])
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"{src}.layers.{i}"
        attn = _KDA_KEYS if (i + 1) in kda else _MLA_KEYS
        ffn = _DENSE_KEYS if i < cfg["first_k_dense_replace"] else _MOE_KEYS
        keys.update(f"{prefix}.{k}" for k in _LAYER_KEYS + attn + ffn)
        if i >= cfg["first_k_dense_replace"]:
            for e in range(cfg["num_experts"]):
                for proj in ("w1", "w2", "w3"):
                    for suffix in ("weight_packed", "weight_scale"):
                        keys.add(f"{prefix}.block_sparse_moe.experts.{e}.{proj}.{suffix}")
    return keys


def _build_abstract_model(mesh, ep_size: int):
    from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

    config = KimiK3Config(
        text_config=dict(K3_TEXT_CONFIG), architectures=["KimiK3ForConditionalGeneration"]
    )
    config.ep_size = ep_size
    config.moe_backend = "epmoe"
    config.use_absorbed_mla = True
    config.quantization_config = QuantizationConfig(
        is_static_checkpoint=True, linear_rules=[], moe_weight_format="mxfp4"
    )
    with jax.set_mesh(mesh):
        return nnx.eval_shape(
            lambda: KimiK3ForConditionalGeneration(config, mesh=mesh, dtype=jnp.bfloat16)
        )


class TestKimiK3FullSizeLayout(unittest.TestCase):
    def test_v5e256_layout_shards_evenly(self):
        if len(jax.devices()) < 256:
            self.skipTest("needs JAX_NUM_CPU_DEVICES=256")
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        mesh = create_device_mesh(
            ici_parallelism=[8, 32], dcn_parallelism=[1, 1], devices=jax.devices()[:256]
        )
        model = _build_abstract_model(mesh, ep_size=128)
        per_chip = {"experts": 0, "dense_sharded": 0, "replicated": 0}
        for path, param in nnx.state(model, nnx.Param).flat_state():
            value = param.value
            name = ".".join(map(str, path))
            shard_shape = value.sharding.shard_shape(value.shape)  # raises if uneven
            nbytes = math.prod(shard_shape) * jnp.dtype(value.dtype).itemsize
            if "experts.w" in name:
                per_chip["experts"] += nbytes
            elif shard_shape == tuple(value.shape):
                per_chip["replicated"] += nbytes
            else:
                per_chip["dense_sharded"] += nbytes
        gb = {k: v / 1e9 for k, v in per_chip.items()}
        print(
            "[kimi_k3 v5e-256 TP32xDP8 EP128xETP2] per-chip weights GB: "
            + ", ".join(f"{k}={v:.2f}" for k, v in gb.items())
            + f", total={sum(gb.values()):.2f}"
        )
        # Routed experts: 1.446 TB of MXFP4 split 256 ways.
        self.assertAlmostEqual(gb["experts"], 1_446_456_066_048 / 256 / 1e9, places=2)
        self.assertLess(sum(gb.values()), 16.0)

        layer = model.model.layers[1]
        experts = layer.block_sparse_moe.experts
        self.assertEqual(experts.moe_mesh.shape, {"expert": 128, "tensor": 2})
        self.assertEqual(experts.experts_per_device, 7)
        self.assertEqual(
            experts.w2_weight_packed.value.sharding.shard_shape(
                experts.w2_weight_packed.value.shape
            ),
            (7, 768, 3584),
        )
        kda = layer.self_attn
        self.assertEqual(kda.q_proj.weight.value.sharding.shard_shape((7168, 12288)), (7168, 384))

    def test_weight_mapping_covers_release_checkpoint(self):
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        mesh = create_device_mesh(
            ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=jax.devices()[:1]
        )
        model = _build_abstract_model(mesh, ep_size=1)
        mappings = model._create_weight_mappings("language_model.")
        sources = []
        for name, spec in mappings.items():
            sources.extend(spec.sources or (name,))
        self.assertEqual(len(sources), len(set(sources)), "duplicate checkpoint readers")
        index_path = os.environ.get("KIMI_K3_INDEX_JSON")
        if index_path:
            with open(index_path) as f:
                expected = {
                    k
                    for k in json.load(f)["weight_map"]
                    if not k.startswith(("vision_tower.", "mm_projector."))
                }
        else:
            expected = _expected_text_keys(K3_TEXT_CONFIG)
        self.assertEqual(len(expected), 497_052)
        self.assertEqual(set(sources) - expected, set(), "mapped names missing from checkpoint")
        self.assertEqual(expected - set(sources), set(), "checkpoint tensors left unmapped")


if __name__ == "__main__":
    unittest.main()
