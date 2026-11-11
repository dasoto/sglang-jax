"""Kimi-K3.1 CPU tests: the DSA indexer and its wiring into Kimi-K3's MLA layer.

Kimi-K3.1 is Kimi-K3 plus a DSA lightning indexer on its MLA layers. The DSA
kernels and backend come from the GLM-5 DSA work and carry their own tests
(test/srt/kernels/dsa); these tests cover what Kimi-K3.1 adds:

* ``apply_kimi_k31_dsa_fields`` flattens ``text_config.dsa_config`` into the
  DeepSeek-V3.2-style fields the DSA backend and pools read, and
  ``build_index_share_map`` gives indexer slots to the MLA layers only.
* ``KimiK31DsaIndexer`` matches a NumPy transcription of the reference indexer
  (no RoPE, LayerNorm with bias at eps 1e-6, normalized Hadamard rotation).
* ``KimiK31MLAAttention`` on its DSA path hands the backend the indexer outputs
  and, with a stand-in backend that attends densely, gives exactly Kimi-K3's
  absorbed MLA output for the same weights.
* The model maps the five indexer tensors of every DSA layer.

    JAX_NUM_CPU_DEVICES=4 python -m pytest test/srt/test_kimi_k31_cpu.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import types
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _make_mesh(tp: int):
    devices = np.asarray(jax.devices()[:tp]).reshape(1, tp)
    return jax.sharding.Mesh(
        devices, ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )


def _k31_text_config(num_layers=8, mla_layers_1idx=(4, 8)):
    from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

    kda = [i for i in range(1, num_layers + 1) if i not in mla_layers_1idx]
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=4,
        routed_expert_hidden_size=128,
        linear_attn_config=dict(
            full_attn_layers=list(mla_layers_1idx),
            kda_layers=kda,
            num_heads=4,
            head_dim=64,
            short_conv_kernel_size=4,
            gate_lower_bound=-5.0,
            use_full_rank_gate=True,
        ),
        dsa_config=dict(
            dsa_layers=list(mla_layers_1idx),
            index_topk=8,
            index_head_dim=32,
            index_num_heads=4,
            index_rope_dim=0,
            index_k_norm_type="layernorm",
            index_topk_freq=1,
            index_skip_topk_offset=1,
        ),
    )


class TestKimiK31Config(unittest.TestCase):
    def test_flatten_and_index_slots(self):
        from sgl_jax.srt.configs.kimi_k3 import apply_kimi_k31_dsa_fields
        from sgl_jax.srt.kernels.dsa.ref import build_index_share_map

        cfg = _k31_text_config()
        self.assertTrue(apply_kimi_k31_dsa_fields(cfg))
        self.assertTrue(apply_kimi_k31_dsa_fields(cfg))  # idempotent
        self.assertEqual((cfg.index_topk, cfg.index_head_dim, cfg.index_n_heads), (8, 32, 4))
        self.assertEqual(cfg.dsa_layer_ids, [3, 7])
        self.assertEqual(cfg.indexer_types, ["none"] * 3 + ["full"] + ["none"] * 3 + ["full"])
        # Indexer layers are exactly the MLA (non-KDA) layers.
        self.assertEqual([i for i in range(8) if not cfg.is_kda_layer(i)], cfg.dsa_layer_ids)
        full_slot, src_slot, num_full = build_index_share_map(cfg.indexer_types, 0, 8)
        self.assertEqual((full_slot, src_slot, num_full), ({3: 0, 7: 1}, {3: 0, 7: 1}, 2))

    def test_kimi_k3_has_no_dsa_fields(self):
        from sgl_jax.srt.configs.kimi_k3 import apply_kimi_k31_dsa_fields

        cfg = _k31_text_config()
        cfg.dsa_config = None
        self.assertFalse(apply_kimi_k31_dsa_fields(cfg))
        self.assertIsNone(getattr(cfg, "indexer_types", None))

    def test_rejects_unsupported_dsa(self):
        from sgl_jax.srt.configs.kimi_k3 import apply_kimi_k31_dsa_fields

        for key, value in (("index_topk_freq", 2), ("index_rope_dim", 64)):
            cfg = _k31_text_config()
            cfg.dsa_config = {**cfg.dsa_config, key: value}
            with self.assertRaises(NotImplementedError):
                apply_kimi_k31_dsa_fields(cfg)


def _hadamard_np(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


class TestKimiK31Indexer(unittest.TestCase):
    def test_matches_numpy_reference(self):
        from sgl_jax.srt.models.kimi_k3 import KimiK31DsaIndexer

        mesh = _make_mesh(1)
        rng = np.random.default_rng(1)
        hidden, q_lora, dim, heads, tokens = 48, 24, 32, 4, 7
        with jax.set_mesh(mesh):
            idx = KimiK31DsaIndexer(hidden, q_lora, dim, heads, mesh, dtype=jnp.float32)
            for lin in (idx.wq_b, idx.wk, idx.weights_proj):
                shape = lin.weight.value.shape
                lin.weight.value = jnp.asarray(rng.standard_normal(shape) / np.sqrt(shape[0]), jnp.float32)
            idx.k_norm.weight.value = jnp.asarray(1 + 0.1 * rng.standard_normal(dim), jnp.float32)
            idx.k_norm.bias.value = jnp.asarray(0.1 * rng.standard_normal(dim), jnp.float32)
            x = rng.standard_normal((tokens, hidden)).astype(np.float32)
            q_lat = rng.standard_normal((tokens, q_lora)).astype(np.float32)
            q, k, w = (np.asarray(a) for a in idx(jnp.asarray(x), jnp.asarray(q_lat)))

        h = _hadamard_np(dim)
        ref_q = (q_lat @ np.asarray(idx.wq_b.weight.value)).reshape(tokens, heads, dim) @ h
        k_lin = x @ np.asarray(idx.wk.weight.value)
        k_ln = (k_lin - k_lin.mean(-1, keepdims=True)) / np.sqrt(k_lin.var(-1, keepdims=True) + 1e-6)
        ref_k = (k_ln * np.asarray(idx.k_norm.weight.value) + np.asarray(idx.k_norm.bias.value)) @ h
        ref_w = x @ np.asarray(idx.weights_proj.weight.value)
        np.testing.assert_allclose(q, ref_q, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(k, ref_k, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(w, ref_w, atol=1e-4, rtol=1e-4)
        # The Hadamard rotation leaves the indexer's q.k scores unchanged.
        np.testing.assert_allclose(
            np.einsum("thd,sd->ths", q, k),
            np.einsum("thd,sd->ths", ref_q @ h.T, ref_k @ h.T),
            atol=1e-3,
            rtol=1e-3,
        )


def _dense_backend(seen):
    """Stand-in attention backend: records the DSA kwargs and attends densely."""

    def backend(q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        del forward_batch, token_to_kv_pool
        seen.append({key: kwargs.get(key) for key in ("indexer_type", "q_idx", "k_idx", "idx_weights")})
        rep = partial(jax.sharding.reshard, out_shardings=jax.sharding.PartitionSpec())
        q, k, v = rep(q), rep(k), rep(v)
        q_rope, k_rope = rep(kwargs["q_rope"]), rep(kwargs["k_rope"])
        scores = jnp.einsum("thr,sr->hts", q, k[:, 0]) + jnp.einsum(
            "thd,sd->hts", q_rope, k_rope[:, 0]
        )
        n = q.shape[0]
        causal = jnp.tril(jnp.ones((n, n), dtype=bool))
        probs = jax.nn.softmax(jnp.where(causal, scores * layer.scaling, -jnp.inf), axis=-1)
        out = jnp.einsum("hts,sr->thr", probs, v[:, 0])
        return jax.sharding.reshard(out, jax.sharding.PartitionSpec("data", "tensor", None)), None

    return backend


class TestKimiK31MLAAttention(unittest.TestCase):
    """The DSA call path of Kimi-K3.1's MLA layer against Kimi-K3's absorbed MLA."""

    def _check(self, tp: int):
        if len(jax.devices()) < tp:
            self.skipTest(f"needs {tp} devices (set JAX_NUM_CPU_DEVICES)")
        from sgl_jax.srt.models.kimi_k3 import KimiK31MLAAttention, KimiK3MLAAttention

        cfg = types.SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            q_lora_rank=48,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            mla_use_nope=True,
            mla_use_output_gate=True,
            index_head_dim=32,
            index_n_heads=4,
            use_dsa_sparse=True,
        )
        mesh = _make_mesh(tp)
        rng = np.random.default_rng(0)
        with jax.set_mesh(mesh):
            k3 = KimiK3MLAAttention(cfg, 3, mesh, dtype=jnp.float32, use_absorbed=True)
            k31 = KimiK31MLAAttention(cfg, 3, mesh, dtype=jnp.float32, use_absorbed=True)
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj", "o_proj", "g_proj"):
                param = getattr(k3, name).weight
                value = rng.standard_normal(param.value.shape) / np.sqrt(param.value.shape[0])
                param.value = jax.device_put(jnp.asarray(value, jnp.float32), param.value.sharding)
                getattr(k31, name).weight.value = param.value
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                scale = getattr(k3, name).scale
                value = 1.0 + 0.1 * rng.standard_normal(scale.value.shape)
                scale.value = jax.device_put(jnp.asarray(value, jnp.float32), scale.value.sharding)
                getattr(k31, name).scale.value = scale.value
            # Absorbed weights as the loading hook would build them (any values will do).
            heads = cfg.num_attention_heads
            for name, shape in (
                ("w_uk", (cfg.kv_lora_rank, heads, cfg.qk_nope_head_dim)),
                ("w_uv", (cfg.kv_lora_rank, heads, cfg.v_head_dim)),
            ):
                value = jnp.asarray(rng.standard_normal(shape) / np.sqrt(shape[0]), jnp.float32)
                sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tensor", None))
                placed = jax.device_put(value, sharding)
                getattr(k3, name).value = placed
                getattr(k31, name).value = placed

            hidden = jnp.asarray(rng.standard_normal((11, cfg.hidden_size)), jnp.float32)
            seen_k3, seen_k31 = [], []
            out_k3, _ = k3(None, hidden, types.SimpleNamespace(attn_backend=_dense_backend(seen_k3)), None)
            out_k31, _ = k31(None, hidden, types.SimpleNamespace(attn_backend=_dense_backend(seen_k31)), None)
            out_k3, out_k31 = np.asarray(out_k3), np.asarray(out_k31)

        self.assertIsNone(seen_k3[0]["indexer_type"])
        self.assertEqual(seen_k31[0]["indexer_type"], "full")
        self.assertEqual(seen_k31[0]["q_idx"].shape, (11, cfg.index_n_heads, cfg.index_head_dim))
        self.assertEqual(seen_k31[0]["k_idx"].shape, (11, cfg.index_head_dim))
        self.assertEqual(seen_k31[0]["idx_weights"].shape, (11, cfg.index_n_heads))
        err = np.abs(out_k31 - out_k3).max()
        print(f"[kimi_k3.1 mla tp={tp}] DSA path vs K3 absorbed max|d|={err:.3e}")
        np.testing.assert_allclose(out_k31, out_k3, atol=1e-5, rtol=1e-5)

    def test_dsa_path_matches_k3_single_device(self):
        self._check(tp=1)

    def test_dsa_path_matches_k3_tp2(self):
        self._check(tp=2)


class TestKimiK31WeightMappings(unittest.TestCase):
    def test_indexer_tensors_mapped_on_dsa_layers(self):
        from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
        from sgl_jax.srt.models.kimi_k3 import KimiK31ForConditionalGeneration

        text = _k31_text_config()
        config = KimiK3Config(text_config=text.to_dict(), architectures=["KimiK31ForConditionalGeneration"])
        config.use_dsa_sparse = True
        mesh = _make_mesh(1)
        with jax.set_mesh(mesh):
            model = nnx_abstract(KimiK31ForConditionalGeneration, config, mesh)
            mappings = model._create_weight_mappings("language_model.")
        names = ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias")
        for layer in range(8):
            keys = [f"language_model.model.layers.{layer}.self_attn.indexer.{n}" for n in names]
            if layer in (3, 7):
                self.assertTrue(all(k in mappings for k in keys), [k for k in keys if k not in mappings])
            else:
                self.assertFalse(any(k in mappings for k in keys))


class TestKimiK31IndexerInServerModel(unittest.TestCase):
    """The indexer must work after the server's abstract construction, not only when built eagerly."""

    def test_indexer_runs_after_abstract_build(self):
        from flax import nnx

        from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
        from sgl_jax.srt.models.kimi_k3 import KimiK31ForConditionalGeneration

        text = _k31_text_config()
        config = KimiK3Config(text_config=text.to_dict(), architectures=["KimiK31ForConditionalGeneration"])
        config.use_dsa_sparse = True
        mesh = _make_mesh(1)
        with jax.set_mesh(mesh):
            abstract = nnx_abstract(KimiK31ForConditionalGeneration, config, mesh)
            graphdef, state = nnx.split(abstract)
            state = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype) * 0.01, state)
            model = nnx.merge(graphdef, state)
            indexer = model.model.layers[3].self_attn.indexer
            q, k, w = indexer(jnp.ones((5, 256), jnp.float32), jnp.ones((5, 64), jnp.float32))
        self.assertEqual((q.shape, k.shape, w.shape), ((5, 4, 32), (5, 32), (5, 4)))


def nnx_abstract(cls, config, mesh):
    """Builds the model without allocating parameter values."""
    from flax import nnx

    return nnx.eval_shape(lambda: cls(config, mesh, dtype=jnp.float32))


if __name__ == "__main__":
    unittest.main()
