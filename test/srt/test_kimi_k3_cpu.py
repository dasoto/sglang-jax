"""Kimi-K3 end-to-end CPU numerics test against the HF PyTorch reference.

A tiny K3-shaped checkpoint (8 layers: KDA at 0-indexed layers 0-2 and 4-6,
gated NoPE MLA at 3 and 7, dense SiTU FFN at layer 0 and LatentMoE elsewhere,
AttnRes with block size 4, 16 MXFP4 experts top-4 in a 128-d latent, 2 shared
experts) is produced from Moonshot's ``modeling_kimi_linear.py`` by
``kimi_k3_make_fixture.py`` together with f32 golden logits for prefill and
greedy decode. This test loads that checkpoint through the regular SGLang-JAX
path (``ModelConfig`` -> ``ModelRunner`` -> ``WeightLoader``, MXFP4 experts kept
packed and dequantized in the MoE layer, KDA recurrent-state + MLA KV pools)
and compares logits for a batched prefill of two prompts and teacher-forced
decode steps:

* f32, 1 device and TP=4 x EP=2 (KDA/MLA heads over 4 devices, experts over an
  ``expert=2 x tensor=2`` sub-mesh): top-1 must match everywhere and logits to
  2e-3 (observed ~5e-5).
* bf16 with the Mega-KDA prefill kernel (the TPU path): this random tiny model
  is very sensitive to bf16 rounding (near-tie top-k routing), so the bf16 run
  is held to the HF reference's own bf16-vs-f32 deviation instead.

The end-to-end runs use the native attention backend, i.e. the non-absorbed MLA
form (the MLA v2 Pallas kernel behind ``--attention-backend fa`` does not run
on CPU). ``TestKimiK3AbsorbedMLA`` (no fixture needed) checks that the absorbed
form used on TPU (``w_uk`` / ``w_uv`` built by the loading hook from
``kv_b_proj``, latent MQA, NoPE, sigmoid output gate) matches the non-absorbed
layer, with a dense causal attention standing in for both kernels.

The fixture needs PyTorch + transformers 4.56 (the version the HF code
targets), so it is generated in a separate interpreter. With
``KIMI_K3_HF_REF_DIR`` (directory holding the release's
``configuration_kimi_k3.py`` + ``modeling_kimi_linear.py``) and
``KIMI_K3_REF_PYTHON`` (python of an env with torch, transformers==4.56.2 and
jax[cpu]) set, the test generates it on the fly::

    JAX_NUM_CPU_DEVICES=4 python -m pytest test/srt/test_kimi_k3_cpu.py -v

or pre-generate it with ``kimi_k3_make_fixture.py`` and point
``KIMI_K3_FIXTURE_DIR`` at the output directory. The test is skipped when
neither is available.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PALLAS_INTERPRET", "1")

import subprocess
import sys
import tempfile
import types
import unittest
from functools import partial
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_CACHE: dict[str, str] = {}


def _fixture_dir() -> str | None:
    if "dir" in _FIXTURE_CACHE:
        return _FIXTURE_CACHE["dir"]
    path = os.environ.get("KIMI_K3_FIXTURE_DIR")
    if path and os.path.exists(os.path.join(path, "golden.npz")):
        _FIXTURE_CACHE["dir"] = path
        return path
    ref_dir = os.environ.get("KIMI_K3_HF_REF_DIR")
    if not ref_dir:
        return None
    path = path or tempfile.mkdtemp(prefix="kimi_k3_tiny_")
    subprocess.check_call(
        [
            os.environ.get("KIMI_K3_REF_PYTHON", sys.executable),
            os.path.join(_HERE, "kimi_k3_make_fixture.py"),
            "--ref-dir",
            ref_dir,
            "--out-dir",
            path,
        ]
    )
    _FIXTURE_CACHE["dir"] = path
    return path


def _case_env(dtype: str) -> dict[str, str]:
    if dtype == "float32":
        # Mega KDA is BF16-only: exercise the chunked prefill kernel (same math)
        # and keep the conv history exact.
        return {
            "SGLANG_JAX_KDA_PREFILL_KERNEL": "chunked",
            "SGLANG_JAX_CONV_STATE_DTYPE": "float32",
        }
    # TPU defaults: Mega KDA prefill, bf16 conv state.
    return {"SGLANG_JAX_KDA_PREFILL_KERNEL": "mega", "SGLANG_JAX_CONV_STATE_DTYPE": "bfloat16"}


def _make_mesh(tp: int):
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    return create_device_mesh(
        ici_parallelism=[1, tp], dcn_parallelism=[1, 1], devices=jax.devices()[:tp]
    )


def _make_runner(
    fixture: str, *, dtype: str, tp: int, ep: int, trust_remote_code: bool, **extra_args
):
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=fixture,
        trust_remote_code=trust_remote_code,
        skip_tokenizer_init=True,
        device="cpu",
        dtype=dtype,
        attention_backend="native",
        tp_size=tp,
        ep_size=ep,
        moe_backend="epmoe",
        page_size=1,
        max_running_requests=4,
        max_total_tokens=512,
        chunked_prefill_size=256,
        mem_fraction_static=0.3,
        disable_radix_cache=True,
        disable_precompile=True,
        enable_topk_kernel=False,
        random_seed=0,
        **extra_args,
    )
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        tp_size=tp,
        dp_size=1,
        server_args=server_args,
        mesh=_make_mesh(tp),
    )
    # Normally sized by the TP worker from the compilation buckets.
    runner.req_to_token_pool.init_cache_loc_host_buffer(4096)
    return runner


def _forward(runner, batch, num_tokens: int):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    page_size = runner.page_size
    bs = len(batch.seq_lens)
    cache_loc = int(np.sum((np.asarray(batch.seq_lens) + page_size - 1) // page_size * page_size))
    worker_batch = batch.get_model_worker_batch([num_tokens], [bs], [cache_loc], page_size, False)
    runner.attn_backend.forward_metadata = runner.attn_backend.get_forward_metadata(worker_batch)
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    logits_metadata = LogitsMetadata.from_model_worker_batch(worker_batch, mesh=runner.mesh)
    logits_output = runner.forward(forward_batch, logits_metadata=logits_metadata)[0]
    logits = np.asarray(jax.device_get(logits_output.next_token_logits), dtype=np.float32)
    return logits[: worker_batch.real_bs, : runner.model_config.vocab_size]


def _run_sglang(runner, golden):
    """Batched prefill of all prompts, then teacher-forced decode steps."""
    from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
    from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    lens = golden["prompt_lens"].tolist()
    flat = golden["prompts"].tolist()
    prompts = [flat[sum(lens[:i]) : sum(lens[: i + 1])] for i in range(len(lens))]
    reqs = []
    for i, ids in enumerate(prompts):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    batch = ScheduleBatch.init_new(
        reqs=[reqs],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=ChunkCache(
            runner.req_to_token_pool, runner.token_to_kv_pool_allocator, runner.page_size
        ),
        model_config=runner.model_config,
        enable_overlap=False,
        dp_size=1,
        enable_custom_logit_processor=False,
        chunked_reqs=None,
        mesh=runner.mesh,
    )
    batch.prepare_for_extend()
    prefill = _forward(runner, batch, int(sum(lens)))
    decode = []
    for step in range(golden["decode_tokens"].shape[1]):
        batch.output_ids = np.asarray(golden["decode_tokens"][:, step], dtype=np.int32)
        batch.prepare_for_decode()
        decode.append(_forward(runner, batch, len(reqs)))
    return prefill, np.stack(decode, axis=1)


class TestKimiK3CPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _fixture_dir()
        if cls.fixture is None:
            raise unittest.SkipTest("set KIMI_K3_FIXTURE_DIR or KIMI_K3_HF_REF_DIR")
        with np.load(os.path.join(cls.fixture, "golden.npz")) as data:
            cls.golden = {k: data[k] for k in data.files}

    def _run(self, *, dtype, tp, ep, trust_remote_code, env=None, **extra_args):
        if len(jax.devices()) < tp:
            self.skipTest(f"needs {tp} devices (set JAX_NUM_CPU_DEVICES)")
        with mock.patch.dict(os.environ, {**_case_env(dtype), **(env or {})}):
            runner = _make_runner(
                self.fixture,
                dtype=dtype,
                tp=tp,
                ep=ep,
                trust_remote_code=trust_remote_code,
                **extra_args,
            )
            return _run_sglang(runner, self.golden)

    def _check_exact(self, prefill, decode, label, atol=2e-3, rtol=2e-3):
        golden = self.golden
        ref_prefill, ref_decode = golden["prefill_logits"], golden["decode_logits"]
        self.assertEqual(prefill.shape, ref_prefill.shape)
        self.assertEqual(decode.shape, ref_decode.shape)
        # Top-1 agreement (greedy tokens) for prefill and every decode step.
        np.testing.assert_array_equal(prefill.argmax(-1), ref_prefill.argmax(-1))
        np.testing.assert_array_equal(decode.argmax(-1), ref_decode.argmax(-1))
        np.testing.assert_allclose(prefill, ref_prefill, atol=atol, rtol=rtol)
        np.testing.assert_allclose(decode, ref_decode, atol=atol, rtol=rtol)
        print(
            f"[kimi_k3 {label}] max|dlogit| prefill={np.abs(prefill - ref_prefill).max():.3e} "
            f"decode={np.abs(decode - ref_decode).max():.3e}"
        )

    def test_f32_single_device(self):
        prefill, decode = self._run(dtype="float32", tp=1, ep=1, trust_remote_code=False)
        self._check_exact(prefill, decode, "f32 tp=1 ep=1 mla=mha")

    def test_f32_tp4_ep2(self):
        # KDA/MLA heads over tensor=4; experts over expert=2 x expert-TP 2.
        prefill, decode = self._run(dtype="float32", tp=4, ep=2, trust_remote_code=True)
        self._check_exact(prefill, decode, "f32 tp=4 ep=2 mla=mha")

    def test_f32_tp4_ep2_eplb_permuted_placement(self):
        # EPLB: every MoE layer stores its experts in a random physical order (the loader
        # permutes the stacked MXFP4 weights) and the router maps its logical top-k ids
        # through the same placement, so the logits must not change. The recorder mode
        # (SGLANG_JAX_EPLB_RECORD=1: replicated top-k ids, -1 on padding) rides along.
        import json

        from sgl_jax.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
            set_global_expert_location_metadata,
        )

        with open(os.path.join(self.fixture, "config.json")) as f:
            cfg = json.load(f)
        text = cfg.get("text_config", cfg)
        rng = np.random.default_rng(1234)
        phy2log = [
            rng.permutation(text["num_experts"]).tolist()
            for _ in range(text["num_hidden_layers"])
        ]
        try:
            prefill, decode = self._run(
                dtype="float32",
                tp=4,
                ep=2,
                trust_remote_code=True,
                env={"SGLANG_JAX_EPLB_RECORD": "1"},
                ep_dispatch_algorithm="static",
                init_expert_location=json.dumps({"physical_to_logical_map": phy2log}),
            )
            metadata = get_global_expert_location_metadata()
            self.assertIsNotNone(metadata)
            np.testing.assert_array_equal(
                np.asarray(jax.device_get(metadata.physical_to_logical_map)), np.asarray(phy2log)
            )
        finally:
            set_global_expert_location_metadata(None)
        self._check_exact(prefill, decode, "f32 tp=4 ep=2 eplb permuted")

    def test_bf16_mega_kda_noise_parity(self):
        prefill, decode = self._run(dtype="bfloat16", tp=1, ep=1, trust_remote_code=False)
        golden = self.golden
        ours = np.concatenate([prefill, decode.reshape(-1, decode.shape[-1])])
        ref = np.concatenate(
            [golden["prefill_logits"], golden["decode_logits"].reshape(-1, decode.shape[-1])]
        )
        hf_bf16 = np.concatenate(
            [
                golden["prefill_logits_hf_bf16"],
                golden["decode_logits_hf_bf16"].reshape(-1, decode.shape[-1]),
            ]
        )
        self.assertTrue(np.all(np.isfinite(ours)))
        err_ours = float(np.abs(ours - ref).mean())
        err_hf = float(np.abs(hf_bf16 - ref).mean())
        print(
            f"[kimi_k3 bf16 mega-kda] mean|dlogit| vs f32 golden: sglang={err_ours:.3f} hf_bf16={err_hf:.3f}"
        )
        self.assertLess(err_ours, 1.5 * err_hf + 0.02)


def _dense_causal_attention(q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
    """One-sequence causal attention standing in for the FA / MLA kernels.

    Non-absorbed MHA: per-head ``q``, ``k``, ``v``. Absorbed MQA (``q_rope`` /
    ``k_rope`` given): ``q`` is ``q_nope @ w_uk`` and ``k = v = c_kv`` is the
    single latent head; scores add the (unrotated) rope parts.
    """
    del forward_batch, token_to_kv_pool
    replicate = partial(jax.sharding.reshard, out_shardings=jax.sharding.PartitionSpec())
    q, k, v = replicate(q), replicate(k), replicate(v)
    q_rope, k_rope = kwargs.get("q_rope"), kwargs.get("k_rope")
    if q_rope is not None:
        q_rope, k_rope = replicate(q_rope), replicate(k_rope)
    if q_rope is None:
        scores = jnp.einsum("thd,shd->hts", q, k)
    else:
        scores = jnp.einsum("thr,sr->hts", q, k[:, 0])
        scores = scores + jnp.einsum("thd,sd->hts", q_rope, k_rope[:, 0])
    num_tokens = q.shape[0]
    causal = jnp.tril(jnp.ones((num_tokens, num_tokens), dtype=bool))
    probs = jax.nn.softmax(jnp.where(causal, scores * layer.scaling, -jnp.inf), axis=-1)
    if q_rope is None:
        out = jnp.einsum("hts,shd->thd", probs, v)
    else:
        out = jnp.einsum("hts,sr->thr", probs, v[:, 0])
    # Head-sharded like the kernels' outputs (e.g. MLA backend P(dp, "tensor", None)).
    return jax.sharding.reshard(out, jax.sharding.PartitionSpec("data", "tensor", None)), None


class TestKimiK3AbsorbedMLA(unittest.TestCase):
    """Absorbed (TPU ``fa``) vs non-absorbed gated NoPE MLA, same weights, f32."""

    def _check(self, tp: int):
        if len(jax.devices()) < tp:
            self.skipTest(f"needs {tp} devices (set JAX_NUM_CPU_DEVICES)")
        from sgl_jax.srt.model_loader.weights import WeightSpec
        from sgl_jax.srt.models.kimi_k3 import KimiK3MLAAttention

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
        )
        mesh = _make_mesh(tp)
        rng = np.random.default_rng(0)
        with jax.set_mesh(mesh):
            mha = KimiK3MLAAttention(cfg, 0, mesh, dtype=jnp.float32, use_absorbed=False)
            absorbed = KimiK3MLAAttention(cfg, 0, mesh, dtype=jnp.float32, use_absorbed=True)
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj", "g_proj"):
                param = getattr(mha, name).weight
                fan_in = param.value.shape[0]
                value = rng.standard_normal(param.value.shape, dtype=np.float32) / np.sqrt(fan_in)
                param.value = jax.device_put(value, param.value.sharding)
                if name != "kv_b_proj":
                    getattr(absorbed, name).weight.value = param.value
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                scale = getattr(mha, name).scale
                value = 1.0 + 0.1 * rng.standard_normal(scale.value.shape, dtype=np.float32)
                scale.value = jax.device_put(value, scale.value.sharding)
                getattr(absorbed, name).scale.value = scale.value

            # The loading hook replaces kv_b_proj by the w_uk / w_uv group;
            # feed it the HF-layout [out, in] kv_b_proj weight.
            mappings = {
                "hf.kv_b_proj.weight": WeightSpec(
                    target_path="attn.kv_b_proj.weight", sharding=(None, "tensor"), transpose=True
                )
            }
            loader = types.SimpleNamespace(
                model_config=types.SimpleNamespace(quantization_config=None)
            )
            mappings = absorbed.prepare_weight_loading(loader, mappings, "attn")
            spec = mappings["hf.kv_b_proj.weight"]
            self.assertEqual(spec.target_path, ["attn.w_uk", "attn.w_uv"])
            self.assertIsNone(absorbed.kv_b_proj)
            kv_b_hf = np.asarray(mha.kv_b_proj.weight.value).T
            absorbed.w_uk.value, absorbed.w_uv.value = spec.recipe([kv_b_hf])

            hidden = jnp.asarray(rng.standard_normal((11, cfg.hidden_size), dtype=np.float32))
            forward_batch = types.SimpleNamespace(attn_backend=_dense_causal_attention)
            out_mha, _ = mha(None, hidden, forward_batch, None)
            out_absorbed, _ = absorbed(None, hidden, forward_batch, None)
            out_mha, out_absorbed = np.asarray(out_mha), np.asarray(out_absorbed)
        err = np.abs(out_absorbed - out_mha).max()
        print(
            f"[kimi_k3 mla tp={tp}] absorbed vs mha max|d|={err:.3e} (max|out|={np.abs(out_mha).max():.2f})"
        )
        np.testing.assert_allclose(out_absorbed, out_mha, atol=1e-4, rtol=1e-4)

    def test_absorbed_matches_mha_single_device(self):
        self._check(tp=1)

    def test_absorbed_matches_mha_tp2(self):
        self._check(tp=2)


if __name__ == "__main__":
    unittest.main()
