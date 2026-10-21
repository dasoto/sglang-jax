"""Build a tiny Kimi-K3-shaped checkpoint plus golden logits from the HF reference.

Runs with PyTorch + transformers 4.56 (the version the Moonshot modeling code
targets); SGLang-JAX itself is not imported (only its MXFP4 NumPy helpers,
loaded by file path). Example (``--ref-dir`` holds the release's
``configuration_kimi_k3.py`` and ``modeling_kimi_linear.py``)::

    python test/srt/kimi_k3_make_fixture.py --ref-dir /path/to/Kimi-K3 --out-dir /tmp/kimi_k3_tiny

The output directory is an HF-layout multimodal K3 checkpoint
(``KimiK3ForConditionalGeneration`` config with a nested ``kimi_linear``
text config and the real ``mxfp4-pack-quantized`` quantization config) whose
routed experts are stored exactly like the release (``experts.E.w{1,2,3}.
weight_packed`` uint8 + ``weight_scale`` E8M0 uint8), A_log zero-padded to
``head_dim``, f32 KDA conv / A_log / dt_bias / o_norm / router bias and bf16
everything else, plus ``golden.npz`` with f32 reference logits for prefill and
greedy decode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kimi_k3_hf_reference import load_hf_reference  # noqa: E402

_MXFP4_PATH = os.path.join(
    _HERE, "..", "..", "python", "sgl_jax", "srt", "layers", "quantization", "mxfp4.py"
)

# 0-indexed MLA layers 3 and 7, KDA elsewhere (config lists are 1-indexed).
TINY_TEXT_CONFIG = dict(
    vocab_size=1024,
    hidden_size=256,
    intermediate_size=384,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=4,
    rms_norm_eps=1e-5,
    hidden_act="situ",
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    attn_res_block_size=4,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    mla_use_nope=True,
    mla_use_output_gate=True,
    num_experts=16,
    num_experts_per_token=4,
    moe_intermediate_size=64,
    routed_expert_hidden_size=128,
    latent_moe_use_norm=True,
    num_shared_experts=2,
    moe_router_activation_func="sigmoid",
    moe_renormalize=True,
    moe_layer_freq=1,
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    num_expert_group=1,
    topk_group=1,
    topk_method="noaux_tc",
    use_grouped_topk=True,
    num_nextn_predict_layers=0,
    linear_attn_config=dict(
        full_attn_layers=[4, 8],
        kda_layers=[1, 2, 3, 5, 6, 7],
        num_heads=4,
        head_dim=64,
        short_conv_kernel_size=4,
        gate_lower_bound=-5.0,
        use_full_rank_gate=True,
    ),
    tie_word_embeddings=False,
    max_position_embeddings=4096,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

# Same quantization block as the release text_config.
QUANTIZATION_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.uint8",
                "strategy": "group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        }
    },
    "format": "mxfp4-pack-quantized",
    "global_compression_ratio": None,
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}

# Tensors the release stores in float32 (everything else is bf16 or MXFP4).
_F32_SUFFIXES = ("A_log", "dt_bias", "conv1d.weight", "o_norm.weight", "e_score_correction_bias")


def _load_mxfp4_numpy():
    spec = importlib.util.spec_from_file_location("_k3_mxfp4", _MXFP4_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _randomize(model: torch.nn.Module, gen: torch.Generator) -> None:
    """Variance-preserving random init with non-trivial norms / gates / biases."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                param.uniform_(-0.75, 2.5, generator=gen)  # release range ~[-0.75, 2.47]
            elif name.endswith("dt_bias"):
                param.normal_(0.0, 1.0, generator=gen)
            elif name.endswith("e_score_correction_bias"):
                param.normal_(0.0, 0.1, generator=gen)
            elif name.endswith("_res_proj.weight"):
                param.normal_(0.0, 0.15, generator=gen)
            elif "conv1d" in name:
                param.normal_(0.0, 0.4, generator=gen)
            elif "norm" in name and param.dim() == 1:
                param.normal_(1.0, 0.1, generator=gen)
            elif "embed_tokens" in name:
                param.normal_(0.0, 1.0, generator=gen)
            elif param.dim() == 2:
                param.normal_(0.0, param.shape[1] ** -0.5, generator=gen)
            else:
                raise ValueError(f"unhandled parameter {name} {tuple(param.shape)}")
            if not name.endswith(_F32_SUFFIXES):
                param.copy_(param.to(torch.bfloat16).float())  # bf16-exact like the release


def _quantize_experts(model, cfg, mxfp4) -> dict[str, np.ndarray]:
    """MXFP4-quantize every routed expert in place; return the packed tensors."""
    packed = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            for expert_idx, expert in enumerate(moe.experts):
                for proj in ("w1", "w2", "w3"):
                    linear = getattr(expert, proj)
                    weight = linear.weight.detach().float().numpy()
                    weight_packed, weight_scale = mxfp4.quantize_mxfp4_numpy(weight)
                    linear.weight.copy_(
                        torch.from_numpy(mxfp4.dequant_mxfp4_numpy(weight_packed, weight_scale))
                    )
                    key = (
                        f"language_model.model.layers.{layer_idx}.block_sparse_moe."
                        f"experts.{expert_idx}.{proj}"
                    )
                    packed[f"{key}.weight_packed"] = weight_packed
                    packed[f"{key}.weight_scale"] = weight_scale
    return packed


def _checkpoint_tensors(model, cfg, packed) -> dict[str, torch.Tensor]:
    tensors = {}
    head_dim = cfg.linear_attn_config["head_dim"]
    for name, param in model.state_dict().items():
        if ".block_sparse_moe.experts." in name:
            continue  # stored as MXFP4 below
        value = param.detach().float()
        if name.endswith("A_log"):
            value = torch.nn.functional.pad(value.reshape(-1), (0, head_dim - value.numel()))
        dtype = torch.float32 if name.endswith(_F32_SUFFIXES) else torch.bfloat16
        tensors[f"language_model.{name}"] = value.to(dtype).contiguous()
    for name, array in packed.items():
        tensors[name] = torch.from_numpy(np.ascontiguousarray(array))
    # Vision weights must be ignored by the text model.
    tensors["vision_tower.patch_embed.proj.weight"] = torch.zeros(
        8, 3, 14, 14, dtype=torch.bfloat16
    )
    tensors["mm_projector.proj.0.weight"] = torch.zeros(16, 16, dtype=torch.bfloat16)
    return tensors


def _config_json(text_config: dict) -> dict:
    text = dict(text_config)
    text.update(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        quantization_config=QUANTIZATION_CONFIG,
        dtype="bfloat16",
    )
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_kimi_k3.KimiK3Config",
            "AutoModel": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "dtype": "bfloat16",
        "media_placeholder_token_id": 1000,
        "model_type": "kimi_k3",
        "text_config": text,
        "tie_word_embeddings": False,
        "vision_config": {"vt_hidden_size": 16, "vt_num_hidden_layers": 1, "patch_size": 14},
    }


@torch.no_grad()
def _golden(model, ref, cfg, prompts, decode_steps):
    prefill, tokens, decode = [], [], []
    for ids in prompts:
        cache = ref.modeling.KimiDynamicCache(cfg)
        out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1].float()
        prefill.append(logits.numpy())
        seq = list(ids)
        step_tokens, step_logits = [], []
        for _ in range(decode_steps):
            token = int(logits.argmax())
            step_tokens.append(token)
            seq.append(token)
            out = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            logits = out.logits[0, -1].float()
            step_logits.append(logits.numpy())
        # Self-check: cached decode == full recompute of the extended sequence.
        full = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1].float()
        assert torch.allclose(full, logits, atol=1e-4), (full - logits).abs().max()
        tokens.append(step_tokens)
        decode.append(np.stack(step_logits))
    return np.stack(prefill), np.array(tokens, dtype=np.int32), np.stack(decode)


@torch.no_grad()
def _teacher_forced(model, ref, cfg, prompts, decode_tokens):
    prefill, decode = [], []
    for ids, step_tokens in zip(prompts, decode_tokens):
        cache = ref.modeling.KimiDynamicCache(cfg)
        out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
        prefill.append(out.logits[0, -1].float().numpy())
        steps = []
        for token in step_tokens:
            out = model(
                input_ids=torch.tensor([[int(token)]]), past_key_values=cache, use_cache=True
            )
            steps.append(out.logits[0, -1].float().numpy())
        decode.append(np.stack(steps))
    return np.stack(prefill), np.stack(decode)


def build_fixture(ref_dir: str, out_dir: str, seed: int = 0, decode_steps: int = 4) -> None:
    from safetensors.torch import save_file

    ref = load_hf_reference(ref_dir)
    mxfp4 = _load_mxfp4_numpy()
    cfg = ref.config.KimiLinearConfig(**TINY_TEXT_CONFIG)
    torch.manual_seed(seed)
    model = ref.modeling.KimiLinearForCausalLM(cfg).float().eval()
    model.config._attn_implementation = "eager"  # flash-attn is GPU-only
    assert all(
        layer.self_attn.config._attn_implementation == "eager" for layer in model.model.layers
    )
    gen = torch.Generator().manual_seed(seed)
    _randomize(model, gen)
    packed = _quantize_experts(model, cfg, mxfp4)

    rng = np.random.default_rng(seed)
    prompts = [rng.integers(3, 1000, size=n).tolist() for n in (9, 14)]
    prefill_logits, decode_tokens, decode_logits = _golden(model, ref, cfg, prompts, decode_steps)
    checkpoint = _checkpoint_tensors(model, cfg, packed)

    # The same model run in bf16 (f32 params stay f32, like the release): its
    # deviation from the f32 golden is the intrinsic bf16 noise level that the
    # SGLang-JAX bf16 run is compared against (this random tiny model is very
    # sensitive: near-tie top-k routing flips under bf16 rounding).
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(_F32_SUFFIXES):
                param.data = param.data.to(torch.bfloat16)
    prefill_bf16, decode_bf16 = _teacher_forced(model, ref, cfg, prompts, decode_tokens)

    os.makedirs(out_dir, exist_ok=True)
    save_file(
        checkpoint,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(_config_json(TINY_TEXT_CONFIG), f, indent=2)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2, "max_length": 4096}, f)
    shutil.copy(os.path.join(ref_dir, "configuration_kimi_k3.py"), out_dir)
    np.savez(
        os.path.join(out_dir, "golden.npz"),
        prompt_lens=np.array([len(p) for p in prompts], dtype=np.int32),
        prompts=np.array(sum(prompts, []), dtype=np.int32),
        prefill_logits=prefill_logits,
        decode_tokens=decode_tokens,
        decode_logits=decode_logits,
        prefill_logits_hf_bf16=prefill_bf16,
        decode_logits_hf_bf16=decode_bf16,
    )
    print(f"wrote {out_dir}: prompts={[len(p) for p in prompts]} decode_steps={decode_steps}")
    print("reference greedy tokens:", decode_tokens.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=4)
    args = parser.parse_args()
    build_fixture(args.ref_dir, args.out_dir, args.seed, args.decode_steps)


if __name__ == "__main__":
    main()
