"""Kimi-K3 configuration helpers.

Kimi-K3 ships as a multimodal checkpoint: the top-level config is
``KimiK3Config`` (``model_type: kimi_k3``, architecture
``KimiK3ForConditionalGeneration``) and the text model lives in
``text_config``, whose ``model_type`` is ``kimi_linear``. That nested model type
collides with the existing Kimi-Linear detection, so K3 is identified by the
top-level architecture and by the K3-only text fields (Attention Residuals and
LatentMoE), never by ``model_type`` alone.

The local ``KimiK3Config`` lets the checkpoint load without
``trust_remote_code`` (the Moonshot ``configuration_kimi_k3.py`` is equivalent
for the fields we read). With ``trust_remote_code`` the remote class is used;
``get_kimi_k3_text_config`` normalizes either form to SGLang-JAX's
``KimiLinearConfig``.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig

# Kimi-K3.1 is Kimi-K3 plus a DSA lightning indexer on its MLA layers (``dsa_config``).
KIMI_K3_ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration")

# Text-config fields that only Kimi-K3 (not Kimi-Linear) sets.
_K3_TEXT_FIELDS = ("attn_res_block_size", "routed_expert_hidden_size")


class KimiK3Config(PretrainedConfig):
    """Minimal local mirror of Moonshot's ``KimiK3Config`` (text + raw vision)."""

    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        self.text_config = text_config
        # Vision is not served yet (Phase 3); keep the raw dict for reference.
        if isinstance(vision_config, PretrainedConfig):
            vision_config = vision_config.to_dict()
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config
        super().__init__(pad_token_id=pad_token_id, **kwargs)


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


def is_kimi_k3_config(hf_config: Any) -> bool:
    """True for a Kimi-K3 top-level config (or its text sub-config)."""
    if hf_config is None:
        return False
    architectures = getattr(hf_config, "architectures", None) or []
    if any(arch in KIMI_K3_ARCHITECTURES for arch in architectures):
        return True
    if getattr(hf_config, "model_type", None) in ("kimi_k3", "kimi_k31"):
        return True
    text = _text_config(hf_config)
    return getattr(text, "linear_attn_config", None) is not None and all(
        getattr(text, field, None) is not None for field in _K3_TEXT_FIELDS
    )


def get_kimi_k3_text_config(hf_config: Any) -> KimiLinearConfig | None:
    """Return the K3 text config as SGLang-JAX's ``KimiLinearConfig`` (else None).

    Accepts the local or remote (``trust_remote_code``) top-level config, or a
    text config directly. The returned object exposes ``is_kda_layer``,
    ``linear_layer_ids`` and ``full_attention_layer_ids`` (1-indexed
    ``kda_layers`` in the checkpoint are converted by ``is_kda_layer``).
    """
    if not is_kimi_k3_config(hf_config):
        return None
    text = _text_config(hf_config)
    if getattr(text, "linear_attn_config", None) is None:
        return None
    if isinstance(text, KimiLinearConfig):
        apply_kimi_k31_dsa_fields(text)
        return text
    kwargs = text.to_dict() if hasattr(text, "to_dict") else dict(vars(text))
    kwargs.pop("transformers_version", None)
    config = KimiLinearConfig(**kwargs)
    apply_kimi_k31_dsa_fields(config)
    return config


def apply_kimi_k31_dsa_fields(text: Any) -> bool:
    """Flatten Kimi-K3.1's nested ``dsa_config`` onto its text config, in place.

    Kimi-K3.1 configures its DSA lightning indexer under ``text_config.dsa_config``
    (1-indexed ``dsa_layers``, ``index_num_heads``, ...). The DSA backend and the KV
    pools read DeepSeek-V3.2-style flat fields instead: ``index_topk``,
    ``index_head_dim``, ``index_n_heads`` and ``indexer_types``, one entry per layer:
    ``"full"`` where the layer owns an indexer, ``"none"`` on the KDA layers, which
    have no indexer and never reach the DSA backend. Idempotent; returns False when
    the config has no ``dsa_config`` (Kimi-K3).
    """
    dsa = getattr(text, "dsa_config", None)
    if not dsa:
        return False
    if not isinstance(dsa, dict):
        dsa = dsa.to_dict() if hasattr(dsa, "to_dict") else dict(vars(dsa))
    num_layers = int(text.num_hidden_layers)
    layers = sorted(int(layer) - 1 for layer in dsa["dsa_layers"])  # 1-indexed in the checkpoint
    if not layers or layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"Kimi-K3.1 dsa_layers out of range: {dsa['dsa_layers']}")
    # Every DSA layer computes its own top-k (no IndexShare); its keys need no RoPE and a
    # LayerNorm. Refuse a checkpoint that asks for anything else.
    if int(dsa.get("index_topk_freq", 1) or 1) != 1:
        raise NotImplementedError("Kimi-K3.1 with index_topk_freq != 1 is not supported")
    if int(dsa.get("index_rope_dim", 0) or 0) != 0:
        raise NotImplementedError("Kimi-K3.1 indexer RoPE (index_rope_dim > 0) is not supported")
    if dsa.get("index_k_norm_type", "layernorm") not in ("layernorm", "layer"):
        raise NotImplementedError(f"Kimi-K3.1 index_k_norm_type={dsa.get('index_k_norm_type')!r}")
    owned = set(layers)
    text.index_topk = int(dsa["index_topk"])
    text.index_head_dim = int(dsa["index_head_dim"])
    text.index_n_heads = int(dsa.get("index_num_heads", dsa.get("index_n_heads")))
    text.index_rope_dim = 0
    text.index_skip_topk_offset = 0
    text.indexer_types = ["full" if i in owned else "none" for i in range(num_layers)]
    text.dsa_layer_ids = layers
    return True
