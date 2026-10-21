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

KIMI_K3_ARCHITECTURES = ("KimiK3ForConditionalGeneration",)

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
    if getattr(hf_config, "model_type", None) == "kimi_k3":
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
        return text
    kwargs = text.to_dict() if hasattr(text, "to_dict") else dict(vars(text))
    kwargs.pop("transformers_version", None)
    return KimiLinearConfig(**kwargs)
