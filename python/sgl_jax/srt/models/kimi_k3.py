"""Kimi-K3 text model (``KimiK3ForConditionalGeneration``) for SGLang-JAX.

Kimi-K3 is a 93-layer hybrid of Kimi Delta Attention (KDA, 69 layers) and gated
NoPE MLA (24 layers) with LatentMoE FFNs and Attention Residuals (AttnRes). This
file implements the text model only; ``vision_tower.*`` / ``mm_projector.*``
checkpoint tensors are ignored.

Semantics follow Moonshot's ``modeling_kimi_linear.py`` (Kimi-K3 release):

* KDA: q/k/v projections + depthwise causal short conv (SiLU, f32 weights),
  forget gate ``g = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))``,
  ``beta = sigmoid(b_proj(x))``, L2-normalized q/k (q scaled by ``head_dim**-0.5``),
  decay -> delta update -> read, then ``RMSNormGated(o, g_proj(x))`` (sigmoid
  gate, full-rank ``g_proj``) and ``o_proj``. ``A_log`` is stored zero-padded to
  ``head_dim`` entries in the checkpoint; the first ``num_heads`` are per-head.
* Gated MLA: DeepSeek MLA with ``q_lora``/``kv_lora`` (their RMSNorms use the
  HF default eps 1e-6), the 64 "rope" dims are never rotated, softmax scale
  ``(qk_nope + qk_rope) ** -0.5``, output ``o_proj(attn * sigmoid(g_proj(x)))``.
* LatentMoE: fp32 sigmoid router on the full hidden state with a selection-only
  ``e_score_correction_bias``; top-k weights are the renormalized *unbiased*
  scores times ``routed_scaling_factor``. Tokens are projected to the latent
  (``routed_expert_down_proj``), routed through ``EPMoE`` (MXFP4 or bf16
  experts, SiTU-GLU), combined, RMS-normalized (``routed_expert_norm``) and
  projected back (``routed_expert_up_proj``); shared experts (one SiTU MLP of
  width ``moe_intermediate_size * num_shared_experts``) run on the same
  normalized hidden state and are added.
* AttnRes: per token, the attention and FFN sub-layer inputs (and the final
  output) are a softmax mix over the block stack (one entry per
  ``attn_res_block_size`` layers, the first being the embedding) plus the
  running partial sum, scored by ``<rmsnorm(v), norm_w * proj_w>`` in f32.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.kimi_k3 import get_kimi_k3_text_config
from sgl_jax.srt.configs.kimi_linear import KimiLinearConfig
from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.activation import situ_and_mul
from sgl_jax.srt.layers.attention.fla.gated_rmsnorm import GatedRMSNorm
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, create_moe_weights_mapping
from sgl_jax.srt.layers.moe_mxfp4 import MXFP4EPMoE, create_mxfp4_moe_weights_mapping
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention

logger = logging.getLogger(__name__)

# KimiRMSNorm default eps, used by the MLA q_a / kv_a layernorms in the HF code.
_MLA_LORA_NORM_EPS = 1e-6


def _situ_params(config) -> tuple[float, float | None]:
    beta = getattr(config, "activation_situ_beta", None) or 1.0
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return float(beta), (float(linear_beta) if linear_beta is not None else None)


def _is_mxfp4_quant(quantization_config) -> bool:
    if quantization_config is None:
        return False
    if isinstance(quantization_config, dict):
        return quantization_config.get("format") == "mxfp4-pack-quantized"
    return bool(getattr(quantization_config, "is_mxfp4_moe", lambda: False)())


class KimiK3MLP(nnx.Module):
    """SiTU-GLU MLP: dense layer-0 FFN and the (fused) shared experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        situ_beta: float,
        situ_linear_beta: float | None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.gate_proj = LinearBase(
            hidden_size,
            intermediate_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            hidden_size,
            intermediate_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            intermediate_size,
            hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="down_proj",
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        output, _ = self.down_proj(self.activation(hidden_states))
        return output

    def activation(self, hidden_states: jax.Array) -> jax.Array:
        """SiTU-GLU of the gate/up projections: the down projection's input."""
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        return situ_and_mul(gate, up, beta=self.situ_beta, linear_beta=self.situ_linear_beta)


def _row_parallel_sum(mesh, x1, w1, x2, w2) -> jax.Array:
    """``x1 @ w1 + x2 @ w2`` for two row-parallel projections, with one all-reduce.

    Inputs are P("data", "tensor"), weights P("tensor", None), output P("data", None).
    Separate row-parallel linears all-reduce each partial product over "tensor"; here
    both local products are summed in f32 first, so the sum costs one all-reduce.
    """

    def local(a, b, c, d):
        y = jnp.dot(a, b, preferred_element_type=jnp.float32)
        y = y + jnp.dot(c, d, preferred_element_type=jnp.float32)
        return jax.lax.psum(y.astype(a.dtype), "tensor")

    spec_x, spec_w = P("data", "tensor"), P("tensor", None)
    x1 = jax.sharding.reshard(x1, NamedSharding(mesh, spec_x))
    x2 = jax.sharding.reshard(x2, NamedSharding(mesh, spec_x))
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(spec_x, spec_w, spec_x, spec_w),
        out_specs=P("data", None),
        check_vma=False,
    )(x1, w1, x2, w2)


class AttnResMix(nnx.Module):
    """Attention-Residual mix (HF ``_apply_attn_res``), strictly per token.

    ``v = [blocks..., partial]``; ``score_j = <v_j / rms(v_j), norm * proj>``;
    ``out = sum_j softmax(score)_j * v_j``, all in f32, cast to ``v.dtype``.
    """

    def __init__(self, hidden_size: int, epsilon: float):
        self.epsilon = epsilon
        self.norm = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32, out_sharding=P(None)))
        self.proj = nnx.Param(jnp.zeros((hidden_size,), dtype=jnp.float32, out_sharding=P(None)))

    def __call__(self, blocks: list[jax.Array], partial_sum: jax.Array) -> jax.Array:
        sources = [*blocks, partial_sum]
        out_dtype = partial_sum.dtype
        w = self.norm.value.astype(jnp.float32) * self.proj.value.astype(jnp.float32)
        scores = []
        for v in sources:
            v32 = v.astype(jnp.float32)
            rstd = jax.lax.rsqrt(jnp.mean(jnp.square(v32), axis=-1, keepdims=True) + self.epsilon)
            scores.append(jnp.sum(v32 * rstd * w, axis=-1))
        probs = jax.nn.softmax(jnp.stack(scores, axis=-1), axis=-1)
        out = probs[:, 0:1] * sources[0].astype(jnp.float32)
        for j in range(1, len(sources)):
            out = out + probs[:, j : j + 1] * sources[j].astype(jnp.float32)
        return out.astype(out_dtype)


class KimiK3DeltaAttention(nnx.Module):
    """KDA with a full-rank output gate and the K3 lower-bounded forget gate."""

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        linear_config = config.linear_attn_config
        if not linear_config.get("use_full_rank_gate", False):
            raise ValueError("Kimi-K3 KDA expects use_full_rank_gate=True (full-rank g_proj)")
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = linear_config["num_heads"]
        self.head_dim = linear_config["head_dim"]
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.projection_size = self.num_heads * self.head_dim
        proj = self.projection_size

        def _col(in_dim, out_dim, name):
            return LinearBase(
                in_dim,
                out_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name=name,
            )

        self.q_proj = _col(self.hidden_size, proj, "q_proj")
        self.k_proj = _col(self.hidden_size, proj, "k_proj")
        self.v_proj = _col(self.hidden_size, proj, "v_proj")
        self.g_proj = _col(self.hidden_size, proj, "g_proj")
        self.f_a_proj = LinearBase(
            self.hidden_size,
            self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="f_a_proj",
        )
        self.f_b_proj = _col(self.head_dim, proj, "f_b_proj")
        self.b_proj = _col(self.hidden_size, self.num_heads, "b_proj")

        # Depthwise conv weights [D, K], f32 like the checkpoint. LinearBase is
        # only a sharded parameter container here (never called).
        def _conv(name):
            return LinearBase(
                proj,
                self.conv_size,
                mesh,
                use_bias=False,
                params_dtype=jnp.float32,
                kernel_axes=("tensor", None),
                scope_name=name,
            )

        self.q_conv1d = _conv("q_conv1d")
        self.k_conv1d = _conv("k_conv1d")
        self.v_conv1d = _conv("v_conv1d")

        self.A_log = nnx.Param(
            jnp.zeros(
                (1, 1, self.num_heads, 1),
                dtype=jnp.float32,
                out_sharding=P(None, None, "tensor", None),
            )
        )
        self.dt_bias = nnx.Param(jnp.zeros((proj,), dtype=jnp.float32, out_sharding=P("tensor")))
        self.o_norm = GatedRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.o_proj = LinearBase(
            proj,
            self.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )

        lower_bound = linear_config.get("gate_lower_bound")
        self.attn = RadixLinearAttention(
            layer_id=layer_idx,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_heads,
            num_v_heads=self.num_heads,
            head_q_dim=self.head_dim,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            q_conv1d=self.q_conv1d,
            k_conv1d=self.k_conv1d,
            v_conv1d=self.v_conv1d,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_dim**-0.5,
            kda_lower_bound=lower_bound,
        )
        self.attn.kda_gate_lower_bound = lower_bound

    def __call__(
        self,
        positions: jax.Array | None,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, object]:
        del positions
        num_tokens = hidden_states.shape[0]
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        raw_gate, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        raw_gate = raw_gate.reshape(num_tokens, self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states)[0].astype(jnp.float32))

        o, recurrent_state_pool = self.attn(
            forward_batch, q, k, v, raw_gate, beta, recurrent_state_pool
        )
        o = o.reshape(num_tokens, self.num_heads, self.head_dim)
        gate, _ = self.g_proj(hidden_states)
        gate = gate.reshape(num_tokens, self.num_heads, self.head_dim)
        o = self.o_norm(o, gate).reshape(num_tokens, self.projection_size)
        o, _ = self.o_proj(o)
        return o, recurrent_state_pool


class KimiK3MLAAttention(DeepseekV3Attention):
    """DeepSeek MLA without RoPE plus Kimi's elementwise sigmoid output gate."""

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        use_absorbed: bool = True,
    ):
        if not getattr(config, "mla_use_nope", False):
            raise ValueError("Kimi-K3 MLA expects mla_use_nope=True")
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            mesh=mesh,
            layer_id=layer_idx,
            dtype=dtype,
            use_absorbed=use_absorbed,
            skip_rope=True,
        )
        if config.q_lora_rank is not None:
            self.q_a_layernorm.epsilon = _MLA_LORA_NORM_EPS
        self.kv_a_layernorm.epsilon = _MLA_LORA_NORM_EPS
        if getattr(config, "mla_use_output_gate", False):
            self.g_proj = LinearBase(
                config.hidden_size,
                self.num_heads * self.v_head_dim,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="g_proj",
            )
        else:
            self.g_proj = None

    def _pre_o_proj(self, attn_output: jax.Array, hidden_states: jax.Array) -> jax.Array:
        if self.g_proj is None:
            return attn_output
        flat_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        gate, _ = self.g_proj(hidden_states, out_sharding=flat_sharding)
        gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(attn_output.dtype)
        attn_output = jax.sharding.reshard(attn_output, flat_sharding)
        return attn_output * gate


class KimiK3LatentMoE(nnx.Module):
    """Router on the full hidden state, experts in the latent, plus shared experts."""

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        ep_size: int,
        quantization_config,
    ):
        self.hidden_size = config.hidden_size
        self.mesh = mesh
        # HF: LatentMoE whenever routed_expert_hidden_size is set.
        self.use_latent = getattr(config, "routed_expert_hidden_size", None) is not None
        self.latent_size = config.routed_expert_hidden_size if self.use_latent else self.hidden_size
        situ_beta, situ_linear_beta = _situ_params(config)
        if getattr(config, "hidden_act", "situ") != "situ":
            raise ValueError(f"Kimi-K3 expects hidden_act='situ', got {config.hidden_act!r}")
        if config.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi-K3 LatentMoE expects a sigmoid router")

        tp_size = int(mesh.shape.get("tensor", 1)) if mesh is not None else 1
        gate_kernel_axes = (
            (None, "tensor") if config.num_experts % tp_size == 0 else (None, None)
        )
        self.gate = GateLogit(
            input_size=self.hidden_size,
            num_experts=config.num_experts,
            enable_expert_bias=True,
            weight_dtype=dtype,
            kernel_dtype=dtype,
            compute_dtype=jnp.float32,
            bias_dtype=jnp.float32,
            score_func="sigmoid",
            kernel_axes=gate_kernel_axes,
        )
        # One expert group (K3: num_expert_group=1) is plain biased top-k.
        num_expert_group = getattr(config, "num_expert_group", 1) or 1
        topk_group = getattr(config, "topk_group", 1) or 1
        grouped = num_expert_group > 1
        self.topk = TopK(
            topk=config.num_experts_per_token,
            renormalize=config.moe_renormalize,
            num_expert_group=num_expert_group if grouped else 0,
            topk_group=topk_group if grouped else 0,
            routed_scaling_factor=config.routed_scaling_factor,
            layer_id=layer_idx,
            mesh=mesh,
        )

        if self.use_latent:
            self.routed_expert_down_proj = LinearBase(
                self.hidden_size,
                self.latent_size,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, "tensor"),
                scope_name="routed_expert_down_proj",
            )
            self.routed_expert_up_proj = LinearBase(
                self.latent_size,
                self.hidden_size,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=("tensor", None),
                scope_name="routed_expert_up_proj",
            )
            if getattr(config, "latent_moe_use_norm", False):
                self.routed_expert_norm = RMSNorm(
                    self.latent_size,
                    epsilon=config.rms_norm_eps,
                    param_dtype=dtype,
                    scope_name="routed_expert_norm",
                )
            else:
                self.routed_expert_norm = None
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
            self.routed_expert_norm = None

        self.is_mxfp4 = _is_mxfp4_quant(quantization_config)
        expert_cls = MXFP4EPMoE if self.is_mxfp4 else EPMoE
        self.experts = expert_cls(
            hidden_size=self.latent_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_token,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=config.moe_intermediate_size,
            weight_dtype=dtype,
            dtype=dtype,
            activation="situ",
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            layer_id=layer_idx,
            quantization_config=None if self.is_mxfp4 else quantization_config,
        )

        num_shared = getattr(config, "num_shared_experts", 0) or 0
        if num_shared > 0:
            self.shared_experts = KimiK3MLP(
                self.hidden_size,
                config.moe_intermediate_size * num_shared,
                mesh,
                situ_beta,
                situ_linear_beta,
                dtype=dtype,
            )
        else:
            self.shared_experts = None

    def __call__(
        self,
        hidden_states: jax.Array,
        dispatch_info: ExpertLocationMetadata | None = None,
        token_mask: jax.Array | None = None,
        compact_pos: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        router_scores = self.gate(hidden_states)
        topk_weights, topk_ids = self.topk(
            router_scores, self.gate.bias.value, dispatch_info=dispatch_info
        )
        if self.routed_expert_down_proj is not None:
            latent, _ = self.routed_expert_down_proj(hidden_states)
        else:
            latent = hidden_states
        if token_mask is not None and getattr(self.experts, "supports_token_mask", False):
            extra = {"compact_pos": compact_pos} if compact_pos is not None else {}
            routed = self.experts(latent, topk_weights, topk_ids, token_mask=token_mask, **extra)
        else:
            routed = self.experts(latent, topk_weights, topk_ids)
        if self.routed_expert_norm is not None:
            routed = self.routed_expert_norm(routed)
        if self._fuse_output_allreduce(routed):
            # The RS combine left `routed` P("data", "tensor"): sum the row-parallel
            # routed_expert_up_proj and shared-expert down_proj products, then all-reduce once.
            routed = _row_parallel_sum(
                self.mesh,
                routed,
                self.routed_expert_up_proj.weight.value,
                self.shared_experts.activation(hidden_states),
                self.shared_experts.down_proj.weight.value,
            )
            return routed, topk_ids
        if self.routed_expert_up_proj is not None:
            routed, _ = self.routed_expert_up_proj(routed)
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states)
        return routed, topk_ids

    def _fuse_output_allreduce(self, routed: jax.Array) -> bool:
        """One all-reduce for the routed up-projection plus the shared experts.

        On with the MoE reduce-scatter combine, whose P("data", "tensor") output is what the
        fused row-parallel product consumes; needs both projections and a divisible shape.
        """
        if self.routed_expert_up_proj is None or self.shared_experts is None:
            return False
        if not getattr(self.experts, "rs_combine", False) or routed.ndim != 2:
            return False
        shape = dict(self.mesh.shape)
        if set(shape) != {"data", "tensor"}:
            return False
        dp, tp = shape["data"], shape["tensor"]
        inter = self.shared_experts.down_proj.weight.value.shape[0]
        return routed.shape[0] % dp == 0 and routed.shape[1] % tp == 0 and inter % tp == 0


class KimiK3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        layer_idx: int,
        dtype: jnp.dtype,
        ep_size: int,
        quantization_config,
        use_absorbed_mla: bool,
    ):
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_kda = config.is_kda_layer(layer_idx)
        self.attn_res_block_size = config.attn_res_block_size

        if self.is_kda:
            self.self_attn = KimiK3DeltaAttention(config, layer_idx, mesh, dtype)
        else:
            self.self_attn = KimiK3MLAAttention(
                config, layer_idx, mesh, dtype, use_absorbed=use_absorbed_mla
            )

        self.is_moe_layer = (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if self.is_moe_layer:
            self.block_sparse_moe = KimiK3LatentMoE(
                config, layer_idx, mesh, dtype, ep_size, quantization_config
            )
            self.mlp = None
        else:
            situ_beta, situ_linear_beta = _situ_params(config)
            self.mlp = KimiK3MLP(
                config.hidden_size,
                config.intermediate_size,
                mesh,
                situ_beta,
                situ_linear_beta,
                dtype=dtype,
            )
            self.block_sparse_moe = None

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )
        self.self_attention_res = AttnResMix(config.hidden_size, config.rms_norm_eps)
        self.mlp_res = AttnResMix(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        positions: jax.Array,
        partial_sum: jax.Array,
        blocks: list[jax.Array],
        forward_batch: ForwardBatch,
        memory_pools,
        dispatch_info: ExpertLocationMetadata | None = None,
        compact_pos: jax.Array | None = None,
    ):
        """One layer of HF ``KimiDecoderLayer._forward_attn_residual``.

        ``partial_sum`` is the running sum of sub-layer outputs since the last
        block boundary (the embedding before layer 0) and ``blocks`` the
        completed block summaries. Returns the updated pair. ``compact_pos`` is the
        step's compact-MoE token positions (see ``MXFP4EPMoE.compact_positions``).
        """
        # No completed blocks yet (layer 0): the attention input is the embedding.
        hidden_states = self.self_attention_res(blocks, partial_sum) if blocks else partial_sum
        if self.layer_idx % self.attn_res_block_size == 0:
            blocks = [*blocks, partial_sum]
            partial_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        pool = memory_pools.recurrent_state_pool if self.is_kda else memory_pools.token_to_kv_pool
        attn_output, attn_state = self.self_attn(positions, hidden_states, forward_batch, pool)
        partial_sum = attn_output if partial_sum is None else partial_sum + attn_output

        hidden_states = self.mlp_res(blocks, partial_sum)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            # DP-padding slots have out_cache_loc == -1; the MoE skips them in prefill.
            token_mask = (
                forward_batch.out_cache_loc >= 0 if forward_batch.out_cache_loc is not None else None
            )
            ffn_output, topk_ids = self.block_sparse_moe(
                hidden_states, dispatch_info, token_mask=token_mask, compact_pos=compact_pos
            )
        else:
            ffn_output, topk_ids = self.mlp(hidden_states), None
        partial_sum = partial_sum + ffn_output
        return partial_sum, blocks, attn_state, topk_ids


class KimiK3Model(nnx.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
        ep_size: int,
        quantization_config,
        use_absorbed_mla: bool,
    ):
        if getattr(config, "attn_res_block_size", None) is None:
            raise ValueError("Kimi-K3 expects attn_res_block_size (Attention Residuals)")
        self.config = config
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                KimiK3DecoderLayer(
                    config,
                    mesh,
                    layer_idx,
                    dtype,
                    ep_size,
                    quantization_config,
                    use_absorbed_mla,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.output_attn_res = AttnResMix(config.hidden_size, config.rms_norm_eps)
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        partial_sum = self.embed_tokens(forward_batch.input_ids)
        blocks: list[jax.Array] = []
        layers_kv_fused, layers_recurrent, layers_conv, layers_topk_ids = [], [], [], []
        compact_pos = self._compact_positions(forward_batch)
        for layer in self.layers:
            partial_sum, blocks, attn_state, topk_ids = layer(
                forward_batch.positions,
                partial_sum,
                blocks,
                forward_batch,
                memory_pools,
                dispatch_info=forward_batch.expert_location_metadata,
                compact_pos=compact_pos,
            )
            if layer.is_kda:
                recurrent_buffer, conv_buffers = attn_state
                layers_recurrent.append(recurrent_buffer)
                layers_conv.append(conv_buffers)
            else:
                layers_kv_fused.append(attn_state)
            layers_topk_ids.append(topk_ids)

        hidden_states = self.output_attn_res(blocks, partial_sum)
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, (layers_recurrent, layers_conv), layers_topk_ids

    def _compact_positions(self, forward_batch: ForwardBatch) -> jax.Array | None:
        """The compact-MoE token positions of this step, once for all MoE layers (or None)."""
        loc = forward_batch.out_cache_loc
        experts = next(
            (l.block_sparse_moe.experts for l in self.layers if l.block_sparse_moe is not None),
            None,
        )
        if loc is None or experts is None or not hasattr(experts, "wants_compact_positions"):
            return None
        if loc.ndim != 1 or loc.shape[0] != forward_batch.input_ids.shape[0]:
            return None
        if not experts.wants_compact_positions(loc.shape[0], experts.num_experts_per_tok):
            return None
        return experts.compact_positions(loc >= 0)  # DP-padding slots have out_cache_loc -1


class KimiK3ForConditionalGeneration(nnx.Module):
    """Kimi-K3 text model. Vision tower / projector weights are not loaded."""

    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        config.attention_arch = AttentionArch.MLA
        text = config.hf_text_config
        qk_nope = getattr(text, "qk_nope_head_dim", 0)
        qk_rope = getattr(text, "qk_rope_head_dim", 0)
        if qk_nope and qk_rope:
            config.head_dim = qk_nope + qk_rope

    def __init__(self, config, mesh: jax.sharding.Mesh, dtype: jnp.dtype = jnp.bfloat16):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        text_config = get_kimi_k3_text_config(config)
        if text_config is None:
            raise ValueError("KimiK3ForConditionalGeneration requires a Kimi-K3 config")
        self.text_config = text_config
        quantization_config = getattr(config, "quantization_config", None)
        self.ep_size = getattr(config, "ep_size", 1) or 1
        moe_backend = getattr(config, "moe_backend", "epmoe")
        if moe_backend not in (None, "epmoe", "auto"):
            logger.warning(
                "Kimi-K3 always uses EPMoE (896 experts are not divisible by a full "
                "v5e mesh and FusedEPMoE replicates shared experts); ignoring "
                "moe_backend=%s",
                moe_backend,
            )
        self.model = KimiK3Model(
            text_config,
            mesh,
            dtype,
            ep_size=self.ep_size,
            quantization_config=quantization_config,
            use_absorbed_mla=getattr(config, "use_absorbed_mla", True),
        )
        enable_dp_lm_head = getattr(config, "enable_dp_lm_head", False)
        if getattr(text_config, "tie_word_embeddings", False):
            raise NotImplementedError("Kimi-K3 has untied embeddings")
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            enable_dp_lm_head=enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=enable_dp_lm_head,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_recurrent_state, layers_topk_ids = self.model(
            forward_batch, memory_pools
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return (
            output,
            {
                "token_to_kv_pool": layers_kv_fused,
                "recurrent_state_pool": layers_recurrent_state,
            },
            True,
            layers_topk_ids,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        prefix = "language_model."
        if (
            not loader.dummy_mode
            and not loader.has_weight_on_disk(f"{prefix}model.embed_tokens.weight")
            and loader.has_weight_on_disk("model.embed_tokens.weight")
        ):
            prefix = ""  # Text-only re-export without the multimodal wrapper.
        loader.load(self._create_weight_mappings(prefix))
        n_kernel = 0
        for layer in self.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None and isinstance(moe.experts, MXFP4EPMoE):
                n_kernel += moe.experts.prepare_gmm_kernel_weights()
        if n_kernel:
            logger.info("Kimi-K3: fused MXFP4 GMM kernel enabled for %d MoE layers", n_kernel)
        logger.info("Kimi-K3 weights loaded successfully!")

    def _create_weight_mappings(self, prefix: str = "language_model.") -> dict:
        cfg = self.text_config
        hidden = cfg.hidden_size
        src = f"{prefix}model"
        mappings = {
            f"{src}.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
            ),
            f"{src}.norm.weight": WeightSpec(target_path="model.norm.scale", sharding=(None,)),
            f"{src}.output_attn_res_norm.weight": WeightSpec(
                target_path="model.output_attn_res.norm", sharding=(None,)
            ),
            f"{src}.output_attn_res_proj.weight": WeightSpec(
                target_path="model.output_attn_res.proj", sharding=(None,), reshape=(hidden,)
            ),
            f"{prefix}lm_head.weight": self.lm_head.weight_mapping("lm_head.embedding"),
        }
        for layer_idx in range(cfg.num_hidden_layers):
            mappings.update(self._create_layer_mappings(layer_idx, src))
        return mappings

    def _create_layer_mappings(self, layer_idx: int, src: str) -> dict:
        cfg = self.text_config
        hidden = cfg.hidden_size
        layer = self.model.layers[layer_idx]
        prefix = f"{src}.layers.{layer_idx}"
        target = f"model.layers.{layer_idx}"

        def linear(name, target_name, sharding):
            return {
                f"{prefix}.{name}.weight": WeightSpec(
                    target_path=f"{target}.{target_name}.weight",
                    sharding=sharding,
                    transpose=True,
                )
            }

        col, row, rep = (None, "tensor"), ("tensor", None), (None, None)
        mappings = {
            f"{prefix}.input_layernorm.weight": WeightSpec(
                target_path=f"{target}.input_layernorm.scale", sharding=(None,)
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightSpec(
                target_path=f"{target}.post_attention_layernorm.scale", sharding=(None,)
            ),
        }
        for res in ("self_attention_res", "mlp_res"):
            mappings[f"{prefix}.{res}_norm.weight"] = WeightSpec(
                target_path=f"{target}.{res}.norm", sharding=(None,)
            )
            mappings[f"{prefix}.{res}_proj.weight"] = WeightSpec(
                target_path=f"{target}.{res}.proj", sharding=(None,), reshape=(hidden,)
            )

        if layer.is_kda:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "g_proj", "f_b_proj", "b_proj"):
                mappings.update(linear(f"self_attn.{name}", f"self_attn.{name}", col))
            mappings.update(linear("self_attn.f_a_proj", "self_attn.f_a_proj", rep))
            mappings.update(linear("self_attn.o_proj", "self_attn.o_proj", row))
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                mappings[f"{prefix}.self_attn.{name}.weight"] = WeightSpec(
                    target_path=f"{target}.self_attn.{name}.weight",
                    sharding=("tensor", None),
                    reshape=(attn.projection_size, attn.conv_size),
                )
            mappings[f"{prefix}.self_attn.dt_bias"] = WeightSpec(
                target_path=f"{target}.self_attn.dt_bias", sharding=("tensor",)
            )
            mappings[f"{prefix}.self_attn.o_norm.weight"] = WeightSpec(
                target_path=f"{target}.self_attn.o_norm.weight", sharding=(None,)
            )
            # A_log is stored zero-padded to head_dim entries; keep the first num_heads.
            a_log_key = f"{prefix}.self_attn.A_log"
            mappings[a_log_key] = WeightSpec(
                target_path=f"{target}.self_attn.A_log",
                sources=(a_log_key,),
                sharding=(None, None, "tensor", None),
                recipe=lambda inputs, h=attn.num_heads: (self._a_log_value(inputs[0], h),),
            )
        else:
            attn = layer.self_attn
            if cfg.q_lora_rank is None:
                mappings.update(linear("self_attn.q_proj", "self_attn.q_proj", col))
            else:
                mappings.update(linear("self_attn.q_a_proj", "self_attn.q_a_proj", rep))
                mappings.update(linear("self_attn.q_b_proj", "self_attn.q_b_proj", col))
                mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = WeightSpec(
                    target_path=f"{target}.self_attn.q_a_layernorm.scale", sharding=(None,)
                )
            mappings.update(linear("self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_proj", rep))
            mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightSpec(
                target_path=f"{target}.self_attn.kv_a_layernorm.scale", sharding=(None,)
            )
            # Absorbed MLA: DeepseekV3Attention.prepare_weight_loading turns this
            # into the w_uk / w_uv loading group.
            mappings.update(linear("self_attn.kv_b_proj", "self_attn.kv_b_proj", col))
            mappings.update(linear("self_attn.o_proj", "self_attn.o_proj", row))
            if attn.g_proj is not None:
                mappings.update(linear("self_attn.g_proj", "self_attn.g_proj", col))

        if not layer.is_moe_layer:
            mappings.update(linear("mlp.gate_proj", "mlp.gate_proj", col))
            mappings.update(linear("mlp.up_proj", "mlp.up_proj", col))
            mappings.update(linear("mlp.down_proj", "mlp.down_proj", row))
            return mappings

        moe = layer.block_sparse_moe
        moe_src = f"{prefix}.block_sparse_moe"
        moe_dst = f"{target}.block_sparse_moe"
        mappings[f"{moe_src}.gate.weight"] = WeightSpec(
            target_path=f"{moe_dst}.gate.kernel",
            sharding=moe.gate.kernel_axes,
            transpose=True,
        )
        mappings[f"{moe_src}.gate.e_score_correction_bias"] = WeightSpec(
            target_path=f"{moe_dst}.gate.bias", sharding=(None,)
        )
        if moe.routed_expert_down_proj is not None:
            mappings.update(
                linear(
                    "block_sparse_moe.routed_expert_down_proj",
                    "block_sparse_moe.routed_expert_down_proj",
                    col,
                )
            )
            mappings.update(
                linear(
                    "block_sparse_moe.routed_expert_up_proj",
                    "block_sparse_moe.routed_expert_up_proj",
                    row,
                )
            )
        if moe.routed_expert_norm is not None:
            mappings[f"{moe_src}.routed_expert_norm.weight"] = WeightSpec(
                target_path=f"{moe_dst}.routed_expert_norm.scale", sharding=(None,)
            )
        if moe.shared_experts is not None:
            for name, sharding in (("gate_proj", col), ("up_proj", col), ("down_proj", row)):
                mappings.update(
                    linear(
                        f"block_sparse_moe.shared_experts.{name}",
                        f"block_sparse_moe.shared_experts.{name}",
                        sharding,
                    )
                )
        if moe.is_mxfp4:
            mappings.update(
                create_mxfp4_moe_weights_mapping(
                    prefix=moe_src,
                    target_prefix=moe_dst,
                    num_experts=cfg.num_experts,
                    moe_path="experts",
                )
            )
        else:
            # bf16 (unquantized) experts: experts.E.w{1,3,2}.weight -> experts.{wi_0,wi_1,wo}.
            expert_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target,
                num_experts=cfg.num_experts,
                expert_type_names=("w1", "w3", "w2"),
                moe_backend="epmoe",
                moe_path="block_sparse_moe",
            )
            for spec in expert_mappings.values():
                spec.target_path = spec.target_path.replace(f"{moe_dst}.", f"{moe_dst}.experts.", 1)
                mappings[spec.target_path] = spec
        return mappings

    def _a_log_value(self, a_log: np.ndarray, num_heads: int) -> jax.Array:
        values = np.asarray(a_log, dtype=np.float32).reshape(-1)
        if values.size < num_heads:
            raise ValueError(f"A_log has {values.size} entries, need at least {num_heads}")
        if np.any(values[num_heads:] != 0):
            logger.warning("A_log has non-zero entries beyond num_heads=%d; ignoring", num_heads)
        values = values[:num_heads].reshape(1, 1, num_heads, 1)
        return jax.device_put(values, NamedSharding(self.mesh, P(None, None, "tensor", None)))


EntryClass = KimiK3ForConditionalGeneration
