"""GMM-based Expert-Parallel MoE layer and weight mapping utilities."""

import math
import os
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.kernels.sparse_core.moe_permute import (
    moe_sc_permute_enabled_by_env,
    sc_combine,
    sc_dispatch_gather,
    should_use_sparse_core,
)

# Re-export for backward compatibility: external code imports from this module.
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.model_loader.weights import WeightSpec
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    is_int4_dtype,
    quantize_tensor,
    quantize_tensor_simple,
)


# Local-window dispatch for EPMoE: an expert sees each token at most once, so the
# rows routed to one shard's experts fit in M * min(top_k, experts_per_device)
# sorted rows starting at the shard's first local row. Gathering only that window,
# instead of all M * top_k rows, shrinks the dispatch and GEMM buffers by
# top_k / experts_per_device with identical results (no capacity drop). Read at
# trace time.
MOE_LOCAL_WINDOW = os.environ.get("SGLANG_JAX_MOE_LOCAL_WINDOW", "0") == "1"
# All-to-all EPMoE exchange: each shard keeps its slice of the tokens, sends every
# routed row (int8, per-row scale) to the shard holding its expert, runs the local
# expert over the received rows and sends the results back (bf16). Replaces the
# dense exchange (all tokens gathered to every shard, outputs all-reduced). The
# per-destination capacity is the smallest of a few static buckets that fits the
# step's largest per-destination count (pmax), so no row is ever dropped.
# Needs one expert per shard and no expert-tensor split. Read at trace time.
MOE_A2A = os.environ.get("SGLANG_JAX_MOE_A2A", "0") == "1"
# Capacity buckets: each is a lax.switch branch, so keep the list short (compile time).
# The drop-free cap (one row per local token) is always the last branch.
_MOE_A2A_BUCKETS = tuple(
    int(b) for b in os.environ.get("SGLANG_JAX_MOE_A2A_BUCKETS", "8,32,128").split(",") if b
)
# Dispatch payload dtype: int8 (per-row scale) or bfloat16 (exact, twice the bytes).
MOE_A2A_DISPATCH_INT8 = os.environ.get("SGLANG_JAX_MOE_A2A_DTYPE", "int8") == "int8"
# Dense exchange variants (no control flow, no capacity): reduce-scatter the combine
# over the expert axis along tokens instead of all-reducing it (exact, half the
# combine traffic; the model layout is restored by a gather within the TP group),
# and optionally gather the tokens as int8 with per-token scales (half the
# dispatch traffic, ~1% activation error). Need no expert-tensor split and tokens
# a multiple of the expert-shard count. Read at trace time.
MOE_GATHER_RS = os.environ.get("SGLANG_JAX_MOE_GATHER_RS", "0") == "1"
MOE_GATHER_INT8 = os.environ.get("SGLANG_JAX_MOE_GATHER_INT8", "0") == "1"
# Expert-parallel replicas: split the devices into R groups that each hold every
# expert (ep_size = devices / R) and keep the dense exchange inside a group, so the
# token all-gather and the combine reduce-scatter span 1/R of the devices and move
# 1/R of the tokens, for R x the expert weights per device. Replica r holds the
# r-th contiguous block of devices (whole DP groups), i.e. the r-th block of tokens.
# The gather/RS exchange is used when the token count is a multiple of the device
# count; other batches gather all tokens and every replica computes all of them.
# Read when the layer is built.
MOE_REPLICAS = int(os.environ.get("SGLANG_JAX_MOE_REPLICAS", "1"))


def epmoe_mesh(mesh: Mesh, ep_size: int) -> Mesh:
    """Expert mesh EPMoE runs on over ``mesh``'s devices; the weight loader uses the
    same one so loaded experts land on the layer's mesh. With SGLANG_JAX_MOE_REPLICAS
    the r-th contiguous block of devices is replica r."""
    devices = mesh.devices.flatten()
    if MOE_REPLICAS > 1:
        return jax.sharding.Mesh(
            devices.reshape(MOE_REPLICAS, ep_size, -1),
            axis_names=("replica", "expert", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 3,
        )
    return jax.sharding.Mesh(
        devices.reshape(ep_size, -1),
        axis_names=("expert", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
        moe_dp_size: int = 1,
        use_sc_permute: bool | None = None,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        # Opt-in SparseCore permute/unpermute kernels; ``None`` defers to the env flag.
        self.use_sc_permute = (
            moe_sc_permute_enabled_by_env() if use_sc_permute is None else bool(use_sc_permute)
        )
        self.physical_to_logical_map = physical_to_logical_map
        self.pre_gather_quant_dtype = pre_gather_quant_dtype
        self.moe_dp_size = moe_dp_size
        self.replicate_experts = self.moe_dp_size > 1

        metadata = None if self.replicate_experts else get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype  # original dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.original_mesh = mesh
        self.mesh = mesh
        self.activation = activation
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.hidden_size = hidden_size

        # Get quantization settings from config
        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        self.weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )

        self.moe_replicas = 1 if self.replicate_experts else MOE_REPLICAS
        if self.moe_dp_size < 1:
            raise ValueError(f"moe_dp_size must be at least 1, got {self.moe_dp_size}")
        if self.replicate_experts and self.ep_size != 1:
            raise ValueError(f"replicated EPMoE requires ep_size=1, got ep_size={self.ep_size}")
        if self.replicate_experts and (
            self.quantized_dtype is not None or self.activation_quantized_dtype is not None
        ):
            raise NotImplementedError(
                "replicated EPMoE currently supports unquantized experts only"
            )
        if not self.replicate_experts and self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        if self.replicate_experts:
            if "data" not in self.mesh.axis_names or "tensor" not in self.mesh.axis_names:
                raise ValueError(
                    "replicated EPMoE requires a model mesh with ('data', 'tensor') axes; "
                    f"got {self.mesh.axis_names}"
                )
            if self.mesh.shape["data"] != self.moe_dp_size:
                raise ValueError(
                    "replicated EPMoE requires moe_dp_size to match the model mesh data axis; "
                    f"got moe_dp_size={self.moe_dp_size}, data={self.mesh.shape['data']}"
                )
            self.tp_size = self.mesh.shape["tensor"]
            self.experts_per_device = self.num_experts
            self.moe_mesh = self.mesh
            self.updated_mesh = self.mesh.abstract_mesh
            wi_sharding = P(None, None, "tensor")
            wo_sharding = P(None, "tensor", None)
        else:
            world_size = math.prod(self.mesh.shape.values())
            replicas = self.moe_replicas
            if replicas < 1 or world_size % (self.ep_size * replicas):
                raise ValueError(
                    f"SGLANG_JAX_MOE_REPLICAS={replicas} x ep_size={self.ep_size} must divide "
                    f"the {world_size} devices"
                )
            self.tp_size = world_size // (self.ep_size * replicas)
            if replicas > 1 and self.tp_size != 1:
                raise ValueError(
                    f"MoE replicas need ep_size * replicas == devices; got ep_size={self.ep_size}, "
                    f"replicas={replicas}, devices={world_size}"
                )
            self.experts_per_device = self.num_experts // self.ep_size

            # Weights are sharded over "expert" only, so every replica holds them all.
            self.moe_mesh = epmoe_mesh(self.mesh, self.ep_size)
            abstract_mesh = self.mesh.abstract_mesh
            if replicas > 1:
                self.updated_mesh = abstract_mesh.update(
                    axis_sizes=(replicas, self.ep_size, 1),
                    axis_names=("replica", "expert", "tensor"),
                    axis_types=(jax.sharding.AxisType.Explicit,) * 3,
                )
            else:
                self.updated_mesh = abstract_mesh.update(
                    axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
                )
            wi_sharding = P("expert", None, "tensor")
            wo_sharding = P("expert", "tensor", None)

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            # MOE weights' shape is (num_experts, k, n)
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=wi_sharding,
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=wi_sharding,
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=wo_sharding,
                )
            )

            # Scales are None by default - only set by quantize_weights() if quantization is enabled
            # gmm kernel handles None scales properly (no scaling applied)
            self.wi_0_scale = None
            self.wi_1_scale = None
            self.wo_scale = None

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            return can_use_ragged, primary_device
        except Exception as _:
            return False, "cpu"

    def _get_wo_scale_sharding(
        self, scale: jax.Array | None = None, is_block: bool | None = None
    ) -> P:
        """Derive the sharding spec for wo_scale.

        When contracting-dimension block quantization is used (k_blocks > 1),
        wo_scale's K-dimension is partitioned across the tensor axis.
        """
        if is_block is None:
            if scale is not None:
                is_block = scale.shape[1] > 1 if scale.ndim == 4 else scale.shape[-1] > 1
            elif self.wo_scale is not None:
                wo_scale_val = (
                    self.wo_scale.value if hasattr(self.wo_scale, "value") else self.wo_scale
                )
                is_block = wo_scale_val.shape[1] > 1 if wo_scale_val is not None else False
            else:
                is_block = False
        return P("expert", "tensor" if is_block else None, None, None)

    def _normalize_scale_for_gmm(
        self,
        scale: jax.Array | None,
        weight: jax.Array,
        *,
        scale_name: str,
    ) -> jax.Array | None:
        """Normalize offline/runtime scale tensors to GMM's 4D layout.

        Accepted inputs intentionally cover the layouts we see in practice:

        - per-channel: ``[E, out_dim]``
        - already-kernel-ready: ``[E, k_blocks, 1, out_dim]``
        - sub-channel / block-channel: ``[E, out_dim, k_blocks]`` or
          ``[E, k_blocks, out_dim]``
        - offline 2D block quant: ``[E, out_blocks, k_blocks]``

        The returned tensor always matches the GMM contract
        ``[E, k_blocks, 1, out_dim]``.
        """
        if scale is None:
            return None

        # Weight layout is [E, k, n] where k=contraction dim, n=output dim.
        num_experts, in_dim, out_dim = weight.shape

        scale_gmm: jax.Array | None = None

        if scale.ndim == 4:
            if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
                raise ValueError(
                    f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                    "Expected 4D GMM scale layout [E, k_blocks, 1, out_dim]."
                )
            if self.weight_block_size is None:
                if scale.shape[1] != 1:
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        "Per-channel 4D GMM scales must have k_blocks=1."
                    )
            else:
                block_size_k = int(self.weight_block_size[1])
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k
                if scale.shape[1] not in (1, expected_k_blocks):
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        f"Expected k_blocks dimension to be 1 or {expected_k_blocks}."
                    )
            scale_gmm = scale

        elif scale.ndim == 2 and scale.shape == (num_experts, out_dim):
            scale_gmm = scale[:, None, None, :]

        elif scale.ndim == 3:
            if scale.shape == (num_experts, 1, out_dim):
                scale_gmm = scale[:, :, None, :]
            # Support offline 2D block quant checkpoints whose scales are stored as
            # [num_experts, out_blocks, in_blocks]. GMM expects [E, k_blocks, 1, out_dim].
            elif (
                self.weight_block_size is not None
                and isinstance(self.weight_block_size, (list, tuple))
                and len(self.weight_block_size) == 2
            ):
                block_size_out = int(self.weight_block_size[0])
                block_size_k = int(self.weight_block_size[1])
                expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k

                if scale.shape == (num_experts, out_dim, expected_k_blocks):
                    scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
                elif scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
                    scale_per_out_sharding = (
                        P("expert", None, None)
                        if scale_name == "wo_scale"
                        else P("expert", "tensor", None)
                    )
                    scale_per_out = scale.at[
                        :, jnp.arange(out_dim, dtype=jnp.int32) // block_size_out, :
                    ].get(out_sharding=scale_per_out_sharding)
                    scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
                elif scale.shape == (num_experts, expected_k_blocks, out_dim):
                    scale_gmm = scale[:, :, None, :]

        if scale_gmm is None:
            raise ValueError(
                f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                "Expected one of: [E, out_dim], [E, 1, out_dim], [E, k_blocks, 1, out_dim], "
                "or offline block format [E, out_blocks, k_blocks]."
            )

        final_scale_sharding = (
            self._get_wo_scale_sharding(scale_gmm)
            if scale_name == "wo_scale"
            else P("expert", None, None, "tensor")
        )
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)

    def quantize_weights(self, is_static: bool = False, *, abstract: bool = False):
        """Quantize MoE weights in-place or initialize params for static loading."""
        if self.quantized_dtype is None:
            return

        def _get_block_size_k(
            *,
            hidden_size: int,
            intermediate_dim: int,
            weight_block_size: list[int] | tuple[int, int] | None,
        ) -> int | None:
            """Extract the contracting-dimension block size for MoE weights.

            EPMoE only block-quantizes along the GEMM ``K`` dimension, so for a
            configured ``(block_n, block_k)`` we consume only ``block_k`` here.
            The divisibility checks keep the later GMM scale layout well-defined.
            """
            if weight_block_size is None:
                return None
            if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
                raise ValueError(
                    f"EPMoE weight_block_size must be a 2-element list [block_n, block_k], "
                    f"got {weight_block_size}"
                )

            block_size_k = int(weight_block_size[1])
            if block_size_k <= 0:
                raise ValueError(f"EPMoE weight_block_size[1] must be > 0, got {block_size_k}")
            if hidden_size % block_size_k != 0:
                raise ValueError(
                    f"EPMoE hidden_size={hidden_size} not divisible by block_size_k={block_size_k}"
                )
            if intermediate_dim % block_size_k != 0:
                raise ValueError(
                    f"EPMoE intermediate_dim={intermediate_dim} not divisible by block_size_k={block_size_k}"
                )
            return block_size_k

        mesh_context = (
            jax.sharding.use_abstract_mesh(self.moe_mesh.abstract_mesh)
            if abstract
            else jax.set_mesh(self.moe_mesh)
        )
        with mesh_context:
            if is_static:
                if abstract or isinstance(self.wi_0.value, jax.ShapeDtypeStruct):
                    # Prepare quantized placeholders for checkpoint and dummy
                    # loading without replacing already loaded concrete weights.
                    for name in ("wi_0", "wi_1", "wo"):
                        param = getattr(self, name)
                        param.value = jax.ShapeDtypeStruct(
                            param.value.shape,
                            self.quantized_dtype,
                            sharding=jax.sharding.NamedSharding(
                                self.moe_mesh, param.value.sharding.spec
                            ),
                        )
                # Static checkpoints will load real scale tensors later, but the
                # placeholders must already satisfy expert sharding shape rules.
                num_experts = self.wi_0.value.shape[0]
                # [E, k, n] layout: wi_0=[E, hidden_size, intermediate_dim],
                #                    wo=[E, intermediate_dim, hidden_size]
                hidden_size = self.wi_0.value.shape[1]
                intermediate_dim = self.wo.value.shape[1]

                # Compute k_blocks for block quant placeholders.
                # weight_block_size = [hf_out_block, hf_in_block] (HF convention).
                # EPMoE quantizes along axis=1 (k/contraction dim).
                block_size_k = _get_block_size_k(
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    weight_block_size=self.weight_block_size,
                )
                k_blocks_wi = (hidden_size // block_size_k) if block_size_k else 1
                k_blocks_wo = (intermediate_dim // block_size_k) if block_size_k else 1
                wi_scale_sharding = P("expert", None, None, "tensor")
                wo_scale_sharding = self._get_wo_scale_sharding(is_block=(k_blocks_wo > 1))
                is_abstract = isinstance(self.wi_0.value, jax.ShapeDtypeStruct)
                scale_dtype = self.dtype if is_int4_dtype(self.quantized_dtype) else jnp.float32

                def _make_param(shape, dtype, sharding_spec):
                    if is_abstract:
                        return nnx.Param(
                            jax.ShapeDtypeStruct(
                                shape,
                                dtype=dtype,
                                sharding=jax.sharding.NamedSharding(self.moe_mesh, sharding_spec),
                            )
                        )
                    return nnx.Param(
                        jnp.zeros(shape, dtype=dtype, out_sharding=sharding_spec),
                        out_sharding=sharding_spec,
                    )

                if hasattr(self, "wi_0_scale"):
                    del self.wi_0_scale
                self.wi_0_scale = _make_param(
                    (num_experts, k_blocks_wi, 1, intermediate_dim),
                    scale_dtype,
                    wi_scale_sharding,
                )

                if hasattr(self, "wi_1_scale"):
                    del self.wi_1_scale
                self.wi_1_scale = _make_param(
                    (num_experts, k_blocks_wi, 1, intermediate_dim),
                    scale_dtype,
                    wi_scale_sharding,
                )

                if hasattr(self, "wo_scale"):
                    del self.wo_scale
                self.wo_scale = _make_param(
                    (num_experts, k_blocks_wo, 1, hidden_size),
                    scale_dtype,
                    wo_scale_sharding,
                )

                return

            # Quantize weights along k-dim (axis=1 in [g, k, n] layout)
            # wi_0=[E, hidden_size, intermediate_dim], wo=[E, intermediate_dim, hidden_size]
            hidden_size = self.wi_0.value.shape[1]
            intermediate_dim = self.wo.value.shape[1]
            block_size_k = _get_block_size_k(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                weight_block_size=self.weight_block_size,
            )
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=1,
                block_size=block_size_k,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=1,
                block_size=block_size_k,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=1,
                block_size=block_size_k,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", "tensor", None))

            if block_size_k is not None:
                # axis=1 quantization on [g, k, n] gives scale [g, k_blocks, n]
                # → expand to [g, k_blocks, 1, n]
                w0_scale = w0_scale[:, :, None, :]
                w1_scale = w1_scale[:, :, None, :]
                wo_scale = wo_scale[:, :, None, :]
            else:
                w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
                w1_scale = w1_scale.reshape(w1_scale.shape[0], 1, 1, w1_scale.shape[1])
                wo_scale = wo_scale.reshape(wo_scale.shape[0], 1, 1, wo_scale.shape[1])

            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale,
                out_sharding=self._get_wo_scale_sharding(wo_scale),
            )

    @named_scope
    def __call__(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding | None = None,
    ) -> jax.Array:
        if self.replicate_experts:
            if out_sharding is None:
                out_sharding = jax.sharding.NamedSharding(
                    self.mesh,
                    P("data", *([None] * (hidden_states.ndim - 1))),
                )
            return self._call_replicated(
                hidden_states,
                topk_weights,
                topk_ids,
                out_sharding=out_sharding,
            )

        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))

        if self._use_a2a(hidden_states.shape[0]):
            return self._call_a2a(hidden_states, topk_weights, topk_ids, out_sharding)

        # Translate the caller's target sharding (on self.mesh: data,tensor)
        # into shard_map out_specs (on self.moe_mesh: expert,tensor). Only
        # 'tensor' is shared between the two meshes; everything else is
        # irrelevant inside the per-expert shard_map context.
        out_specs = P(
            *[
                ("tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None)
                for s in out_sharding.spec
            ]
        )
        scatter_on_tensor = "tensor" in out_specs
        replicas = self.moe_replicas
        gather_rs = (
            MOE_GATHER_RS
            and self.tp_size == 1
            and self.ep_size > 1
            and hidden_states.shape[0] % (self.ep_size * replicas) == 0
        )
        gather_int8 = gather_rs and MOE_GATHER_INT8
        # With replicas each one gathers only its own block of tokens.
        token_spec = P("replica") if gather_rs and replicas > 1 else P(None)
        if gather_rs:
            out_specs = P(("replica", "expert")) if replicas > 1 else P("expert")
            scatter_on_tensor = False
        token_scale = None
        if gather_int8:
            hidden_states, token_scale = quantize_tensor_simple(hidden_states, jnp.int8, dim=-1)

        # Run MoE computation on the expert-parallel mesh
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, token_spec)
            topk_weights_reshard = jax.sharding.reshard(topk_weights, token_spec)
            topk_ids_reshard = jax.sharding.reshard(topk_ids, token_spec)
            if token_scale is not None:
                token_scale = jax.sharding.reshard(token_scale, token_spec)

            # Normalize scales to GMM's 4D layout [E, k_blocks, 1, out_dim]
            w0_scale = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                self.wi_0.value,
                scale_name="wi_0_scale",
            )
            w1_scale = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                self.wi_1.value,
                scale_name="wi_1_scale",
            )
            wo_scale = self._normalize_scale_for_gmm(
                self.wo_scale.value if self.wo_scale is not None else None,
                self.wo.value,
                scale_name="wo_scale",
            )

            result = shard_map(
                partial(
                    self._forward,
                    scatter_on_tensor=scatter_on_tensor,
                    scatter_on_expert=gather_rs,
                    has_token_scale=token_scale is not None,
                ),
                mesh=self.moe_mesh,
                in_specs=(
                    token_spec,
                    token_spec,
                    token_spec,
                    # weights [g, k, n]
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    # scales [g, 1, 1, n]
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    self._get_wo_scale_sharding(wo_scale),
                    # biases [g, 1, n] (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", None, None),
                    # per-token int8 scale of the gathered tokens (or None)
                    token_spec,
                ),
                out_specs=out_specs,
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.wi_0.value,
                self.wi_1.value,
                self.wo.value,
                w0_scale,
                w1_scale,
                wo_scale,
                None,
                None,
                None,
                token_scale,
            )

        # The shard_map ran under updated_mesh (expert, tensor); land back on
        # the original mesh so downstream ops (residual add, layernorm) see a
        # consistent context.
        return jax.sharding.reshard(result, out_sharding)

    def _use_a2a(self, num_tokens: int) -> bool:
        return (
            MOE_A2A
            and not self.replicate_experts
            and self.moe_replicas == 1
            and self.experts_per_device == 1
            and self.tp_size == 1
            and self.ep_size > 1
            and num_tokens % self.ep_size == 0
        )

    def _call_a2a(self, hidden_states, topk_weights, topk_ids, out_sharding):
        """All-to-all exchange (see MOE_A2A): tokens stay sharded over the expert axis."""
        m_local = hidden_states.shape[0] // self.ep_size
        buckets = tuple(
            sorted({min(b, m_local * self.num_experts_per_tok) for b in _MOE_A2A_BUCKETS})
        )
        buckets = tuple(b for b in buckets if b < m_local) + (m_local,)
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            x = jax.sharding.reshard(hidden_states, P("expert"))
            w = jax.sharding.reshard(topk_weights, P("expert"))
            ids = jax.sharding.reshard(topk_ids, P("expert"))
            w0_scale = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                self.wi_0.value,
                scale_name="wi_0_scale",
            )
            w1_scale = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                self.wi_1.value,
                scale_name="wi_1_scale",
            )
            wo_scale = self._normalize_scale_for_gmm(
                self.wo_scale.value if self.wo_scale is not None else None,
                self.wo.value,
                scale_name="wo_scale",
            )
            result = shard_map(
                partial(self._forward_a2a, buckets=buckets),
                mesh=self.moe_mesh,
                in_specs=(
                    P("expert"),
                    P("expert"),
                    P("expert"),
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    self._get_wo_scale_sharding(wo_scale),
                ),
                out_specs=P("expert"),
                check_vma=False,
            )(
                x,
                w,
                ids,
                self.wi_0.value,
                self.wi_1.value,
                self.wo.value,
                w0_scale,
                w1_scale,
                wo_scale,
            )
        return jax.sharding.reshard(result, out_sharding)

    def _forward_a2a(
        self, x, weights, topk_ids, w0, w1, wo, w0_scale, w1_scale, wo_scale, *, buckets
    ):
        ep = self.ep_size
        m, hidden = x.shape
        top_k = self.num_experts_per_tok
        n = m * top_k
        dest = jnp.ravel(topk_ids).astype(jnp.int32)  # one expert per shard: expert id == shard
        token = jnp.arange(n, dtype=jnp.int32) // top_k
        # slot of each routed row inside its destination's send buffer
        order = jnp.argsort(dest, stable=True)
        counts = jnp.bincount(dest, length=ep).astype(jnp.int32)
        starts = jnp.cumsum(counts) - counts
        pos = (
            jnp.zeros(n, jnp.int32)
            .at[order]
            .set(jnp.arange(n, dtype=jnp.int32) - starts[dest[order]])
        )
        if MOE_A2A_DISPATCH_INT8:
            x_q, x_scale = quantize_tensor_simple(x, jnp.int8, dim=-1)
            x_scale = jnp.reshape(x_scale, (m,)).astype(jnp.float32)
        else:
            x_q, x_scale = x.astype(self.dtype), jnp.ones((m,), jnp.float32)
        flat_w = jnp.ravel(weights).astype(jnp.float32)

        def exchange(cap):
            def run(_):
                send = (
                    jnp.zeros((ep, cap, hidden), x_q.dtype)
                    .at[dest, pos]
                    .set(x_q[token], mode="drop")
                )
                send_scale = (
                    jnp.zeros((ep, cap), jnp.float32).at[dest, pos].set(x_scale[token], mode="drop")
                )
                recv = jax.lax.all_to_all(send, "expert", 0, 0)
                recv_scale = jax.lax.all_to_all(send_scale, "expert", 0, 0)
                rows = (
                    recv.reshape(ep * cap, hidden).astype(jnp.float32)
                    * recv_scale.reshape(ep * cap, 1)
                ).astype(self.dtype)
                # every received row belongs to this shard's single expert; padding rows
                # are zero and stay zero through the (bias-free) expert MLP.
                out = self._local_expert_mlp(rows, w0, w1, wo, w0_scale, w1_scale, wo_scale)
                back = jax.lax.all_to_all(
                    out.reshape(ep, cap, hidden).astype(self.dtype), "expert", 0, 0
                )
                picked = back[dest, jnp.minimum(pos, cap - 1)].astype(jnp.float32)
                return jnp.sum((picked * flat_w[:, None]).reshape(m, top_k, hidden), axis=1)

            return run

        cmax = jax.lax.pmax(jnp.max(counts), "expert")
        index = jnp.sum(jnp.asarray(buckets, jnp.int32) < cmax)
        out = jax.lax.switch(
            jnp.minimum(index, len(buckets) - 1), [exchange(b) for b in buckets], None
        )
        return out.astype(self.dtype)

    def _local_expert_mlp(self, x, w0, w1, wo, w0_scale, w1_scale, wo_scale):
        """The shard's single expert as dense XLA matmuls (no Pallas GMM, so the
        all-to-all capacity branches stay cheap to compile). Weights [1, k, n]
        with optional scales in GMM layout [1, k_blocks, 1, n]."""

        def dequant(w, scale):
            if scale is None:
                return w[0].astype(self.dtype)
            _, k, n = w.shape
            kb = scale.shape[1]
            w_f = w[0].astype(jnp.float32).reshape(kb, k // kb, n) * scale[0, :, 0, :][:, None, :]
            return w_f.reshape(k, n).astype(self.dtype)

        gate = jnp.dot(x, dequant(w0, w0_scale), preferred_element_type=jnp.float32)
        up = jnp.dot(x, dequant(w1, w1_scale), preferred_element_type=jnp.float32)
        if self.activation == "silu":
            h = jax.nn.silu(gate) * up
        elif self.activation == "gelu":
            h = jax.nn.gelu(gate) * up
        elif self.activation == "swigluoai":
            g = jnp.clip(gate, max=self.swiglu_limit)
            u = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
            h = (u + 1.0) * g * jax.nn.sigmoid(g * self.swiglu_alpha)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")
        return jnp.dot(
            h.astype(self.dtype), dequant(wo, wo_scale), preferred_element_type=jnp.float32
        ).astype(self.dtype)

    def _call_replicated(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        *,
        out_sharding: jax.sharding.NamedSharding,
    ) -> jax.Array:
        token_spec = P("data", *([None] * (hidden_states.ndim - 1)))
        routing_spec = P("data", *([None] * (topk_ids.ndim - 1)))
        out_spec = out_sharding.spec
        token_axis = out_spec[0] if len(out_spec) > 0 else None
        token_axes = token_axis if isinstance(token_axis, tuple) else (token_axis,)
        if self.mesh.shape["data"] > 1 and "data" not in token_axes:
            raise ValueError(
                "replicated EPMoE output must shard the token dimension over the data axis"
            )
        scatter_on_tensor = "tensor" in token_axes

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states = jax.sharding.reshard(hidden_states, token_spec)
            topk_weights = jax.sharding.reshard(topk_weights, routing_spec)
            topk_ids = jax.sharding.reshard(topk_ids, routing_spec)
            result = shard_map(
                partial(self._forward, scatter_on_tensor=scatter_on_tensor),
                mesh=self.moe_mesh,
                in_specs=(
                    token_spec,
                    routing_spec,
                    routing_spec,
                    P(None, None, "tensor"),
                    P(None, None, "tensor"),
                    P(None, "tensor", None),
                ),
                out_specs=out_spec,
                check_vma=False,
            )(
                hidden_states,
                topk_weights,
                topk_ids,
                self.wi_0[...],
                self.wi_1[...],
                self.wo[...],
            )

        return jax.sharding.reshard(result, out_sharding)

    def _forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        token_scale=None,
        *,
        scatter_on_tensor: bool = False,
        scatter_on_expert: bool = False,
        has_token_scale: bool = False,
    ):
        if has_token_scale:
            hidden_states = (hidden_states.astype(jnp.float32) * token_scale).astype(self.dtype)
        expert_shard_id = (
            jnp.array(0, dtype=jnp.int32)
            if self.replicate_experts
            else jax.lax.axis_index("expert")
        )

        inputs_2d, token_indices, sorted_selected_experts, group_sizes = self._permute(
            hidden_states, topk_ids
        )

        group_sizes = group_sizes.astype(jnp.int32)

        group_offset = self._dispatch(group_sizes, expert_shard_id)

        local_range = None
        valid_mask = None
        # Trace-time gate: small (decode-sized) batches keep the exact XLA path so
        # their HLO is identical to the flag-off build.
        if self.use_sc_permute and should_use_sparse_core(
            token_indices.shape[0], inputs_2d.shape[-1], self.dtype
        ):
            # Sorted-row range owned by this expert shard, and which routed slots
            # (token-major order) landed on a local expert.
            csum = jnp.cumsum(group_sizes)
            start = jnp.where(group_offset == 0, 0, csum[jnp.maximum(group_offset - 1, 0)])
            end = csum[group_offset + self.experts_per_device - 1]
            if self.ep_size > 1:
                local_range = (start.astype(jnp.int32), end.astype(jnp.int32))
            flat_experts = jnp.ravel(topk_ids)
            valid_mask = (flat_experts >= group_offset) & (
                flat_experts < group_offset + self.experts_per_device
            )

        window = None
        cap = hidden_states.shape[0] * min(self.num_experts_per_tok, self.experts_per_device)
        if (
            MOE_LOCAL_WINDOW
            and self.ep_size > 1
            and local_range is None
            and cap < token_indices.shape[0]
        ):
            csum = jnp.cumsum(group_sizes)
            start = jnp.where(group_offset == 0, 0, csum[jnp.maximum(group_offset - 1, 0)])
            n_local = csum[group_offset + self.experts_per_device - 1] - start
            window = (start.astype(jnp.int32), n_local.astype(jnp.int32), cap)

        intermediate_output = self._gmm_compute(
            inputs_2d,
            token_indices,
            group_sizes,
            w0_weights,
            w1_weights,
            wo_weights,
            group_offset,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
            w0_kernel_bias,
            w1_kernel_bias,
            wo_kernel_bias,
            local_range=local_range,
            window=window,
        )

        output = self._unpermute(
            intermediate_output,
            sorted_selected_experts,
            topk_weights,
            valid_mask=valid_mask,
            window=window,
        )

        # Reduce on the "tensor" axis. RS (psum_scatter) when caller asked
        # for SP layout on the token dim, AR (psum) otherwise. The matching
        # out_specs is set in __call__ from the same source of truth.
        if self.tp_size > 1:
            if scatter_on_tensor:
                output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
            else:
                output = jax.lax.psum(output, "tensor")
        if self.ep_size > 1:
            if scatter_on_expert:
                output = jax.lax.psum_scatter(output, "expert", scatter_dimension=0, tiled=True)
            else:
                output = self._combine(output)

        return output

    def _gmm_compute(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        group_offset,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
        local_range=None,
        window=None,
    ):
        if token_indices.shape[0] == 0:
            return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)

        # indexed_gmm: gather sorted_inputs here instead of in _permute,
        # so XLA can fuse the gather with the matmul and avoid materializing
        # the full [M*top_k, D] sorted_inputs tensor at peak memory.
        pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
        if window is not None:
            # Rows [start, start + cap) of the expert-sorted order: the local experts'
            # n_local rows first, then rows the GEMM leaves to a non-local group (or to
            # the last local expert, whose extra rows _unpermute masks out).
            start, n_local, cap = window
            pos = jnp.minimum(start + jnp.arange(cap, dtype=jnp.int32), token_indices.shape[0] - 1)
            if pre_gather_q is not None:
                x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
                x = x_q[token_indices[pos]]
                x = (x.astype(jnp.float32) * x_scale[token_indices[pos]]).astype(self.dtype)
            else:
                x = inputs_2d[token_indices[pos]].astype(self.dtype)
            experts = jnp.arange(group_sizes.shape[0])
            is_local = (experts >= group_offset) & (
                experts < group_offset + self.experts_per_device
            )
            group_sizes = jnp.where(is_local, group_sizes, 0)
            group_sizes = group_sizes.at[-1].add(cap - n_local)
        elif pre_gather_q is not None:
            x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
            x = x_q[token_indices]
            x_scale = x_scale[token_indices]
            x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
        elif local_range is not None:
            # SparseCore ragged gather: only rows [start, end) (local experts) are
            # materialized; gmm never reads the others thanks to group_offset.
            x = sc_dispatch_gather(inputs_2d, token_indices, *local_range).astype(self.dtype)
        else:
            x = inputs_2d[token_indices].astype(self.dtype)

        # NOTE: do NOT pad LHS / bump group_sizes here. The megablox backend
        # ``gmm`` (sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py:67-73)
        # already pads ``lhs`` to its required alignment (32 for v2, 128 for
        # v1), bumps ``group_sizes[-1]`` accordingly, and slices the output
        # back to the original ``m`` afterwards. An outer pre-pad is at best
        # redundant; in practice the previous workaround pre-padded to a
        # hard-coded ``128`` which forced v2 (alignment=32) into a 4x larger
        # tile, hit a kernel auto-tiler edge case at decode bs=8 / top_k=8
        # (m=64 -> 128) and triggered an on-device SparseCore halt.
        group_sizes = group_sizes.astype(jnp.int32)
        act_q_dtype = self.activation_quantized_dtype

        gmm_kwargs = dict(
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            group_offset=group_offset,
            maybe_quantize_lhs=act_q_dtype is not None,
            acc_dtype=jnp.float32,
        )

        # === GEMM1: x @ w0 and x @ w1 ===
        layer_w0 = gmm(
            lhs=x,
            rhs=w0_kernel,
            rhs_scale=w0_kernel_scale,
            rhs_bias=w0_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )
        layer_w1 = gmm(
            lhs=x,
            rhs=w1_kernel,
            rhs_scale=w1_kernel_scale,
            rhs_bias=w1_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

        # === Activation ===
        if self.activation == "silu":
            intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
        elif self.activation == "gelu":
            intermediate_layer = jax.nn.gelu(layer_w0) * layer_w1
        elif self.activation == "swigluoai":
            gate = jnp.clip(layer_w0, max=self.swiglu_limit)
            up = jnp.clip(layer_w1, -self.swiglu_limit, self.swiglu_limit)
            intermediate_layer = (up + 1.0) * gate * jax.nn.sigmoid(gate * self.swiglu_alpha)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")

        # === GEMM2: intermediate @ wo ===
        return gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            rhs_scale=wo_kernel_scale,
            rhs_bias=wo_kernel_bias,
            zero_initialize=True,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

    def _dispatch(self, group_sizes, expert_shard_id):
        if self.ep_size <= 1:
            return jnp.array(0, dtype=jnp.int32)
        group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
        return group_offset

    def _get_all_to_all_params(
        self,
        tokens_group: jax.Array,
        shard_id: jax.Array,
        start_idx: jax.Array,
        *,
        ep_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        input_offsets = jnp.full(ep_size, start_idx, dtype=tokens_group.dtype)
        send_sizes = jnp.repeat(tokens_group[shard_id], ep_size)
        output_offset = jnp.concatenate(
            (jnp.array([0], dtype=tokens_group.dtype), jnp.cumsum(tokens_group[:-1]))
        )[shard_id]
        output_offsets = jnp.repeat(output_offset, ep_size)
        recv_sizes = tokens_group

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _combine(self, data):
        return jax.lax.psum(data, "expert")

    def _permute(self, inputs, top_k_indices):
        if inputs.ndim != 2:
            raise ValueError(
                "EPMoE._permute expects 2-D hidden states [tokens, hidden], "
                f"got shape {inputs.shape}"
            )

        expected_indices_shape = (inputs.shape[0], self.num_experts_per_tok)
        if top_k_indices.shape != expected_indices_shape:
            raise ValueError(
                "EPMoE._permute expects routing indices with shape "
                f"{expected_indices_shape}, got shape {top_k_indices.shape}"
            )

        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        # token_indices: maps each sorted position to the original token index.
        # Pass to _gmm_compute so the gather happens there (indexed_gmm pattern),
        # avoiding a full [M*top_k, D] materialization in _permute.
        token_indices = sorted_selected_experts // self.num_experts_per_tok

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        return (
            inputs,
            token_indices,
            sorted_selected_experts,
            group_sizes,
        )

    def _unpermute(
        self, intermediate, sorted_selected_experts, weights, valid_mask=None, window=None
    ):
        top_k = self.num_experts_per_tok
        if weights.ndim != 2 or weights.shape[1] != top_k:
            raise ValueError(
                "EPMoE._unpermute expects 2-D routing weights "
                f"[tokens, {top_k}], got shape {weights.shape}"
            )

        expected_tokens = weights.shape[0] * top_k
        if sorted_selected_experts.ndim != 1 or sorted_selected_experts.shape[0] != expected_tokens:
            raise ValueError(
                "EPMoE._unpermute expects 1-D sorted routing indices with "
                f"{expected_tokens} entries, got shape {sorted_selected_experts.shape}"
            )

        if window is not None:
            # intermediate holds window rows [start, start + cap) of the sorted order;
            # each (token, slot) routed to a local expert maps to row sorted_pos - start.
            start, n_local, cap = window
            argsort_indices = (
                jnp.zeros(expected_tokens, dtype=jnp.int32)
                .at[sorted_selected_experts]
                .set(jnp.arange(expected_tokens, dtype=jnp.int32))
            )
            rows = jnp.reshape(argsort_indices - start, (weights.shape[0], top_k))
            valid = (rows >= 0) & (rows < n_local)
            # A token has at most min(top_k, experts_per_device) slots on local experts:
            # move those to the front and gather only that many rows per token.
            slots = min(top_k, self.experts_per_device)
            if slots < top_k:
                order = jnp.argsort(jnp.logical_not(valid), axis=1, stable=True)[:, :slots]
                rows = jnp.take_along_axis(rows, order, axis=1)
                valid = jnp.take_along_axis(valid, order, axis=1)
                weights = jnp.take_along_axis(weights, order, axis=1)
            rows = jnp.clip(rows, 0, cap - 1)
            weights_fp32 = jnp.where(valid, weights.astype(jnp.float32), 0.0)
            output = None
            for k in range(slots):
                contribution = (
                    jnp.take(intermediate, indices=rows[:, k], axis=0).astype(jnp.float32)
                    * weights_fp32[:, k, None]
                )
                output = contribution if output is None else output + contribution
            return output.astype(self.dtype)

        actual_tokens = intermediate.shape[0]

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros((padding_size, intermediate.shape[1]), dtype=intermediate.dtype)
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        argsort_indices = (
            jnp.zeros(expected_tokens, dtype=jnp.int32)
            .at[sorted_selected_experts]
            .set(jnp.arange(expected_tokens, dtype=jnp.int32))
        )
        if self.use_sc_permute and valid_mask is not None:
            # SparseCore fused gather + top-k weighted reduce (falls back to XLA when
            # SparseCore is absent or the problem is too small to benefit).
            return sc_combine(
                intermediate,
                argsort_indices,
                jnp.reshape(weights, (-1,)),
                valid_mask,
                self.num_experts_per_tok,
            ).astype(self.dtype)

        weights_fp32 = weights.astype(jnp.float32)

        # Static token-count branch: small (decode) batches avoid the unrolled
        # per-k gather overhead; large (prefill) batches avoid the fp32
        # (tokens, top_k, hidden) intermediate. Benchmarks in the PR description.
        if weights.shape[0] <= 256:
            unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)
            reshaped_intermediate = jnp.reshape(
                unsort_intermediate,
                (weights.shape[0], top_k, -1),
            )
            output = jnp.einsum(
                "BKE,BK -> BE",
                reshaped_intermediate.astype(jnp.float32),
                weights_fp32,
            )
        else:
            grouped_indices = jnp.reshape(argsort_indices, (weights.shape[0], top_k))
            output = None
            for k in range(top_k):
                contribution = (
                    jnp.take(intermediate, indices=grouped_indices[:, k], axis=0).astype(
                        jnp.float32
                    )
                    * weights_fp32[:, k, None]
                )
                output = contribution if output is None else output + contribution

        final_output = output.astype(self.dtype)

        return final_output


# create_moe_weights_mapping is utility function to generate weight mapping for MOE layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,  # num logical experts
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),  # expert source names [gate, up, down]
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is None)
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,  # np.ndarray shape (num_physical,) or None
    weight_suffix: str = "weight",
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend in ("fused", "fused_v2"):
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    if expert_concat_axis_map is None:
        expert_concat_axis_map = {}

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters (matching EPMoE internal variables)
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths for logical experts only
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.{weight_suffix}"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Weights are transposed from HF [n, k] to [k, n], stacked to [g, k, n].
            # wi_0/wi_1: [g, hidden_size, intermediate_dim] -> P("expert", None, "tensor")
            # wo:        [g, intermediate_dim, hidden_size] -> P("expert", "tensor", None)
            sharding = (
                ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            )
            transpose = True
        elif moe_backend in ("fused", "fused_v2"):
            # Fused MoE kernel shards experts across the full EP mesh, i.e. the
            # product of ("data", "tensor"). Shard expert dim (axis=0) across
            # both mesh axes so each device owns a disjoint expert slice.
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use  prefix to indicate aggregated MoE weight loading
        mappings[f"{target_path_base}"] = WeightSpec(
            target_path=target_path_base,
            sources=tuple(expert_keys),
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )

    return mappings
