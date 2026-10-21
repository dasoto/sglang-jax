"""CPU reference for Kimi-K3 built from Moonshot's HF modeling code.

``load_hf_reference(ref_dir)`` imports ``configuration_kimi_k3.py`` and
``modeling_kimi_linear.py`` from ``ref_dir`` (the official release files, or the
raiden_lm ``dummy_hf`` vendored copy whose model code is semantically identical
but tolerates a missing ``fla``) and replaces the GPU-only ``fla`` kernels with
pure-torch implementations that follow fla-core semantics exactly:

* ``ShortConvolution``: depthwise causal conv (kernel K, zero left pad, no bias),
  fp32 accumulation, SiLU after the conv, output in the input dtype; the cache
  holds the last ``K - 1`` inputs.
* ``FusedRMSNormGated(activation="sigmoid")``: ``rmsnorm(x) * w * sigmoid(g)``
  in fp32, output in ``x.dtype``.
* ``chunk_kda`` / ``fused_recurrent_kda``: exact token recurrence in fp32 with
  ``l2norm(x) = x / sqrt(sum(x^2) + 1e-6)``, ``q *= K**-0.5``, ``beta =
  sigmoid(beta)``, per-head ``A_log`` and the K3 safe gate
  ``g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` (softplus form when
  ``lower_bound`` is None); decay the key rows, delta update, read after the
  update. The state is returned as ``[B, H, V, K]`` when
  ``transpose_state_layout`` is set, like fla.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

_PKG = "_kimi_k3_hf_modeling"


class ShortConvolution(nn.Module):
    def __init__(
        self, hidden_size: int, kernel_size: int = 4, activation: str = "silu", bias=False
    ):
        super().__init__()
        assert not bias
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        assert cu_seqlens is None, "reference conv supports dense batches only"
        batch, seq_len, dim = x.shape
        if cache is None:
            cache = x.new_zeros(batch, self.kernel_size - 1, dim)
        x_all = torch.cat([cache.to(x.dtype), x], dim=1)
        out = F.conv1d(
            x_all.transpose(1, 2).float(), self.weight.float(), None, groups=dim
        ).transpose(1, 2)
        if self.activation == "silu":
            out = F.silu(out)
        final_state = x_all[:, -(self.kernel_size - 1) :] if output_final_state else None
        return out.to(x.dtype), final_state


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "sigmoid"):
        super().__init__()
        assert activation == "sigmoid"
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float() * torch.sigmoid(g.float())
        return y.to(x.dtype)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def kda_recurrent(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    lower_bound=None,
    transpose_state_layout=False,
    cu_seqlens=None,
    **kwargs,
):
    assert cu_seqlens is None, "reference KDA supports dense batches only"
    batch, seq_len, heads, dk = q.shape
    dv = v.shape[-1]
    out_dtype = v.dtype
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q = q.float() * (dk**-0.5 if scale is None else scale)
    k, v = k.float(), v.float()
    beta = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()
    g = g.float()
    if use_gate_in_kernel:
        if dt_bias is not None:
            g = g + dt_bias.float().view(heads, dk)
        a = torch.exp(A_log.float().reshape(-1)[:heads]).view(heads, 1)
        if lower_bound is not None:
            g = lower_bound * torch.sigmoid(a * g)
        else:
            g = -a * F.softplus(g)
    if initial_state is None:
        state = q.new_zeros(batch, heads, dk, dv)
    else:
        state = initial_state.float().clone()
        if transpose_state_layout:
            state = state.transpose(-1, -2)
    outputs = []
    for t in range(seq_len):
        state = state * torch.exp(g[:, t]).unsqueeze(-1)
        v_pred = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        delta = beta[:, t].unsqueeze(-1) * (v[:, t] - v_pred)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    out = torch.stack(outputs, dim=1).to(out_dtype)
    final = state.transpose(-1, -2) if transpose_state_layout else state
    return out, (final if output_final_state else None)


def _install_fake_fla() -> None:
    """Expose the CPU kernels under the ``fla`` names the HF code imports.

    The official ``modeling_kimi_linear.py`` hard-requires ``fla``; with these
    stand-ins it imports unmodified (the vendored copy would otherwise use its
    own fallbacks, which differ from fla in eps / fp32 details).
    """
    if "fla" in sys.modules and getattr(sys.modules["fla"], "_kimi_k3_cpu_stub", False):
        return

    def _module(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
        return module

    def prepare_lens_from_mask(mask):
        return mask.sum(-1, dtype=torch.int32)

    def prepare_cu_seqlens_from_mask(mask):
        return F.pad(prepare_lens_from_mask(mask).cumsum(0, dtype=torch.int32), (1, 0))

    _module("fla", _kimi_k3_cpu_stub=True)
    _module("fla.modules", FusedRMSNormGated=FusedRMSNormGated, ShortConvolution=ShortConvolution)
    _module("fla.ops")
    _module("fla.ops.kda", chunk_kda=kda_recurrent, fused_recurrent_kda=kda_recurrent)
    _module("fla.ops.utils")
    _module(
        "fla.ops.utils.index",
        prepare_cu_seqlens_from_mask=prepare_cu_seqlens_from_mask,
        prepare_lens_from_mask=prepare_lens_from_mask,
    )
    _module("fla.utils", tensor_cache=lambda fn: fn)


def load_hf_reference(ref_dir: str) -> types.SimpleNamespace:
    """Import the HF Kimi-K3 text modeling code with CPU kernels patched in."""
    _install_fake_fla()
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [ref_dir]
        sys.modules[_PKG] = pkg
        for name in ("configuration_kimi_k3", "modeling_kimi_linear"):
            path = os.path.join(ref_dir, f"{name}.py")
            spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{_PKG}.{name}"] = module
            spec.loader.exec_module(module)
    cfg = sys.modules[f"{_PKG}.configuration_kimi_k3"]
    mod = sys.modules[f"{_PKG}.modeling_kimi_linear"]
    mod.ShortConvolution = ShortConvolution
    mod.FusedRMSNormGated = FusedRMSNormGated
    mod.chunk_kda = kda_recurrent
    mod.fused_recurrent_kda = kda_recurrent
    return types.SimpleNamespace(config=cfg, modeling=mod)
