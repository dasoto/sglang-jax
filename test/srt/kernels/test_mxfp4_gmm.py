"""CPU (Pallas interpret mode) correctness of the fused MXFP4 grouped matmul.

Run:
    JAX_PLATFORMS=cpu python -m pytest test/srt/kernels/test_mxfp4_gmm.py -q
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mxfp4_gmm import format as F
from sgl_jax.srt.kernels.mxfp4_gmm import kernel as Kn
from sgl_jax.srt.kernels.mxfp4_gmm.gmm import mxfp4_gmm, mxfp4_gmm_reference
from sgl_jax.srt.layers.quantization.mxfp4 import dequant_mxfp4


def _all_bytes_weights(k=512, n=256, n_experts=2, seed=0):
    rng = np.random.default_rng(seed)
    per = k // 2 * n
    packed = np.stack(
        [rng.permutation(np.tile(np.arange(256, dtype=np.uint8), per // 256)).reshape(k // 2, n) for _ in range(n_experts)]
    )
    scale = rng.integers(100, 140, size=(n_experts, k // 32, n), dtype=np.uint8)
    scale[0, :4, 0] = (2, 127, 200, 252)  # extremes of the exact (normal, finite) range
    return packed, scale


def test_l24_roundtrip_and_jax_encoder():
    packed, _ = _all_bytes_weights()
    words = F.encode_l24(packed)
    assert words.shape == (2, 512 // 8, 256) and words.dtype == np.uint32
    np.testing.assert_array_equal(F.decode_l24_codes(words), F.canonical_codes(packed))
    words_jax = jax.jit(F.encode_l24_jax)(jnp.asarray(packed))
    np.testing.assert_array_equal(np.asarray(words_jax), words)


def test_l24_swar_bits_match_dequant():
    packed, scale = _all_bytes_weights()
    want = np.asarray(dequant_mxfp4(jnp.asarray(packed), jnp.asarray(scale), axis=-2, out_dtype=jnp.float32))
    bits = F.bf16_bits_l24_reference(F.encode_l24(packed), scale)
    got = np.asarray(jnp.asarray(bits).view(jnp.bfloat16).astype(jnp.float32))
    nz = want != 0
    np.testing.assert_array_equal(got[nz].view(np.uint32), want[nz].view(np.uint32))
    # zeros decode to the bf16 subnormal +-2**-127 (contributes nothing to an f32 MXU sum)
    assert np.all(np.abs(got[~nz]) <= 2.0**-126)


@pytest.mark.parametrize("zero_mode", ["mul", "mask", "none"])
def test_kernel_block_decode_bits(zero_mode):
    packed, scale = _all_bytes_weights(n_experts=1)
    words = F.encode_l24(packed)[0]
    ref_bits = F.bf16_bits_l24_reference(words, scale[0])
    want = F.dequant_reference(packed, scale)[0]
    s32 = jnp.asarray(scale[0].astype(np.uint32))

    @jax.jit
    def dec(w, g0, g1):
        ga = jnp.broadcast_to(g0, (8, w.shape[1]))
        gb = jnp.broadcast_to(g1, (8, w.shape[1]))
        return jax.lax.bitcast_convert_type(Kn.decode_l24_block(w, ga, gb, zero_mode=zero_mode), jnp.uint16)

    g = Kn.scale_operand(s32, zero_mode=zero_mode)
    for blk in range(words.shape[0] // 8):
        got = np.asarray(dec(jnp.asarray(words[8 * blk : 8 * blk + 8]), g[2 * blk][None], g[2 * blk + 1][None]))
        rows = slice(64 * blk, 64 * blk + 64)
        if zero_mode != "none":
            np.testing.assert_array_equal(got, ref_bits[rows])
        else:  # no zero handling: nonzero weights are still exact
            nz = want[rows] != 0
            np.testing.assert_array_equal(got[nz], ref_bits[rows][nz])


@pytest.mark.parametrize("zero_mode", ["mul", "mask"])
def test_f8l_decode_values(zero_mode):
    packed, scale = F.random_mxfp4((512, 256), 1, seed=3, zero_frac=0.15)
    words, eg, cs = F.encode_f8l(packed, scale)
    want = F.dequant_reference(packed, scale)[0]

    @jax.jit
    def dec(w, e0, e1):
        return Kn.decode_f8l_block(w, jnp.broadcast_to(e0, (8, 256)), jnp.broadcast_to(e1, (8, 256)),
                                   zero_mode=zero_mode)

    g = Kn.f8_scale_operand(jnp.asarray(eg[0].astype(np.uint32)), zero_mode=zero_mode)
    for blk in range(words.shape[1] // 8):
        got = np.asarray(dec(jnp.asarray(words[0, 8 * blk : 8 * blk + 8]), g[2 * blk][None], g[2 * blk + 1][None]))
        got = got.astype(np.float32) * cs[0]
        ref = want[64 * blk : 64 * blk + 64]
        nz = ref != 0
        np.testing.assert_array_equal(got[nz], ref[nz])
        assert np.all(np.abs(got[~nz]) <= np.abs(ref).max() * 2.0**-20)


@pytest.mark.parametrize("fmt,zero_mode", [("l24", "mul"), ("l24", "mask"), ("f8l", "mul"), ("f8l", "mask")])
def test_dense_kernel(fmt, zero_mode):
    E, T, K, N = 2, 8, 512, 256
    packed, scale = F.random_mxfp4((K, N), E, seed=1, zero_frac=0.12)
    ref = F.dequant_reference(packed, scale)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((E, T, K)) * 0.5, jnp.bfloat16)
    if fmt == "l24":
        args = (jnp.asarray(F.encode_l24(packed)), jnp.asarray(scale))
    else:
        args = tuple(map(jnp.asarray, F.encode_f8l(packed, scale)))
    out = Kn.dense_mxfp4_matmul(x, *args, bn=256, nl=128, kc=256, zero_mode=zero_mode, out_dtype=jnp.float32,
                                interpret=True)
    want = np.einsum("etk,ekn->etn", np.asarray(x, np.float32), ref)
    np.testing.assert_allclose(np.asarray(out), want, rtol=0, atol=1e-5 * np.abs(want).max())
    out2 = Kn.dense_mxfp4_matmul(x, *args, bn=128, kc=128, zero_mode=zero_mode, out_dtype=jnp.float32, bigdot=True,
                                 interpret=True)
    np.testing.assert_allclose(np.asarray(out2), want, rtol=0, atol=1e-5 * np.abs(want).max())


CASES = [
    # (G, E_local, group_offset, group sizes, tm, bn, activation, cache)
    (4, 4, 0, [5, 0, 17, 10], 16, 128, None, True),
    (4, 4, 0, [5, 0, 17, 10], 16, 128, None, False),
    (6, 2, 2, [3, 9, 20, 7, 0, 4], 16, 128, None, True),  # expert-sharded: groups 2, 3 local
    (6, 2, 4, [3, 9, 20, 7, 0, 40], 16, 256, None, True),  # last shard (M padding lands here)
    (5, 2, 1, [7, 70, 33, 1, 9], 32, 256, None, True),  # groups spanning several m-tiles
    (3, 3, 0, [40, 1, 23], 16, 128, "situ_glu", True),
    (3, 3, 0, [70, 2, 50], 16, 128, "situ_glu", True),
    (3, 3, 0, [70, 2, 50], 16, (256, 128), "situ_glu", True),  # (bn, nl): lane sub-tiles
    (3, 3, 0, [70, 2, 50], 16, (256, 128), "situ_glu", "mask"),  # cache=True + zero_mode="mask"
]


@pytest.mark.parametrize("G,E_l,goff,sizes,tm,bn,act,cache", CASES)
def test_ragged_gmm(G, E_l, goff, sizes, tm, bn, act, cache):
    K, N = 256, 256
    rng = np.random.default_rng(len(sizes) + goff)
    packed, scale = F.random_mxfp4((K, N), E_l, seed=goff, zero_frac=0.12)
    w = jnp.asarray(F.encode_l24(packed))
    s = jnp.asarray(scale)
    deq = jnp.asarray(F.dequant_reference(packed, scale))
    kw, deq_u = {}, None
    if act:
        pu, su = F.random_mxfp4((K, N), E_l, seed=goff + 100, zero_frac=0.12)
        kw = dict(w_up=jnp.asarray(F.encode_l24(pu)), scale_up=jnp.asarray(su), activation=act)
        deq_u = jnp.asarray(F.dequant_reference(pu, su))
    gs = jnp.asarray(sizes, jnp.int32)
    x = jnp.asarray(rng.standard_normal((int(sum(sizes)), K)) * 0.5, jnp.bfloat16)
    bn, nl = bn if isinstance(bn, tuple) else (bn, None)
    zero_mode = cache if isinstance(cache, str) else "mul"
    out = mxfp4_gmm(
        x, w, s, gs, jnp.int32(goff), tm=tm, bn=bn, nl=nl, kc=128, out_dtype=jnp.float32, cache=bool(cache),
        zero_mode=zero_mode, interpret=True, **kw
    )
    ref = mxfp4_gmm_reference(x, deq, gs, goff, w_up_deq=deq_u, activation=act)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=0, atol=2e-6 * float(jnp.abs(ref).max()))


def test_ragged_gmm_situ_bf16_inputs_and_empty():
    K, N = 256, 128
    packed, scale = F.random_mxfp4((K, N), 2, seed=7)
    pu, su = F.random_mxfp4((K, N), 2, seed=8)
    w, s, wu, su_ = map(jnp.asarray, (F.encode_l24(packed), scale, F.encode_l24(pu), su))
    gs = jnp.asarray([9, 30], jnp.int32)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((39, K)), jnp.bfloat16)
    out = mxfp4_gmm(x, w, s, gs, w_up=wu, scale_up=su_, activation="situ_glu", tm=16, bn=128, kc=128,
                    act_input_dtype=jnp.bfloat16, out_dtype=jnp.float32, interpret=True)
    ref = mxfp4_gmm_reference(x, jnp.asarray(F.dequant_reference(packed, scale)), gs,
                              w_up_deq=jnp.asarray(F.dequant_reference(pu, su)), activation="situ_glu",
                              act_input_dtype=jnp.bfloat16)
    # Rounding g/u to bf16 can flip by one bf16 ulp where kernel and reference f32 sums differ in
    # the last bit: require near-exact agreement almost everywhere and <= ~1 ulp elsewhere.
    out, ref = np.asarray(out), np.asarray(ref)
    close = np.isclose(out, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())
    assert close.mean() > 0.99
    np.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-3 * np.abs(ref).max())
    empty = mxfp4_gmm(x[:0], w, s, jnp.zeros((2,), jnp.int32), tm=16, bn=128, interpret=True)
    assert empty.shape == (0, N)


def test_moe_mlp_matches_bf16_epmoe_reference():
    """mxfp4_moe_mlp == dequant_mxfp4 + per-expert matmuls + situ_and_mul (EPMoE math)."""
    from sgl_jax.srt.kernels.mxfp4_gmm import mxfp4_moe_mlp, prepare_l24

    H, I, G, E_l, goff = 256, 128, 6, 3, 2
    sizes = [5, 3, 17, 0, 9, 4]
    rng = np.random.default_rng(11)
    pg, sg = F.random_mxfp4((H, I), E_l, seed=21, zero_frac=0.1)
    pu, su = F.random_mxfp4((H, I), E_l, seed=22, zero_frac=0.1)
    pd, sd = F.random_mxfp4((I, H), E_l, seed=23, zero_frac=0.1)
    x = jnp.asarray(rng.standard_normal((sum(sizes), H)), jnp.bfloat16)
    gs = jnp.asarray(sizes, jnp.int32)
    enc = jax.jit(prepare_l24)
    out = mxfp4_moe_mlp(
        x, gs, jnp.int32(goff), enc(jnp.asarray(pg)), jnp.asarray(sg), enc(jnp.asarray(pu)), jnp.asarray(su),
        enc(jnp.asarray(pd)), jnp.asarray(sd), tm=16, bn_up=128, bn_down=256, nl=128, out_dtype=jnp.float32,
        interpret=True,
    )
    deq = lambda p, s: jnp.asarray(F.dequant_reference(p, s))
    h = mxfp4_gmm_reference(x, deq(pg, sg), gs, goff, w_up_deq=deq(pu, su), activation="situ_glu",
                            act_input_dtype=jnp.bfloat16).astype(jnp.bfloat16)
    ref = mxfp4_gmm_reference(h, deq(pd, sd), gs, goff)
    out, ref = np.asarray(out), np.asarray(ref)
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-3 * np.abs(ref).max())
    assert np.isclose(out, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max()).mean() > 0.98
