from typing import List, Optional
from functools import partial

import jax
from jax import lax, random, numpy as jnp
from jax.ops import index, index_update

import flax
from flax import linen as nn

import einops as e


@partial(jax.jit, static_argnums=(1,))
def sinusoidal_positional_encoding(t: jnp.ndarray, d: int, w):
    """
    Computes the sinusoidal positional embedding
    https://arxiv.org/abs/1706.03762
    Args:
        t: positions [L]
        d: size of sequence hidden (H)
        w: rate multiplier, [1]
    Returns:
        Sinusoidal position, [L, H]
    """
    assert len(
        t.shape) == 1, f"Position must be 1D array, got {len(t.shape)}D"
    k = jnp.arange(0, d // 2, dtype=float)
    w = w / jnp.power(10000., 2.*k/d)
    wkt = jnp.outer(t, w)
    p_sin = jnp.sin(wkt)
    p_cos = jnp.cos(wkt)
    return jnp.dstack([p_sin, p_cos]).reshape(-1, d)


@jax.jit
def rotary_positional_encoding(query: jnp.ndarray, w):
    """
    Implements Rotary Positional Embedding
    https://zhuanlan.zhihu.com/p/359502624
    https://blog.eleuther.ai/rotary-embeddings/
    https://arxiv.org/abs/2104.09864

    Args:
        query: [B, L, H]
    """
    d = query.shape[-1]
    t = query.shape[-2]
    t = jnp.arange(t)
    pos = jax.vmap(sinusoidal_positional_encoding,
                   (None, None, 0), 0
                   )(t, d, w)
    cos_pos = jnp.repeat(pos[..., 1::2], 2, -1)
    sin_pos = jnp.repeat(pos[..., ::2], 2, -1)
    q2 = jnp.stack([-query[..., 1::2], query[..., ::2]], -
                   1).reshape(query.shape)
    q = query * cos_pos + q2 * sin_pos
    return q


def efficient_dot_attention_init(kernel: str = "elu"):
    """
    Efficient dot product attention, non-causal
    softmax: https://arxiv.org/pdf/1812.01243.pdf
    elu + 1: https://arxiv.org/pdf/2006.16236.pdf
    """
    if kernel == "elu":
        def phi_q(x): return nn.elu(x) + 1.0
        phi_k = phi_q
    elif kernel == "softmax":
        def phi_q(x): return nn.softmax(x, -1)
        def phi_k(x): return nn.softmax(x, -2)
    else:
        raise ValueError(f"Invalid scaling method {kernel}")

    def forward(query, key, value, mask=None):
        """
        Computes Efficient Attention with specified kernel, non-causal
        Args:
            query: [B, LQ, HK]
            key: [B, LK, HK]
            value: [B, LK, HV]
            mask: [B, LK], values of True would be masked
        Returns:
            [B, LQ, HK]
        """
        assert query.shape[0] == key.shape[0] == value.shape[0], \
            f"Batch size must match: {query.shape[0]} == {key.shape[0]} == {value.shape[0]}"
        assert key.shape[1] == value.shape[1], \
            f"Key/value must have same length: {key.shape[1]} == {value.shape[1]}"
        assert mask is None or mask.dtype == bool, f"Mask must be boolean, got {mask.dtype}"
        Q = query
        K = key
        V = value
        if mask is not None:
            mask_key = e.repeat(mask, "b lk -> b lk h", h=key.shape[-1])
            mask_value = e.repeat(mask, "b lk -> b lk h", h=value.shape[-1])
            K = jnp.where(mask_key, -1e10, K)
            V = jnp.where(mask_value, 0.0, V)
        # kv: [B, HK, HV]
        kv = jnp.einsum("blk,blv->bkv", phi_k(K), V)
        # qkv: [B, LQ, HK]
        qkv = jnp.einsum("blk,bkv->blk", phi_q(Q), kv)
        return qkv
    return jax.jit(forward)


class EfficientMultiHeadAttention(nn.Module):
    """Multihead attention using efficient/linear attention mechanism, non-causal"""
    d_model: int
    n_heads: int
    kernel: str = "elu"

    @nn.compact
    def __call__(self, query, key=None, value=None, mask=None):
        """
        Efficient Multihead attention, non-causal
        Args:
            query: [B, LQ, HQ]
            key: [B, LK, HK]
            value: [B, LK, HV]
        Returns:
            [B, LQ, HK]
        """
        # maps over the head dimension
        attn = jax.vmap(efficient_dot_attention_init(self.kernel),
                        (2, 2, 2, None), (2))
        # Self attention
        key = key if key is not None else query
        value = value if value is not None else query
        # Linear projections
        init = nn.initializers.xavier_normal()
        q = nn.Dense(self.d_model, kernel_init=init, name=f"Q_proj")(query)
        k = nn.Dense(self.d_model, kernel_init=init, name=f"K_proj")(key)
        v = nn.Dense(self.d_model, kernel_init=init, name=f"V_proj")(value)
        qs = self.split_head(q)
        ks = self.split_head(k)
        vs = self.split_head(v)
        x = attn(qs, ks, vs, mask)
        x = self.combine_head(x)
        return x

    def split_head(self, q):
        return e.rearrange(q, "b l (n h) -> b l n h", n=self.n_heads)

    def combine_head(self, q):
        return e.rearrange(q, "b l n h -> b l (n h)", n=self.n_heads)


class EfficientTransformerEncoderLayer(nn.Module):
    n_heads: int = 4
    d_model: int = 256
    ff_size: int = 2048
    kernel: str = "elu"  # elu or softmax
    dropout: Optional[float] = None

    @nn.compact
    def __call__(self, src, src_padding_mask=None):
        """
        Efficient Pre-LN Transformer encoder layer, non-causal
        Args:
            src: [B, L, D]
            src_padding_mask: [B, L]
        Returns:
            [B, L, D]
        """
        # Initialize
        attn = EfficientMultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            kernel=self.kernel,
            name="enc_self_attn")
        # LN -> Self Attention -> Residue
        # x: [B, L, D]
        x = nn.LayerNorm(name="enc_ln_1")(src)
        # x: [B, L, D]
        x = attn(x, mask=src_padding_mask)
        x = x + src
        # LN -> Feedforward -> Residue
        # y: [B, L, D]
        y = nn.LayerNorm(name="enc_ln_2")(x)
        # y: [B, L, HF]
        y = nn.Dense(self.ff_size,
                     kernel_init=nn.initializers.kaiming_normal(),
                     name="enc_ff_1")(y)
        y = nn.relu(y)
        if self.dropout is not None:
            y = nn.Dropout(rate=self.dropout)(y)
        # y: [B, L, D]
        y = nn.Dense(self.d_model,
                     kernel_init=nn.initializers.xavier_normal(),
                     name="enc_ff_2")(y)
        y = y + x
        return y


def efficient_dot_attention_causal_init(kernel: str = "elu"):
    """
    Efficient (Linear) dot product attention, causal
    softmax: https://arxiv.org/pdf/1812.01243.pdf
    elu + 1: https://arxiv.org/pdf/2006.16236.pdf
    """
    if kernel == "elu":
        def phi_q(x): return nn.elu(x) + 1.0
        phi_k = phi_q
    elif kernel == "softmax":
        def phi_q(x): return nn.softmax(x, -1)
        def phi_k(x): return nn.softmax(x, -2)
    else:
        raise ValueError(f"Invalid scaling method {kernel}")

    @jax.custom_vjp
    def attn(query, key, value, mask=None):
        pass

    def forward(query, key, value, mask=None):
        return attn(query, key, value, mask), {
            "query": query,
            "key": key,
            "value": value,
            "mask": mask
        }

    def backward(res, g):
        pass
    attn.defvjp(forward, backward)

    return jax.jit(attn)


class RotaryPositionalEmbedding(nn.Module):
    w_init: float = 1.0

    @nn.compact
    def __call__(self, inp):
        """
        Applies rotary positional embedding
        Args:
            inp: [B, L, H]
        """
        w = self.param("w", lambda _: jnp.array([self.w_init]))
        return rotary_positional_encoding(inp, w)

