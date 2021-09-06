"""
This file hosts modules using the regular transformer which takes O(N^2) time and memory
"""
from typing import *

import jax
from jax import random, lax, numpy as jnp
import flax
import flax.linen as nn
import einops as e


class TransformerEncoderLayer(nn.Module):
    d_model: int = 256
    n_heads: int = 4
    ff_size: int = 2048
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, src, src_padding_mask=None):
        """
        O(N^2) Pre-LN Transformer encoder layer, non-causal
        Args:
            src: [B, L, D]
            src_padding_mask: [B, L]
        Returns:
            [B, L, D]
        """
        if src_padding_mask is None:
            src_padding_mask = jnp.ones((src.shape[0], src.shape[1]))
        # LN -> Self Attention -> Residue
        # x: [B, L, D]
        x = nn.LayerNorm(name="enc_ln_1")(src)
        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            name="enc_attn"
        )(x, src_padding_mask[:, None, None, :], deterministic=False)
        x = x + src
        # LN -> Feedforward -> Residue
        # y: [B, L, D]
        y = nn.LayerNorm(name="enc_ln_2")(x)
        # y: [B, L, HF]
        y = nn.Dense(self.ff_size,
                     kernel_init=nn.initializers.kaiming_normal(),
                     name="enc_ff_1")(y)
        y = nn.relu(y)
        if self.dropout_rate is not None:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)
        # y: [B, L, D]
        y = nn.Dense(self.d_model,
                     kernel_init=nn.initializers.xavier_normal(),
                     name="enc_ff_2")(y)
        y = y + x
        return y


class TransformerDecoderLayer(nn.Module):
    d_model: int = 256
    n_heads: int = 4
    ff_size: int = 2048
    dropout_rate: float = 0.0
    decode: bool = False

    @nn.compact
    def __call__(self, memory, tgt, memory_padding_mask=None, tgt_padding_mask=None, causal=False):
        """
        O(N^2) Pre-LN Transformer decoder layer
        Args:
            memory: [B, LS, D]
            tgt: [B, LT, D]
            memory_padding_mask: [B, LS]
            tgt_padding_mask: [B, LT]
        Returns:
            [B, LT, D]
        """
        if tgt_padding_mask is not None:
            tgt_padding_mask = tgt_padding_mask[:, None, None, :]
        if causal:
            causal_mask = nn.make_causal_mask(tgt[:, :, 0])
            tgt_mask = nn.combine_masks(tgt_padding_mask,
                                        causal_mask)
        else:
            tgt_mask = tgt_padding_mask
        if memory_padding_mask is not None:
            memory_mask = e.repeat(
                memory_padding_mask, "b l -> b 1 f l",
                f=tgt.shape[1])
        else:
            memory_mask = memory_padding_mask

        # LN -> Self Attention -> Residue
        # x: [B, L, D]
        x = nn.LayerNorm(name="dec_ln_1")(tgt)
        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_normal(),
            decode=self.decode,
            name="dec_attn_1"
        )(x, tgt_mask, deterministic=False)
        x = x + tgt
        assert x.shape == tgt.shape
        # LN -> Attention -> Residue
        y = nn.LayerNorm(name="dec_ln_2")(x)
        assert y.shape == tgt.shape
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_normal(),
            name="dec_attn_2",
        )(y, memory, memory_mask, deterministic=False)
        assert y.shape == tgt.shape
        y = y + x
        assert y.shape == tgt.shape
        # LN -> FF -> Residue
        z = nn.LayerNorm(name="dec_ln_3")(y)
        z = nn.Dense(self.ff_size,
                     kernel_init=nn.initializers.kaiming_normal(),
                     name="dec_ff_1")(z)
        z = nn.relu(z)
        if self.dropout_rate is not None:
            z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=False)
        z = nn.Dense(self.d_model,
                     kernel_init=nn.initializers.xavier_normal(),
                     name="dec_ff_2")(z)
        z = z + y
        assert z.shape == tgt.shape, f"Output shape {z.shape} != input shape {tgt.shape}"
        return z


class Transformer(nn.Module):
    d_model: int = 256
    n_enc_heads: int = 4
    n_enc_layers: int = 4
    n_dec_heads: int = 4
    n_dec_layers: int = 4
    ff_size: int = 2048
    dropout_rate: float = 0.0
    decode: bool = False

    @nn.compact
    def __call__(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        x = src
        for i in range(self.n_enc_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_enc_heads,
                ff_size=self.ff_size,
                dropout_rate=self.dropout_rate,
                name=f"transformer_enc_{i}"
            )(x, src_padding_mask)
        y = tgt
        for i in range(self.n_dec_layers):
            y = TransformerDecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_dec_heads,
                ff_size=self.ff_size,
                dropout_rate=self.dropout_rate,
                decode=self.decode,
                name=f"transformer_dec_{i}"
            )(x, y, src_padding_mask, tgt_padding_mask, True)
        return y
