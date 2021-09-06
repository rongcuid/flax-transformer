"""Memcpy with O(n^2) Pre-LN Transformer"""

from modules import RotaryPositionalEmbedding
import jax
from jax import random, lax, numpy as jnp
import flax
import flax.linen as nn
import einops as e

from modules_n2 import *


def str2onehot(s: str) -> jnp.ndarray:
    """ String to one-hot lower case alphabet
        Args:
            s: Input string
        Returns:
            one-hot array: [L, 26]
            idx array: [L]
    """
    idxs = jnp.array([
        ord(c) - ord('a') for c in s
    ])
    return idx2onehot(idxs), idxs

@jax.jit
def idx2onehot(idxs: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.one_hot(idxs, 26)

@jax.jit
def onehot2idx(a: jnp.ndarray) -> Tuple[str, jnp.ndarray]:
    """ One-hot lower case alphabet to string
        Args:
            a: [L, 26]
        Returns:
            str, index array
    """
    idx = jnp.argmax(a, -1)
    return idx


def idx2str(idx):
    cs = idx + ord('a')
    return "".join([chr(c) for c in cs])


class MemCpy(nn.Module):
    d_model: int = 32
    decode: bool = False
    decoder_hid: int = 128

    @nn.compact
    def __call__(self, input) -> Tuple[List[str], jnp.ndarray]:
        """
        Memcpy model
        Args:
            input: [B, L, 26]
        Returns:
            predicted strings: List[str]
            one-hot logit arrays: [B, L, 26]
        """
        assert len(input.shape) == 3
        inp = nn.Dense(
            self.d_model,
            name="input_proj")(input)
        # Position
        x = RotaryPositionalEmbedding(
            name="positional_enc"
        )(inp)
        y = RotaryPositionalEmbedding(
            name="positional_dec"
        )(inp)
        # Transformer
        z = Transformer(
            d_model=self.d_model,
            decode=self.decode,
            name="transformer"
        )(x, y)
        # Output
        z = nn.Dense(self.decoder_hid,
                     name="decoder_fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(26, 
            name="decoder_fc2")(z)
        # Conversion
        return onehot2idx(z), z
