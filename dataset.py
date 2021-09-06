import jax
from jax import lax, random, numpy as jnp


def get_batch(rkey: random.PRNGKey, batch_size: int):
    """ Generates one batch from a random key
        Returns:
            [B, L=32]
    """
    return random.randint(rkey, (batch_size, 32), 0, 26)
    