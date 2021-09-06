import argparse
from network_n2 import *
from pathlib import Path
from tqdm import trange

import jax
from jax import lax, random, numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

from dataset import get_batch
from utils import dir_path


def create_train_state(rng, lr):
    model = MemCpy(decode=False)
    params = model.init(rng, jnp.zeros((1, 32, 26)))["params"]
    tx = optax.adamw(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def compute_metrics(idxs_tgt, idxs_pred, logits_tgt, logits_pred):
    return {
        "loss":
            jnp.mean(optax.softmax_cross_entropy(
                logits_tgt, logits_pred)),
        "accuracy": (idxs_tgt == idxs_pred).sum() / float(idxs_tgt.size),
    }

# @jax.jit


def train_step(state, batch):
    labels = idx2onehot(batch)

    def loss_fn(params):
        strs, idxs, logits = \
            MemCpy().apply({"params": params}, labels)
        loss = optax.softmax_cross_entropy(logits, labels)
        loss = jnp.mean(loss)
        return loss, (strs, idxs, logits)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (strs, idxs_pred, logits)), grads = \
        grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, strs, \
        compute_metrics(batch, idxs_pred, labels, logits)


# @jax.jit
def val_step(params, batch):
    labels = idx2onehot(batch)
    strs, idxs_pred, logits = MemCpy(decode=False).apply(
        {"params": params},
        labels)
    return strs, compute_metrics(batch, idxs_pred, labels, logits)


def train(batch_size: int, steps: int, rkey=random.PRNGKey(0),
          output_dir=None):
    rng, rng_init = random.split(rkey)
    state = create_train_state(rng_init, 0.001)
    del rng_init
    for i in trange(steps):
        rng, key_batch = random.split(rng)
        batch = get_batch(key_batch, batch_size)
        state, _, metric = train_step(state, batch)
        if i % 10 == 9:
            print(f"Loss: {metric['loss']}; accuracy: {metric['accuracy']}")
    return state


def validate(params, rng, batch_size):
    rng, rng_batch = random.split(rng)
    batch = get_batch(rng_batch, batch_size)
    strs = [idx2str(batch[i, :]) for i in range(batch.shape[0])]
    strs_pred, metric = val_step(params, batch)
    print(f"Loss: {metric['loss']}; accuracy: {metric['accuracy']}")
    for tgt, pred in zip(strs, strs_pred):
        print(f"EXPECT : {tgt}")
        print(f"PREDICT: {pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, help="Number of steps to train")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("output", type=dir_path, help="Output directory")
    args = parser.parse_args()
    rng = random.PRNGKey(0)
    rng, train_rng, val_rng = random.split(rng, 3)
    state = train(args.batch_size, args.steps, output_dir=args.output)
    validate(state.params, val_rng, args.batch_size)
