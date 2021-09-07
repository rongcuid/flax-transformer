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


@jax.jit
def compute_metrics(idxs_tgt, idxs_pred, logits_tgt, logits_pred):
    return {
        "loss":
            jnp.mean(optax.softmax_cross_entropy(
                logits_tgt, logits_pred)),
        "accuracy": (idxs_tgt == idxs_pred).sum() / float(idxs_tgt.size),
    }


def train_step(state, batch):
    labels = idx2onehot(batch)
    # @jax.jit
    def loss_fn(params):
        idxs, logits = \
            MemCpy().apply({"params": params}, labels)
        loss = optax.softmax_cross_entropy(logits, labels)
        loss = jnp.mean(loss)
        return loss, (idxs, logits)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (idxs_pred, logits)), grads = \
        grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, \
        compute_metrics(batch, idxs_pred, labels, logits)


def val_step(params, batch):
    labels = idx2onehot(batch)
    idxs_pred, logits = MemCpy(decode=False).apply(
        {"params": params},
        labels)
    return idxs_pred, \
        compute_metrics(batch, idxs_pred, labels, logits)


def train(batch_size: int, steps: int, rkey=random.PRNGKey(0),
          output_dir=None):
    rng, rng_init = random.split(rkey)
    state = create_train_state(rng_init, 0.001)
    del rng_init
    for i in range(steps):
        rng, key_batch = random.split(rng)
        batch = get_batch(key_batch, batch_size)
        state, metric = train_step(state, batch)
        print(f"{i+1}/{steps} Loss: {metric['loss']}; accuracy: {metric['accuracy']}")
    return state


def validate(params, rng, batch_size):
    rng, rng_batch = random.split(rng)
    batch = get_batch(rng_batch, batch_size)
    strs = [idx2str(batch[i, :]) for i in range(batch.shape[0])]
    idxs_pred, metric = val_step(params, batch)
    strs_pred = [
        idx2str(idxs_pred[i, :])
        for i in range(idxs_pred.shape[0])]
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
