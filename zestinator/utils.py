import json
import string
import random
import numpy as np
import jax.numpy as jnp

from os.path import join as opj


def sim_save(sm, name, array):
    np.save(opj(sm.paths.results_path, name), array)


def unfold_convolution(array, win, hop, axis=0):
    array = jnp.moveaxis(array, axis, 0)
    first = jnp.repeat(array[:-1], hop, axis=0)
    second = jnp.repeat(array[-1][jnp.newaxis], win, axis=0)
    together = jnp.concatenate([first, second], axis=0)
    return jnp.moveaxis(together, 0, axis)
