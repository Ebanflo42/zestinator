import json
import string
import random
import numpy as np
import numpy.random as rd

import jax.numpy as jnp
import jax.random as jrd

from jax import vmap, pmap

from functools import partial
from os.path import join as opj


def sim_save(sm, name, array):
    np.save(opj(sm.paths.results_path, name), array)


def unfold_convolution(array, win, hop, axis=0):
    array = jnp.moveaxis(array, axis, 0)
    first = jnp.repeat(array[:-1], hop, axis=0)
    second = jnp.repeat(array[-1].unsqueeze(0), win, axis=0)
    together = jnp.concatenate([first, second], axis=0)
    return jnp.moveaxis(together, 0, axis)
