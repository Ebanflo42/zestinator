import json
import string
import random
import numpy as np
import numpy.random as rd

import jax.numpy as jnp
import jax.random as jrd

from jax.example_libraries.optimizers import rmsprop
from jax import vmap, pmap, grad

from simmanager import SimManager
from functools import partial
from datetime import datetime
from absl import app, flags
from os.path import join as opj


def batch_covariance(x):
    x = x.reshape((x.shape[0], -1)) - jnp.mean(x)
    vmap(partial(jnp.dot, x))(jnp.flip(x, axis=0))


def encoder_loss(in_batch, out_batch):
    in_cov = batch_covariance(in_batch )
    out_cov = batch_covariance(out_batch)
    return 0.5*jnp.sum((in_cov - out_cov)**2)