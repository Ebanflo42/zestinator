import jax.numpy as jnp

from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal

from functools import partial
from jax import random
from jax import lax


def gru(out_dim, W_init=glorot_normal(), b_init=normal()):

    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax """

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
                (update_W, update_U, update_b),
                (reset_W, reset_U, reset_b),
                (out_W, out_U, out_b))

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def dynamic_step(params, hidden, inp):
            """ Perform single step update of the network """
            (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b) = params

            update_gate = sigmoid(jnp.dot(inp, update_W) +
                                  jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(inp, reset_W) +
                                 jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(inp, out_W)
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U)
                                   + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1-update_gate, output_gate)
            hidden = output
            return hidden, hidden

        # Move the time dimension to position 0
        inputs = jnp.moveaxis(inputs, 1, 0)
        f = partial(dynamic_step, params)
        _, hiddens = lax.scan(f, h, inputs)

        return hiddens

    return init_fun, apply_fun


def recursive_gru(out_dim, n_iters, W_init=glorot_normal(), b_init=normal()):

    def init_fun(rng, input_shape):
        """ Initialize the GRU layer for stax """

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], n_iters, out_dim)
        return (output_shape,
                (update_W, update_U, update_b),
                (reset_W, reset_U, reset_b),
                (out_W, out_U, out_b))

    def apply_fun(params, input, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def dynamic_step(params, hidden):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b) = params

            update_gate = sigmoid(jnp.dot(input, update_W) +
                                  jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(input, reset_W) +
                                 jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(input, out_W)
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U)
                                   + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1-update_gate, output_gate)
            hidden = output

            return hidden

        # initialize hidden states
        states = jnp.zeros(
            (input.shape[0], n_iters, input.shape[1]), dtype=jnp.float32)
        # iterate GRU on itself for n_iters
        def iteration(t, s):
            return s.at[:, t].set(dynamic_step(params, s.at[:, t - 1].get()))
        states = lax.fori_loop(1, n_iters, iteration, states)

        return states

    return init_fun, apply_fun
