import numpy as np
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

from jax.nn import sigmoid
from jax.lax import conv_general_dilated
from jax.nn.initializers import glorot_normal, normal

from os.path import join as opj

from utils import sim_save, unfold_convolution


def gru(out_dim: int, W_init=glorot_normal(), b_init=normal()):

    # initializer
    def init_fun(rng, in_dim):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        return (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b)

    # apply model
    def apply_fun(params, inputs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(
                jnp.dot(inp, update_W) + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(inp, reset_W) +
                                 jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(inp, out_W)
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U) + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1 - update_gate, output_gate)

            return output, output

        # run GRU over input
        _, hiddens = lax.scan(dynamic_step, inputs)

        return hiddens

    return init_fun, apply_fun


def convolutional_gru(out_dim: int, win: int, hop: int, W_init=glorot_normal(), b_init=normal()):
    """
    GRU where the input weights are actually a convolution along the time axis
    """

    # initializer
    def init_fun(rng, in_dim):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (out_dim, in_dim, win)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (out_dim, in_dim, win)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (out_dim, in_dim, win)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        return (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b)

    # apply model
    def apply_fun(params, inputs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # run convolutions on input
        inputs = jnp.reshape(jnp.moveaxis(inputs, 1, 0),
                             (1, inputs.shape[1], inputs.shape[0]))
        update_conv = conv_general_dilated(
            inputs, update_W, window_strides=[hop], padding='SAME')
        reset_conv = conv_general_dilated(
            inputs, reset_W, window_strides=[hop], padding='SAME')
        out_conv = conv_general_dilated(
            inputs, out_W, window_strides=[hop], padding='SAME')
        stacked_conv = jnp.concatenate(
            [update_conv, reset_conv, out_conv], axis=0)
        stacked_conv = jnp.transpose(stacked_conv, (2, 0, 1))

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(
                inp[0] + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(inp[1] + jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(inp[2]
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U) + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1 - update_gate, output_gate)

            return output, output

        # run GRU over convolved input
        _, hiddens = lax.scan(dynamic_step, jnp.zeros_like(
            stacked_conv[0, 0]), stacked_conv)

        return hiddens

    return init_fun, apply_fun


def repeating_gru(out_dim: int, win: int, hop: int, W_init=glorot_normal(), b_init=normal()):
    """
    GRU which unrolls the input as if it had been convolved by the given window and hop.
    """

    # initializer
    def init_fun(rng, in_dim):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (in_dim, out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        return (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b)

    # apply model
    def apply_fun(params, inputs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # unroll convolutions
        unfolded_inputs = unfold_convolution(inputs, win, hop)

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(
                jnp.dot(inp, update_W) + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(inp, reset_W) +
                                 jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(inp, out_W)
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U) + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1 - update_gate, output_gate)

            return output, output

        # run GRU over unrolled input
        _, hiddens = lax.scan(dynamic_step, jnp.zeros(
            out_dim, dtype=jnp.float32), unfolded_inputs)

        return hiddens

    return init_fun, apply_fun


def encoder(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = convolutional_gru(
        64, 6, 2, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = convolutional_gru(
        32, 6, 2, W_init=W_init, b_init=b_init)
    l3_init, l3_apply = convolutional_gru(
        16, 6, 5, W_init=W_init, b_init=b_init)

    def init_fun(rng):

        l1_params = l1_init(rng, 128)
        l2_params = l2_init(rng, 64)
        l3_params = l3_init(rng, 32)

        return l1_params, l2_params, l3_params

    def apply_fun(params, inputs):

        l1_params, l2_params, l3_params = params

        l3_output = l3_apply(l3_params, l2_apply(
            l2_params, l1_apply(l1_params, inputs)))

        return l3_output

    return init_fun, apply_fun


def decoder(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = repeating_gru(
        32, 6, 5, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = repeating_gru(
        64, 6, 2, W_init=W_init, b_init=b_init)
    l3_init, l3_apply = repeating_gru(
        128, 6, 2, W_init=W_init, b_init=b_init)

    def init_fun(rng):

        l1_params = l1_init(rng, 16)
        l2_params = l2_init(rng, 32)
        l3_params = l3_init(rng, 64)

        return l1_params, l2_params, l3_params

    def apply_fun(params, inputs):

        l1_params, l2_params, l3_params = params

        l3_output = l3_apply(l3_params, l2_apply(
            l2_params, l1_apply(l1_params, inputs)))

        return l3_output

    return init_fun, apply_fun


def save_triple_gru(sm, params, component='encoder'):

    l1_params, l2_params, l3_params = params
    (l1_update_W, l1_update_U, l1_update_b), (l1_reset_W, l1_reset_U, l1_reset_b), (
        l1_out_W, l1_out_U, l1_out_b) = l1_params
    (l2_update_W, l2_update_U, l2_update_b), (l2_reset_W, l2_reset_U, l2_reset_b), (
        l2_out_W, l2_out_U, l2_out_b) = l2_params
    (l3_update_W, l3_update_U, l3_update_b), (l3_reset_W, l3_reset_U, l3_reset_b), (
        l3_out_W, l3_out_U, l3_out_b) = l3_params

    sim_save(sm, f'{component}_l1_update_W', l1_update_W)
    sim_save(sm, f'{component}_l1_update_U', l1_update_U)
    sim_save(sm, f'{component}_l1_update_b', l1_update_b)
    sim_save(sm, f'{component}_l1_reset_W', l1_reset_W)
    sim_save(sm, f'{component}_l1_reset_U', l1_reset_U)
    sim_save(sm, f'{component}_l1_reset_b', l1_reset_b)
    sim_save(sm, f'{component}_l1_out_W', l1_out_W)
    sim_save(sm, f'{component}_l1_out_U', l1_out_U)
    sim_save(sm, f'{component}_l1_out_b', l1_out_b)

    sim_save(sm, f'{component}_l2_update_W', l2_update_W)
    sim_save(sm, f'{component}_l2_update_U', l2_update_U)
    sim_save(sm, f'{component}_l2_update_b', l2_update_b)
    sim_save(sm, f'{component}_l2_reset_W', l2_reset_W)
    sim_save(sm, f'{component}_l2_reset_U', l2_reset_U)
    sim_save(sm, f'{component}_l2_reset_b', l2_reset_b)
    sim_save(sm, f'{component}_l2_out_W', l2_out_W)
    sim_save(sm, f'{component}_l2_out_U', l2_out_U)
    sim_save(sm, f'{component}_l2_out_b', l2_out_b)

    sim_save(sm, f'{component}_l3_update_W', l3_update_W)
    sim_save(sm, f'{component}_l3_update_U', l3_update_U)
    sim_save(sm, f'{component}_l3_update_b', l3_update_b)
    sim_save(sm, f'{component}_l3_reset_W', l3_reset_W)
    sim_save(sm, f'{component}_l3_reset_U', l3_reset_U)
    sim_save(sm, f'{component}_l3_reset_b', l3_reset_b)
    sim_save(sm, f'{component}_l3_out_W', l3_out_W)
    sim_save(sm, f'{component}_l3_out_U', l3_out_U)
    sim_save(sm, f'{component}_l3_out_b', l3_out_b)


def load_triple_gru(path, component='encoder'):

    l1_update_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_update_W')))
    l1_update_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_update_U')))
    l1_update_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_update_b')))
    l1_reset_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_reset_W')))
    l1_reset_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_reset_U')))
    l1_reset_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_reset_b')))
    l1_out_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_out_W')))
    l1_out_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_out_U')))
    l1_out_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l1_out_b')))
    l1_params = (l1_update_W, l1_update_U, l1_update_b), (l1_reset_W, l1_reset_U, l1_reset_b), (
        l1_out_W, l1_out_U, l1_out_b)

    l2_update_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_update_W')))
    l2_update_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_update_U')))
    l2_update_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_update_b')))
    l2_reset_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_reset_W')))
    l2_reset_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_reset_U')))
    l2_reset_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_reset_b')))
    l2_out_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_out_W')))
    l2_out_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_out_U')))
    l2_out_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l2_out_b')))
    l2_params = (l2_update_W, l2_update_U, l2_update_b), (l2_reset_W, l2_reset_U, l2_reset_b), (
        l2_out_W, l2_out_U, l2_out_b)

    l3_update_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_update_W')))
    l3_update_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_update_U')))
    l3_update_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_update_b')))
    l3_reset_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_reset_W')))
    l3_reset_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_reset_U')))
    l3_reset_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_reset_b')))
    l3_out_W = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_out_W')))
    l3_out_U = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_out_U')))
    l3_out_b = jnp.ndarray(
        np.load(opj(path, 'results', f'{component}_l3_out_b')))
    l3_params = (l3_update_W, l3_update_U, l3_update_b), (l3_reset_W, l3_reset_U, l3_reset_b), (
        l3_out_W, l3_out_U, l3_out_b)

    return l1_params, l2_params, l3_params


def discriminator(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = gru(72, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = gru(36, W_init=W_init, b_init=b_init)

    def init_fun(rng, in_dim):

        l1_shape, l1_params = l1_init(rng, in_dim)
        l2_shape, l2_params = l2_init(rng, l1_shape)

        k1, k2 = jrd.split(rng, num=2)
        out_W = W_init(k1, (72, 2))
        out_b = b_init(k2, 2)

        return l1_params, l2_params, (out_W, out_b)

    def apply_fun(params, representation, spectrogram):

        l1_params, l2_params, (out_W, out_b) = params

        # unrolling all 3 convolutions should be the same as unrolling
        # one with window 36 and stride 20
        unfolded_representation = unfold_convolution(representation, 36, 20)

        # discriminator receives spectrogram and time-dependent representation as input
        inputs = jnp.stack([unfolded_representation, spectrogram], axis=1)

        l1_output = l1_apply(l1_params, inputs)
        l2_output = l2_apply(l2_params, l1_output)
        output = jnp.dot(l2_output[-1], out_W) + out_b

        return output

    return init_fun, apply_fun


def save_discriminator(sm, params):

    l1_params, l2_params, (out_W, out_b) = params
    (l1_update_W, l1_update_U, l1_update_b), (l1_reset_W, l1_reset_U, l1_reset_b), (
        l1_out_W, l1_out_U, l1_out_b) = l1_params
    (l2_update_W, l2_update_U, l2_update_b), (l2_reset_W, l2_reset_U, l2_reset_b), (
        l2_out_W, l2_out_U, l2_out_b) = l2_params

    sim_save(sm, 'discriminator_l1_update_W', l1_update_W)
    sim_save(sm, 'discriminator_l1_update_U', l1_update_U)
    sim_save(sm, 'discriminator_l1_update_b', l1_update_b)
    sim_save(sm, 'discriminator_l1_reset_W', l1_reset_W)
    sim_save(sm, 'discriminator_l1_reset_U', l1_reset_U)
    sim_save(sm, 'discriminator_l1_reset_b', l1_reset_b)
    sim_save(sm, 'discriminator_l1_out_W', l1_out_W)
    sim_save(sm, 'discriminator_l1_out_U', l1_out_U)
    sim_save(sm, 'discriminator_l1_out_b', l1_out_b)

    sim_save(sm, 'discriminator_l2_update_W', l2_update_W)
    sim_save(sm, 'discriminator_l2_update_U', l2_update_U)
    sim_save(sm, 'discriminator_l2_update_b', l2_update_b)
    sim_save(sm, 'discriminator_l2_reset_W', l2_reset_W)
    sim_save(sm, 'discriminator_l2_reset_U', l2_reset_U)
    sim_save(sm, 'discriminator_l2_reset_b', l2_reset_b)
    sim_save(sm, 'discriminator_l2_out_W', l2_out_W)
    sim_save(sm, 'discriminator_l2_out_U', l2_out_U)
    sim_save(sm, 'discriminator_l2_out_b', l2_out_b)

    sim_save(sm, 'discriminator_out_W', out_W)
    sim_save(sm, 'discriminator_out_b', out_b)


def load_discriminator(path):

    l1_update_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_update_W')))
    l1_update_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_update_U')))
    l1_update_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_update_b')))
    l1_reset_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_reset_W')))
    l1_reset_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_reset_U')))
    l1_reset_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_reset_b')))
    l1_out_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_out_W')))
    l1_out_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_out_U')))
    l1_out_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l1_out_b')))
    l1_params = (l1_update_W, l1_update_U, l1_update_b), (l1_reset_W, l1_reset_U, l1_reset_b), (
        l1_out_W, l1_out_U, l1_out_b)

    l2_update_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_update_W')))
    l2_update_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_update_U')))
    l2_update_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_update_b')))
    l2_reset_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_reset_W')))
    l2_reset_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_reset_U')))
    l2_reset_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_reset_b')))
    l2_out_W = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_out_W')))
    l2_out_U = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_out_U')))
    l2_out_b = jnp.ndarray(
        np.load(opj(path, 'results', 'discriminator_l2_out_b')))
    l2_params = (l2_update_W, l2_update_U, l2_update_b), (l2_reset_W, l2_reset_U, l2_reset_b), (
        l2_out_W, l2_out_U, l2_out_b)

    out_W = jnp.ndarray(np.load(opj(path, 'results', 'discriminator_out_W')))
    out_b = jnp.ndarray(np.load(opj(path, 'results', 'discriminator_out_b')))

    return l1_params, l2_params, (out_W, out_b)
