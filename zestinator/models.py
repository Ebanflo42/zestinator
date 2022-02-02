import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrd

from jax.nn import sigmoid
from jax.lax import conv_general_dilated
from jax.nn.initializers import glorot_normal, normal

from utils import sim_save, unfold_convolution


def gru(out_dim: int, W_init=glorot_normal(), b_init=normal()):

    # initializer
    def init_fun(rng, input_shape):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        out_shape = (input_shape[0], out_dim)

        return out_shape, ((update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b))

    # apply model
    def apply_fun(params, inputs, **kwargs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(jnp.dot(update_W, inp) + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(reset_W, inp) + jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(out_W, inp)
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
    def init_fun(rng, input_shape):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (win, input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (win, input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (win, input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        out_shape = ((input_shape[0] - win)//hop + 1, out_dim)

        return out_shape, ((update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b))

    # apply model
    def apply_fun(params, inputs, **kwargs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # run convolutions on input
        update_conv = conv_general_dilated(
            inputs, update_W, window_strides=[hop], padding='VALID')
        reset_conv = conv_general_dilated(
            inputs, reset_W, window_strides=[hop], padding='VALID')
        out_conv = conv_general_dilated(
            inputs, out_W, window_strides=[hop], padding='VALID')
        stacked_conv = jnp.stack([update_conv, reset_conv, out_conv], axis=1)

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(inp[0] + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(inp[1] + jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(inp[2]
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U) + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1 - update_gate, output_gate)

            return output, output


        # run GRU over convolved input
        _, hiddens = lax.scan(dynamic_step, stacked_conv)

        return hiddens

    return init_fun, apply_fun


def repeating_gru(out_dim: int, win: int, hop: int, W_init=glorot_normal(), b_init=normal()):
    """
    GRU which unrolls the input as if it had been convolved by the given window and hop.
    """

    # initializer
    def init_fun(rng, input_shape):

        k1, k2, k3 = jrd.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = jrd.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[1], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        out_shape = (hop*(input_shape[0] - 1) + win, out_dim)

        return out_shape, ((update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b))

    # apply model
    def apply_fun(params, inputs, **kwargs):

        # unwrap parameters
        (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = params

        # unroll convolutions
        unfolded_inputs = unfold_convolution(inputs, win, hop)

        # define dynamic GRU step
        def dynamic_step(hidden, inp):

            update_gate = sigmoid(jnp.dot(update_W, inp) + jnp.dot(hidden, update_U) + update_b)
            reset_gate = sigmoid(jnp.dot(reset_W, inp) + jnp.dot(hidden, reset_U) + reset_b)
            output_gate = jnp.tanh(jnp.dot(out_W, inp)
                                   + jnp.dot(jnp.multiply(reset_gate,
                                             hidden), out_U) + out_b)
            output = jnp.multiply(update_gate, hidden) + \
                jnp.multiply(1 - update_gate, output_gate)

            return output, output


        # run GRU over unrolled input
        _, hiddens = lax.scan(dynamic_step, unfolded_inputs)

        return hiddens

    return init_fun, apply_fun


def encoder(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = convolutional_gru(256, 6, 2, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = convolutional_gru(128, 6, 2, W_init=W_init, b_init=b_init)
    l3_init, l3_apply = convolutional_gru(64, 6, 5, W_init=W_init, b_init=b_init)

    def init_fun(rng, input_shape):

        l1_shape, l1_params = l1_init(rng, input_shape)
        l2_shape, l2_params = l2_init(rng, l1_shape)
        l3_shape, l3_params = l3_init(rng, l2_shape)

        return l1_params, l2_params, l3_params

    def apply_fun(params, inputs):

        l1_params, l2_params, l3_params = params

        l1_output = l1_apply(l1_params, inputs)
        l2_output = l2_apply(l2_params, l1_output)
        l3_output = l3_apply(l3_params, l2_output)

        return l3_output

    return init_fun, apply_fun


def decoder(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = convolutional_gru(128, 6, 5, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = convolutional_gru(256, 6, 2, W_init=W_init, b_init=b_init)
    l3_init, l3_apply = convolutional_gru(512, 6, 2, W_init=W_init, b_init=b_init)

    def init_fun(rng, input_shape):

        l1_shape, l1_params = l1_init(rng, input_shape)
        l2_shape, l2_params = l2_init(rng, l1_shape)
        l3_shape, l3_params = l3_init(rng, l2_shape)

        return l1_params, l2_params, l3_params

    def apply_fun(params, inputs):

        l1_params, l2_params, l3_params = params

        l1_output = l1_apply(l1_params, inputs)
        l2_output = l2_apply(l2_params, l1_output)
        l3_output = l3_apply(l3_params, l2_output)

        return l3_output

    return init_fun, apply_fun


def discriminator(W_init=glorot_normal(), b_init=normal()):

    l1_init, l1_apply = gru(288, W_init=W_init, b_init=b_init)
    l2_init, l2_apply = gru(144, W_init=W_init, b_init=b_init)

    def init_fun(rng, input_shape):

        l1_shape, l1_params = l1_init(rng, input_shape)
        l2_shape, l2_params = l2_init(rng, l1_shape)

        k1, k2 = jrd.split(rng, num=2)
        out_W = W_init(k1, (144, 2))
        out_b = b_init(k2, 2)

        return l1_params, l2_params, (out_W, out_b)

    def apply_fun(params, representation, spectrogram):

        l1_params, l2_params, (out_W, out_b) = params

        # unrolling all 3 convolutions should be the same as unrolling
        # one with window 36 and stride 20
        unfolded_representation = unfold_convolution(representation, 36, 20)

        # discriminator receives spectrogram and time-dependent representation as input
        inputs = jnp.stack([representation, spectrogram], axis=1)

        l1_output = l1_apply(l1_params, inputs)
        l2_output = l2_apply(l2_params, l1_output)
        output = jnp.dot(out_W, l2_output) + out_b

        return output
