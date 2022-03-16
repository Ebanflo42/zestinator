import json
import string
import random
import numpy as np
import numpy.random as rd

import jax.numpy as jnp
import jax.random as jrd

from jax.nn import logsumexp
from jax import vmap, pmap, value_and_grad, jit
from jax.example_libraries.optimizers import rmsprop

from simmanager import SimManager
from functools import partial
from datetime import datetime
from absl import app, flags
from os.path import join as opj

from models import *
from load_data import get_song_iterator
from utils import l2_norm_tree, plot_spectrogram, sim_save


FLAGS = flags.FLAGS

# system
flags.DEFINE_integer(
    'n_gpus', 1, 'How many GPUs we should try to use.')
flags.DEFINE_string(
    'model_name', '', 'If non-empty works as a special name for this model.')
flags.DEFINE_string('results_path', 'experiments',
                    'Name of the directory to save all results within.')

# training
flags.DEFINE_integer('max_steps', 1000,
                     'How many training batches to show the network.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('duration', 2400, 'Length of spectrogram.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('reg_coeff', 0.00001,
                   'Coefficient for L2 regularization of weights.')
flags.DEFINE_integer(
    'save_every', 500, 'Save the model and print metrics at this many training steps.')
flags.DEFINE_string('data_path', '/home/medusa/Data/fma_small',
                    'Path to metric fuck ton of numpy mel spectrograms.')
flags.DEFINE_bool('convex_combinations', False,
                  'Input convex combinations of the encoding into the decoder.')

# model
flags.DEFINE_string(
    'restore_from', '', 'If non-empty, restore the previous model from this directory and train it using the new flags.')
flags.DEFINE_string('restore_from_autoencoder', '',
                    'If non-empty, restore the encoder and decoder from the autoencoder trained at this path.')


def train_loop(sm, FLAGS, i, apply_encoder, encoder_params,
               apply_decoder, decoder_params, apply_discriminator, discriminator_params, song_iter):

    # construct the optimizer
    opt_initialize, opt_update, opt_get_params = rmsprop(FLAGS.lr)
    opt_state = opt_initialize((encoder_params, decoder_params))

    # define full forward pass from data to cross entropy loss
    # forward pass will be different depending on whether or not we want to do
    # operations on the representation (convex combinations)
    if FLAGS.convex_combinations:

        def forward_pass(rng, encparams, decparams, disparams, x):

            reg = FLAGS.reg_coeff * \
                l2_norm_tree((encparams, decparams, disparams))

            encoding = pmap(vmap(partial(apply_encoder, encparams)))(x)

            coeffs = jrd.uniform(
                rng, (FLAGS.n_gpus, FLAGS.batch_size//FLAGS.n_gpus, FLAGS.batch_size//FLAGS.n_gpus))
            coeffs /= jnp.sum(coeffs, axis=-1)

            def convcombine(m, enc):
                enc = jnp.moveaxis(enc, 0, -1)
                return jnp.moveaxis(jnp.dot(m, enc), -1, 0)

            new_encoding = pmap(convcombine)(coeffs, encoding)
            decoding = pmap(vmap(partial(apply_decoder, decparams)))(
                new_encoding)
            spectrograms = jnp.stack((x, decoding), axis=1)
            truths = jnp.stack(
                (jnp.ones_like(x)[:, :, 0, 0], jnp.zeros_like(decoding)[:, :, 0, 0]), axis=1)

            logits = pmap(
                vmap(partial(apply_discriminator, disparams)))(spectrograms)

            ce = jnp.sum(truths*logsumexp(logits))
            acc = jnp.mean(truths*jnp.heaviside(logits))
            loss = ce + reg

            return loss, (decoding, ce, acc, reg)

    else:

        def forward_pass(rng, encparams, decparams, disparams, x):

            reg = FLAGS.reg_coeff * \
                l2_norm_tree((encparams, decparams, disparams))

            encoding = pmap(vmap(partial(apply_encoder, encparams)))(x)

            decoding = pmap(vmap(partial(apply_decoder, decparams)))(encoding)
            spectrograms = jnp.stack((x, decoding), axis=1)
            truths = jnp.stack(
                (jnp.ones_like(x)[:, :, 0, 0], jnp.zeros_like(decoding)[:, :, 0, 0]), axis=1)

            logits = pmap(
                vmap(partial(apply_discriminator, disparams)))(spectrograms)

            ce = jnp.sum(truths*logsumexp(logits))
            acc = jnp.mean(truths*jnp.heaviside(logits))
            loss = ce + reg

            return loss, (decoding, ce, acc, reg)

    # derivative of forward pass with respect to parameters
    forward_backward_pass = value_and_grad(
        forward_pass, argnums=(0, 1), has_aux=True)

    # compile training step
    def train_step(x, eparams, dparams, ostate, ix):
        (loss, (y, mse, reg)), grads = forward_backward_pass(eparams, dparams, x)
        new_opt_state = opt_update(ix, grads, ostate)
        new_eparams, new_dparams = opt_get_params(new_opt_state)
        return y, mse, new_eparams, new_dparams, new_opt_state
    jit_train_step = jit(train_step)

    # track mse loss
    mse_log = []

    # begin training loop
    while i < FLAGS.max_steps:

        # get next training sample
        x = next(song_iter)
        jx = jnp.asarray(x)

        y, mse, encoder_params, decoder_params, opt_state = jit_train_step(
            jx, encoder_params, decoder_params, opt_state, i)

        if i > 0 and i % FLAGS.save_every == 0:

            # record metrics
            mse_log.append(mse.item())

            sim_save(sm, 'mse_log', mse_log)
            sim_save(sm, 'iteration', i)

            save_triple_gru(sm, encoder_params, component='encoder')
            save_triple_gru(sm, decoder_params, component='decoder')

            plot_spectrogram(sm, x[0].T, i, 'original')
            plot_spectrogram(sm, np.array(y[0]).T, i, 'decoder')

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f'{timestamp} Saved {sm.sim_name} at iteration {i}.')

            print(
                f'\tMSE: {mse_log[-1]}\n')

        i += 1

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'{timestamp} Finished training.')

    print(f'Final save of {sm.sim_name}.')
    save_triple_gru(sm, encoder_params, component='encoder')
    save_triple_gru(sm, decoder_params, component='decoder')
    sim_save(sm, 'mse_log', mse_log)
    sim_save(sm, 'iteration', i)

    date = datetime.now().strftime('%d-%m-%Y')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'Training {sm.sim_name} ended on {date} at {timestamp}.')


def main(_argv):

    # construct simulation manager
    if FLAGS.model_name == '':
        identifier = ''.join(random.choice(
            string.ascii_lowercase + string.digits) for _ in range(4))
        sim_name = 'autoencoder_{}'.format(identifier)
    else:
        sim_name = FLAGS.model_name
    date = datetime.now().strftime('%d-%m-%Y')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'\nBeginning to train {sim_name} on {date} at {timestamp}.\n')
    sm = SimManager(
        sim_name, FLAGS.results_path, write_protect_dirs=False, tee_stdx_to='output.log')

    with sm:

        # dump FLAGS for this experiment
        with open(opj(sm.paths.data_path, 'FLAGS.json'), 'w') as f:
            flag_dict = {}
            for k in FLAGS._flags().keys():
                if k not in FLAGS.__dict__['__hiddenflags']:
                    flag_dict[k] = FLAGS.__getattr__(k)
            json.dump(flag_dict, f)

        # generate, save, and set random seed
        random_seed = datetime.now().microsecond
        np.save(opj(sm.paths.data_path, 'random_seed'), random_seed)
        rd.seed(random_seed)
        random.seed(random_seed)
        rng = jrd.PRNGKey(random_seed)

        # get model functions
        init_encoder, apply_encoder = encoder()
        init_decoder, apply_decoder = decoder()
        init_discriminator, apply_discriminator = discriminator()

        # restore if necessary
        if FLAGS.restore_from != '':
            encoder_params = load_triple_gru(
                FLAGS.restore_from, component='encoder')
            decoder_params = load_triple_gru(
                FLAGS.restore_from, component='decoder')
            discriminator_params = load_discriminator(FLAGS.restore_from)
            i = np.load(opj(FLAGS.restore_from, 'results', 'iteration.npy'))
        elif FLAGS.restor_from_autoencoder != '':
            encoder_params = load_triple_gru(
                FLAGS.restore_from, component='encoder')
            decoder_params = load_triple_gru(
                FLAGS.restore_from, component='decoder')
            discriminator_params = init_discriminator(rng)
        else:
            encoder_params = init_encoder(rng)
            decoder_params = init_decoder(rng)
            discriminator_params = init_discriminator(rng)
            i = 0

        song_iter = get_song_iterator(FLAGS)

        train_loop(sm, FLAGS, i, apply_encoder, encoder_params,
                   apply_decoder, decoder_params, apply_discriminator, discriminator_params, song_iter)


if __name__ == '__main__':

    app.run(main)
