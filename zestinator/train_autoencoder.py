import json
import string
import random
from matplotlib.pyplot import plot
import numpy as np
import numpy.random as rd

import jax.numpy as jnp
import jax.random as jrd

from jax import vmap, pmap, value_and_grad, jit
from jax.example_libraries.optimizers import rmsprop

from simmanager import SimManager
from functools import partial
from datetime import datetime
from absl import app, flags
from os.path import join as opj

from load_data import get_song_iterator
from utils import l2_norm_tree, plot_spectrogram, sim_save
from models import encoder, decoder, save_triple_gru, load_triple_gru


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
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('duration', 1, 'Length of track clip in seconds.')
flags.DEFINE_integer('maxlen', 25, 'Cut each song to this length.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('reg_coeff', 0.00001,
                   'Coefficient for L2 regularization of weights.')
flags.DEFINE_integer(
    'save_every', 500, 'Save the model and print metrics at this many training steps.')
flags.DEFINE_string('data_path', '/home/medusa/Data/fma_small',
                    'Path to metric fuck ton of mp3s.')

# model
flags.DEFINE_string(
    'restore_from', '', 'If non-empty, restore the previous model from this directory and train it using the new flags.')


def train_loop(sm, FLAGS, i, apply_encoder, encoder_params,
                   apply_decoder, decoder_params, song_iter):

    # construct the optimizer
    opt_initialize, opt_update, opt_get_params = rmsprop(FLAGS.lr)
    opt_state = opt_initialize((encoder_params, decoder_params))

    # define full forward pass from data to MSE Loss
    def forward_pass(eparams, dparams, x):
        reg = FLAGS.reg_coeff*l2_norm_tree((eparams, dparams))
        f = lambda x1: apply_decoder(dparams, apply_encoder(eparams, x1))
        decoding = vmap(f)(x)
        mse = jnp.mean((x - decoding)**2)
        loss = mse + reg
        return loss, (decoding, mse, reg)

    # derivative of forward pass with respect to parameters
    forward_backward_pass = value_and_grad(forward_pass, argnums=(0, 1), has_aux=True)

    # compile training step
    def train_step(x, eparams, dparams, ostate, ix):
        (loss, (y, mse, reg)), grads = forward_backward_pass(eparams, dparams, x)
        new_opt_state = opt_update(ix, grads, ostate)
        new_eparams, new_dparams = opt_get_params(new_opt_state)
        return y, mse, new_eparams, new_dparams, new_opt_state
    jit_train_step = jit(train_step)

    # track some metrics
    mse_log = []
    #r2_log = []

    # begin training loop
    while i < FLAGS.max_steps:

        # get next training sample
        x = next(song_iter)
        jx = jnp.asarray(x)
        print(f'Got sample! {x.shape}')

<<<<<<< HEAD
        y, mse, encoder_params, decoder_params, opt_state = jit_train_step(x, encoder_params, decoder_params, opt_state, i)
=======
        # forward pass and backward pass
        (loss, (y, mse, reg)), grads = forward_backward_pass(encoder_params, decoder_params, jx)

        # optimizer step
        opt_state = opt_update(i, grads, opt_state)
        encoder_params, decoder_params = opt_get_params(opt_state)
>>>>>>> 71b754fbdbf23ab942a77f80c0bce20f4b3fb152

        if i > 0 and i % FLAGS.save_every == 0:

            # record metrics
            mse_log.append(mse.item())
            #r2 = 1 - jnp.sum(jnp.var(y, axis=(2, 3))/jnp.var(x, axis=(2, 3)))
            #r2_log.append(r2.item())

            sim_save(sm, 'mse_log', mse_log)
            #sim_save(sm, 'r2_log', r2_log)
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
    #sim_save(sm, 'r2_log', r2_log)
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

        # restore if necessary
        if FLAGS.restore_from != '':
            encoder_params = load_triple_gru(
                FLAGS.restore_from, component='encoder')
            decoder_params = load_triple_gru(
                FLAGS.restore_from, component='decoder')
            i = np.load(opj(FLAGS.restore_from, 'results', 'iteration.npy'))
        else:
            encoder_params = init_encoder(rng)
            decoder_params = init_decoder(rng)
            i = 0

        song_iter = get_song_iterator(FLAGS)

        train_loop(sm, FLAGS, i, apply_encoder, encoder_params,
                   apply_decoder, decoder_params, song_iter)


if __name__ == '__main__':

    app.run(main)
