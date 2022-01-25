import json
import string
import random
import numpy as np
import numpy.random as rd

import jax.numpy as jnp
import jax.random as jrd

from jax.example_libraries.optimizers import rmsprop
from jax import vmap, pmap, value_and_grad

from simmanager import SimManager
from functools import partial
from datetime import datetime
from absl import app, flags
from os.path import join as opj

from models import gru, save_gru
from utils import encoder_loss, sim_save


FLAGS = flags.FLAGS

# system
flags.DEFINE_integer(
    'n_gpus', 0, 'How many GPUs we should try to use.')
flags.DEFINE_string(
    'model_name', '', 'If non-empty works as a special name for this model.')
flags.DEFINE_string('results_path', 'experiments',
                    'Name of the directory to save all results within.')
flags.DEFINE_integer(
    'random_seed', -1, 'If not -1, set the random seed to this value. Otherwise the random seed will be the current microsecond.')

# preprocessing
flags.DEFINE_integer('sample_rate', 22050, 'Sample rate for audio files.')
flags.DEFINE_integer('n_mel_features', 256,
                     'Number of channels in mel spectrogram.')
flags.DEFINE_integer(
    'stft_win', 1000, 'Window length for short-time Fourier transform.')
flags.DEFINE_integer(
    'stft_hop', 250, 'Window hop for short-time Fourier transform.')

# training
flags.DEFINE_integer('max_steps', 10000,
                     'How many training batches to show the network.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('reg_coeff', 0.0001,
                   'Coefficient for L2 regularization of weights.')
flags.DEFINE_integer(
    'save_every', 500, 'Save the model and print metrics at this many training steps.')

# model
flags.DEFINE_string(
    'restore_from', '', 'If non-empty, restore the previous model from this directory and train it using the new flags.')


def train_loop(sm, FLAGS, rng, data_iter):

    # get GRU model and initialize parameters
    init_l1, apply_l1 = gru(512)
    x = next(data_iter)
    l1_shape, l1_params = init_l1(rng, x.shape)

    # construct the optimizer
    opt_initialize, opt_update, opt_get_params = rmsprop(FLAGS.lr)
    opt_state = opt_initialize(l1_params)

    # define full forward pass and its derivative
    def forward_pass(params, inp):
        y = apply_l1(params, inp)
        loss = encoder_loss(x, y)
        return loss, y
    forward_backward_pass = value_and_grad(forward_pass, has_aux=True)

    # track some metrics
    loss_log = []
    input_variance_log = []
    output_variance_log = []

    # begin training loop
    for i in range(FLAGS.n_steps):

        # get next training sample
        x = next(data_iter)

        # forward pass and backward pass
        (loss, y), grads = forward_backward_pass

        # optimizer step
        opt_state = opt_update(i, grads, opt_state)
        l1_params = opt_get_params(opt_state)

        # record metrics
        loss_log.append(loss.numpy().item())
        input_variance_log.append(np.std(x.numpy()))
        output_variance_log.append(np.std(y.numpy()))

        if i > 0 and i % FLAGS.save_every == 0:

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f'{timestamp} Saving {sm.sim_name} at iteration {i}.\n')

            sim_save(sm, 'loss_log', loss_log)
            sim_save(sm, 'input_variance_log', input_variance_log)
            sim_save(sm, 'output_variance_log', output_variance_log)

            print(
                f'\tLoss: {loss_log[-1]}\n\tInput variance: {input_variance_log[-1]}\n\tOutput variance: {output_variance_log[-1]}')

            save_gru(sm, l1_params)

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'{timestamp} Finished training.')

    print(f'Final save of {sm.sim_name}.')
    save_gru(sm, l1_params)
    sim_save(sm, 'loss_log', loss_log)
    sim_save(sm, 'input_variance_log', input_variance_log)
    sim_save(sm, 'output_variance_log', output_variance_log)

        
    date = datetime.now().strftime('%d-%m-%Y')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'Training {sm.sim_name} ended on {date} at {timestamp}.')


def main(_argv):

    # construct simulation manager
    if FLAGS.model_name == '':
        identifier = ''.join(random.choice(
            string.ascii_lowercase + string.digits) for _ in range(4))
        sim_name = 'model_{}'.format(identifier)
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
        if FLAGS.random_seed != -1:
            random_seed = FLAGS.random_seed
        else:
            random_seed = datetime.now().microsecond
        np.save(opj(sm.paths.data_path, 'random_seed'), random_seed)
        rd.seed(random_seed)
        random.seed(random_seed)
        rng = jrd.PRNGKey(random_seed)

        train_loop(sm, FLAGS, rng, data_iter)


if __name__ == '__main__':

    app.run(main)
