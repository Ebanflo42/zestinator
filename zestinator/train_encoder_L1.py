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

from models import gru
from utils import encoder_loss


FLAGS = flags.FLAGS

# system
flags.DEFINE_bool(
    'use_gpus', 0, 'How many GPUs we should try to use.')
flags.DEFINE_string(
    'model_name', '', 'If non-empty works as a special name for this model.')
flags.DEFINE_string('results_path', 'experiments',
                    'Name of the directory to save all results within.')
flags.DEFINE_integer(
    'random_seed', -1, 'If not -1, set the random seed to this value. Otherwise the random seed will be the current microsecond.')

# preprocessing
flags.DEFINE_integer('sample_rate', 22050, 'Sample rate for audio files.')
flags.DEFINE_integer('n_mel_features', 256, 'Number of channels in mel spectrogram.')
flags.DEFINE_integer('stft_win', 1000, 'Window length for short-time Fourier transform.')
flags.DEFINE_integer('stft_hop', 250, 'Window hop for short-time Fourier transform.')

# training
flags.DEFINE_string('datapath', '', 'Path to a metric fuck ton of mp3s.')
flags.DEFINE_integer('max_steps', 100000,
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

    # construct the optimizer
    init_optimizer, update_optimizer, optimizer_params = rmsprop(FLAGS.lr)

    init_l1, apply_l1 = gru(512)
    x = next(data_iter)
    l1_shape, l1_params = init_l1(rng, x.shape)
    

    # begin training loop
    for i in range(FLAGS.n_steps):

        # get next training sample
        x = next(data_iter)

        # forward pass
        y = apply_l1(l1_params, x)

        # backward pass and optimization step
        loss = encoder_loss(x, y)

        if i > 0 and i % FLAGS.save_every == 0:

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f'{timestamp} Saving {sm.sim_name} at iteration {i}.\n')

            np.save(opj(sm.paths.results_path, 'training_loss'), train_loss)
            np.save(opj(sm.paths.results_path,
                    'training_accuracy'), train_acc)
            np.save(opj(sm.paths.results_path,
                    'training_regularization'), train_reg)

            np.save(opj(sm.paths.results_path, 'validation_loss'), valid_loss)
            np.save(opj(sm.paths.results_path,
                    'validation_accuracy'), valid_acc)

            torch.save(model.state_dict(), opj(
                sm.paths.results_path, 'model_checkpoint.pt'))

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'{timestamp} Finished training.')

    test_loss = []
    test_acc = []
    tot_test_samples = 0

    # loop through entire testing set
    for x, y, mask in test_iter:

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        output, hidden = model(x)

        bce_loss = loss_fcn(output, y, mask).cpu().item()
        acc = acc_fcn(output, y, mask).cpu().item()

        batch_size = x.shape[0]
        tot_test_samples += batch_size
        test_loss.append(batch_size*bce_loss)
        test_acc.append(batch_size*acc)

    final_test_loss = np.sum(test_loss)/tot_test_samples
    final_test_acc = np.sum(test_acc)/tot_test_samples
    print(
        f'  Testing loss: {final_test_loss:.3}\n  Testing accuracy: {100*final_test_acc:.3}%')

    print(f'Final save of.')
    torch.save(model.state_dict(), opj(
        sm.paths.results_path, 'model_checkpoint.pt'))
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

        # check for cuda
        if FLAGS.use_gpu and not torch.cuda.is_available():
            raise OSError(
                'CUDA is not available. Check your installation or set `use_gpu` to False.')

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
        jrd.seed(random_seed)
        rd.seed(random_seed)
        random.seed(random_seed)

        # check for old model to restore from
        if FLAGS.restore_from != '':

            with open(opj(FLAGS.restore_from, 'data', 'FLAGS.json'), 'r') as f:
                old_FLAGS = json.load(f)

            architecture = old_FLAGS['architecture']
            if architecture != FLAGS.architecture:
                print(
                    'Warning: restored architecture does not agree with architecture specified in FLAGS.')
            n_rec = old_FLAGS['n_rec']
            if n_rec != FLAGS.n_rec:
                print(
                    'Warning: restored number of recurrent units does not agree with number of recurrent units specified in FLAGS.')

            model = MusicRNN(
                architecture, n_rec, use_grad_clip=FLAGS.use_grad_clip, grad_clip=FLAGS.grad_clip)
            model.load_state_dict(torch.load(
                opj(FLAGS.restore_from, 'results', 'model_checkpoint.pt')))

        else:
            model = MusicRNN(FLAGS.architecture, FLAGS.n_rec,
                             use_grad_clip=FLAGS.use_grad_clip, grad_clip=FLAGS.grad_clip)
            initialize(model, FLAGS)

        data_iter, valid_iter, test_iter = get_datasets(FLAGS)

        train_loop(sm, FLAGS, model, data_iter, valid_iter, test_iter)


if __name__ == '__main__':

    app.run(main)
