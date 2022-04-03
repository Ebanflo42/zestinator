import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from os.path import join as opj
from jax.tree_util import tree_flatten


def sim_save(sm, name, array):
    np.save(opj(sm.paths.results_path, name), array)


def unfold_convolution(array, win, hop, axis=0):
    array = jnp.moveaxis(array, axis, 0)
    return jnp.moveaxis(jnp.repeat(array, hop, axis=0), 0, axis)
    #first = jnp.repeat(array[:-1], hop, axis=0)
    #second = jnp.repeat(array[-1][jnp.newaxis], win, axis=0)
    #second = array[-1][jnp.newaxis]
    #together = jnp.concatenate([first, second], axis=0)
    #return jnp.moveaxis(together, 0, axis)


def l2_norm_tree(array_tree):
    flat, _ = tree_flatten(array_tree)
    norms = jnp.asarray([jnp.linalg.norm(leaf) for leaf in flat])
    return jnp.sum(norms)


def plot_spectrogram(sm, spectrogram, i, component):

    fig = plt.figure()
    ax = fig.add_subplot()

    m, v = np.mean(spectrogram), 2*np.std(spectrogram)

    ax.pcolormesh(spectrogram, vmin=0, vmax=10, cmap='seismic')

    ax.set_title(f'{component} spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Mels)')

    plt.savefig(opj(sm.paths.results_path, f'{component}_spectrogram_{i}.png'))
