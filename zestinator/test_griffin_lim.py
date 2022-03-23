import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import vmap
from functools import partial
from os.path import join as opj

from librosa import load
from soundfile import write
#from librosa.feature import melspectrogram
from preprocessing import melspectrogram
from librosa.util._nnls import nnls
from librosa.feature.inverse import mel_to_audio
from librosa.core.convert import fft_frequencies
from librosa.core.spectrum import _spectrogram, griffinlim


def twelve_tone_temperament(n_notes=112):
    scale = np.zeros(n_notes + 2, dtype=np.float32)
    scale[1:] = 440 * (2**(np.linspace(-4*12, -4*12 + n_notes, 1 + n_notes)/12))
    return scale


def ttt_filter(sr=22050, n_fft=2048, n_notes=112):

    weights = np.zeros((n_notes, int(1 + n_fft // 2)), dtype=np.float32)

    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    tttfreqs = twelve_tone_temperament(n_notes=n_notes)

    fdiff = np.diff(tttfreqs)
    ramps = np.subtract.outer(tttfreqs, fftfreqs)

    for i in range(n_notes):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (tttfreqs[2 : n_notes + 2] - tttfreqs[:n_notes])
    weights *= enorm[:, np.newaxis]

    return weights


def twelve_tone_spectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_notes=112):

    S, n_fft = _spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    ttt_basis = ttt_filter(sr=sr, n_fft=n_fft, n_notes=n_notes)

    return np.dot(ttt_basis, S), ttt_basis, S


def ttt_to_stft(M, basis):
    inverse = nnls(basis, M)
    return np.power(inverse, 0.5, out=inverse)


def ttt_to_audio(M, basis, hop_length=512, n_iter=32):

    stft = ttt_to_stft(M, basis)
    #stft = M
    audio = griffinlim(stft, hop_length=hop_length, n_iter=n_iter)
    return audio


if __name__ == '__main__':

    '''
    signal, sr = load('/home/medusa/Music/That_Kid-Cobra.mp3', sr=20000)
    print('Audio loaded')
    #spectrogram, basis, original = twelve_tone_spectrogram(signal[:30*sr], sr=sr, n_notes=112, n_fft=2000, hop_length=400)
    #spectrogram = original
    melspectrogram = melspectrogram(signal[:30*sr], sr=sr, n_mels=128, n_fft=2000, hop_length=400)
    print('Spectrogram computed.')
    '''
    sr = 20000
    songdir = '/home/medusa/Music'
    songpaths = [f for f in os.listdir(songdir) if f.endswith('.mp3')]
    waveforms = []
    for i, path in enumerate(songpaths):
        if i < 4:
            waveforms.append(jnp.array(load(opj(songdir, path), sr=sr)[0][:30*sr]))
    waveforms = jnp.stack(waveforms, axis=0)
    print('Waveforms loaded')
    spectrograms = vmap(partial(melspectrogram, n_mels=128, n_fft=2048, hop_length=512))(waveforms)

    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.pcolormesh(spectrogram, cmap='seismic', vmin=0, vmax=10)
    #plt.savefig('spectrogram.png')
    #print('Plot finished.')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.pcolormesh(np.array(spectrograms[0]), cmap='seismic', vmin=0, vmax=10)
    plt.savefig('melspectrogram.png')
    print('Plot finished.')

    #reconstructed = ttt_to_audio(spectrogram, basis, hop_length=400, n_iter=16)
    reconstructed = mel_to_audio(np.array(spectrograms[0]), hop_length=512, n_iter=32, n_fft=2048)
    print('Audio reconstructed.')
    write('reconstruction.wav', reconstructed, sr)
    print('Finished.')