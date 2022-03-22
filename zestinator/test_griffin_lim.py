import numpy as np
import matplotlib.pyplot as plt

from librosa import load
from soundfile import write
from librosa.util._nnls import nnls
from librosa.core.convert import fft_frequencies
from librosa.core.spectrum import _spectrogram, griffinlim


def twelve_tone_temperament(n_notes=88):
    scale = np.zeros(n_notes + 2, dtype=np.float32)
    scale[1:] = 440 * (2**(np.linspace(-3*12, -3*12 + n_notes, 1 + n_notes)/24))
    return scale


def ttt_filter(sr=22050, n_fft=2048, n_notes=88):

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


def twelve_tone_spectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_notes=88):

    S, n_fft = _spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length
    )

    ttt_basis = ttt_filter(sr=sr, n_fft=n_fft, n_notes=n_notes)

    return np.dot(ttt_basis, S), ttt_basis


def ttt_to_stft(M, basis):
    inverse = nnls(basis, M)
    return np.power(inverse, 0.5, out=inverse)


def ttt_to_audio(M, sr=22050, n_fft=2048, hop_length=512, n_iter=32):

    stft = ttt_to_stft(M, sr=sr, n_fft=n_fft)
    audio = griffinlim(stft, hop_length=hop_length, n_iter=n_iter)
    return audio


if __name__ == '__main__':

    signal, sr = load('/home/medusa/Music/That_Kid-Cobra.mp3', sr=20000)
    print('Audio loaded')
    spectrogram = twelve_tone_spectrogram(signal[:30*sr], sr=sr, n_notes=176, n_fft=2000, hop_length=400)
    print('Spectrogram computed.')

    #variance = np.std(spectrogram)
    #avg = np.mean(spectrogram)
    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.pcolormesh(spectrogram, cmap='seismic')
    #plt.savefig('spectrogram.png')
    #print('Plot finished.')

    reconstructed = ttt_to_audio(spectrogram, sr=sr, n_fft=2000, hop_length=400, n_iter=16)
    print('Audio reconstructed.')
    write('that_kid_cobra_reconstructed.wav', reconstructed, sr)
    print('Finished.')