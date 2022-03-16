import os
import numpy as np
import numpy.random as rd

from os.path import join as opj
from joblib import delayed, Parallel
from prefetch_generator import background
from librosa.feature import melspectrogram
from librosa.core.audio import __audioread_load, to_mono, resample


def load_check_track(mp3_paths, maxlen):
    sr = 20000
    ix = rd.randint(0, len(mp3_paths))
    path = mp3_paths[ix]
    track, sr_native = __audioread_load(path, 0.0, None, np.float32)
    waveform = resample(to_mono(track), sr_native, sr,
                        res_type='kaiser_best')[:int(maxlen*sr)]
    if len(waveform) < sr*maxlen:
        return load_check_track(mp3_paths, maxlen)
    return waveform


def preprocess_one_example(mp3_paths, duration, maxlen):

    sr = 20000

    waveform = load_check_track(mp3_paths, maxlen)
    cut = rd.randint(0, maxlen - duration - 1)
    waveform = waveform[cut*sr: (cut + duration)*sr]

    spectrogram = melspectrogram(
        waveform, sr=sr, n_fft=1000, hop_length=200, n_mels=256)

    return spectrogram


def load_rand_spectrogram(paths, duration):

    ix = rd.randint(0, len(paths))
    spectrogram = np.load(paths[ix])
    if spectrogram.shape[1] > duration + 1:
        cut = rd.randint(0, spectrogram.shape[1] - duration - 1)
        return spectrogram[:, cut: cut + duration]
    else:
        return load_rand_spectrogram(paths, duration)


def get_song_iterator(FLAGS):

    npy_paths = [opj(FLAGS.data_path, f)
                 for f in os.listdir(FLAGS.data_path) if f.endswith('.npy')]

    @background(max_prefetch=4)
    def song_iter():
        while True:
            samples = Parallel(n_jobs=-1, prefer='threads')(
                delayed(load_rand_spectrogram)(npy_paths, FLAGS.duration) for _ in range(FLAGS.batch_size))
            samples = np.stack([s.T for s in samples], axis=0)
            yield samples.reshape(FLAGS.n_gpus, FLAGS.batch_size//FLAGS.n_gpus, -1, FLAGS.duration)

    return song_iter()
