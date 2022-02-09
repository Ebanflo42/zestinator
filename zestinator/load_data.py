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
    waveform = resample(to_mono(track), sr_native, sr, res_type='kaiser_best')[:int(maxlen*sr)]
    if len(waveform) < sr*maxlen:
        return load_check_track(mp3_paths, maxlen)
    return waveform


def preprocess_one_example(mp3_paths, duration, maxlen):

    sr = 20000

    waveform = load_check_track(mp3_paths, maxlen)
    cut = rd.randint(0, maxlen - duration - 1)
    waveform = waveform[cut*sr : (cut + duration)*sr]

    spectrogram = melspectrogram(
        waveform, sr=sr, n_fft=1000, hop_length=200, n_mels=256)

    return spectrogram


def get_song_iterator(FLAGS):

    mp3_paths = [opj(FLAGS.data_path, f)
                 for f in os.listdir(FLAGS.data_path) if f.endswith('.mp3')]

    n_batches = np.maximum(1, FLAGS.n_gpus)
    minibatch_size = FLAGS.batch_size//n_batches
    assert FLAGS.batch_size % n_batches == 0, 'Batch size should be divisible by the number of GPUs.'

    @background(max_prefetch=4)
    def song_iter():
        while True:
            samples = []
            for i in range(FLAGS.batch_size):
                samples.append(preprocess_one_example(mp3_paths, FLAGS.duration, FLAGS.maxlen))
            #samples = Parallel(n_jobs=2, prefer='threads')(
            #    delayed(preprocess_one_example)(mp3_paths, FLAGS.duration, FLAGS.maxlen) for _ in range(FLAGS.batch_size))
            samples = np.stack([s.T for s in samples], axis=0)
            yield samples

    return song_iter()
