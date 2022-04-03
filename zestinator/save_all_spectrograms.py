import os
from os.path import join as opj
import numpy as np
from librosa import load
from librosa.feature import melspectrogram
from tqdm import tqdm

base_dir = '/home/medusa/Music'
mp3s = [opj(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.mp3')]

for i, f in enumerate(mp3s):
    waveform, sr = load(f, sr=22050)
    spectrogram = melspectrogram(waveform, sr, n_mels=128)
    np.save(opj(base_dir, 'spectrograms', str(i)), spectrogram)