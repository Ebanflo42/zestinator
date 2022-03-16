import os
import librosa
import numpy as np

import jax.numpy as jnp

from jax import vmap, pmap, value_and_grad, jit
from jax.example_libraries.optimizers import rmsprop

from simmanager import SimManager
from functools import partial
from datetime import datetime
from absl import app, flags
from os.path import join as opj

from soundfile import write
from load_data import get_song_iterator
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio
from utils import l2_norm_tree, plot_spectrogram, sim_save
from models import encoder, decoder, save_triple_gru, load_triple_gru

songpath1 = "/home/medusa/Music/Animals_As_Leaders-The_Brain_Dance.mp3"
songpath2 = "/home/medusa/Music/That_Kid-Cobra.mp3"

modelpath = "experiments/autoencoder_bvda"

sr = 20000
waveform1, _ = librosa.load(songpath1, sr=sr)
waveform2, _ = librosa.load(songpath2, sr=sr)
spectrogram1 = melspectrogram(
    waveform1, sr=sr, n_fft=1000, hop_length=200, n_mels=256)[:, 100*60 : 100*120]
spectrogram2 = melspectrogram(
    waveform2, sr=sr, n_fft=1000, hop_length=200, n_mels=256)[:, :100*60]
print(spectrogram1.shape, spectrogram2.shape)
input = jnp.asarray(np.stack([spectrogram1.T, spectrogram2.T], axis=0))
print('Input loaded')

encoder_params = load_triple_gru(modelpath, component='encoder')
decoder_params = load_triple_gru(modelpath, component='decoder')
init_encoder, apply_encoder = encoder()
init_decoder, apply_decoder = decoder()
print('Models loaded')

encoding = vmap(partial(apply_encoder, encoder_params))(input)
recombination = jnp.mean(encoding, axis=0)[jnp.newaxis]
print('Encoding computed.')

new_spectrogram = vmap(partial(apply_decoder, decoder_params))(recombination)[0]
print('New spectrogram computed')

new_waveform = mel_to_audio(np.array(new_spectrogram.T), sr=sr, n_fft=1000, hop_length=200)
write(opj(modelpath, 'results', 'Cobra_Dance.wav'), new_waveform, sr)