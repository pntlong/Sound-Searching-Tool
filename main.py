import os
import librosa
# import librosa.display as dsp
# from IPython.display import Audio as ipd
# from IPython.lib.display import Audio
# import matplotlib
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from librosa import display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

data_dir = './audio-folder/yamaha'
audio_files = glob(data_dir + '/*.wav')

# Read wav
for i in range(0, len(audio_files), 1):
    audio_data = audio_files[i]
    y, sr = librosa.load(audio_files[i], sr=44100)
    rms = librosa.feature.rms(y)
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='Zero crossing rate')
    S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    librosa.feature.rms(S=S)
    plt.show()



