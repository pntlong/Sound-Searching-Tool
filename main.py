# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import librosa
# import librosa.display as dsp
# from IPython.display import Audio as ipd
# from IPython.lib.display import Audio
import matplotlib
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from librosa import display


def print_plot_play(x, Fs, text=''):
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' %
          (text, Fs, x.shape, x.dtype))
    matplotlib.figure(figsize=(8, 2))
    matplotlib.plot(x, color='gray')
    matplotlib.xlim([0, x.shape[0]])
    matplotlib.xlabel('Time (samples)')
    matplotlib.ylabel('Amplitude')
    matplotlib.tight_layout()
    matplotlib.show()
    # ipd.display(ipd.Audio(data=x, rate=Fs))


def calAverage(arr):
    sum = np.sum(arr)
    length = len(arr)
    return sum / length


data_dir = './audio-folder/yamaha'
audio_files = glob(data_dir + '/*.wav')
print(audio_files)

# Read wav
for i in range(49, len(audio_files), 1):
    audio_data = audio_files[i]
    y, sr = librosa.load(audio_files[i], sr=44100)
    specCenteroid = librosa.feature.spectral_centroid(y, sr)
    S, phase = librosa.magphase(librosa.stft(y=y))
    freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)
    times = librosa.times_like(specCenteroid)
    fig, ax = plt.subplots()
    display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, specCenteroid.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram ' + str(i))
    plt.show()



