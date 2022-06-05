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


data_dir = './audio-folder/honda'
audio_files = glob(data_dir + '/*.wav')
print(audio_files)

# Read wav
for i in range(29, len(audio_files), 1):
    audio_data = audio_files[i]
    y, sr = librosa.load(audio_files[i], sr=22050)
    # print_plot_play(x=x, Fs=Fs, text='WAV file: ')

    rms = librosa.feature.rms(y)
    # print("Average power: ", calAverage(rms[0]))

    zeroXrate = librosa.feature.zero_crossing_rate(y)
    # print('Zero crossing rate: ', calAverage(zeroXrate[0]))

    specCenteroid = librosa.feature.spectral_centroid(y, sr)
    # print('Spectral centroid: ', calAverage(specCenteroid[0]))

    bandwidth = librosa.feature.spectral_bandwidth(y, sr)
    bandwidth = sorted(bandwidth[0])
    # print('Bandwidth range: ', bandwidth[0], bandwidth[len(bandwidth) - 1])
    # print('Bandwidth: ', calAverage(bandwidth))

    tempo, beats = librosa.beat.beat_track(y, sr)
    # print('Tempo: ', tempo)

    file = open("data/honda.txt", "a")  # append mode
    rms = round(calAverage(rms[0]), 3)
    zeroXrate = round(calAverage(zeroXrate[0]), 3)
    specCenteroid = round(calAverage(specCenteroid[0]), 3)
    bandwidthMin = round(bandwidth[0], 3)
    bandwidthMax = round(bandwidth[len(bandwidth) - 1], 3)
    tempo = round(tempo, 3)

    data = "\t" + str(rms) + "\t\t" + str(zeroXrate) + " \t\t" \
           + str(specCenteroid) + "\t\t" + str(bandwidthMin) + "-" + str(bandwidthMax) + "\t" \
           + str(tempo) + "\n"

    print()
    file.write(data)
    file.close()

    file = open("data/honda.txt", "r")
    print("Output of Readlines after appending")
    print(file.readlines())
    print()
    file.close()

"""
audio_data = "audio-folder/Honda Wave S 110 .wav"
y, sr = librosa.load(audio_data, sr=22050)
# print_plot_play(x=x, Fs=Fs, text='WAV file: ')

rms = librosa.feature.rms(y)
print("Average power: ", calAverage(rms[0]))

zeroXrate = librosa.feature.zero_crossing_rate(y)
print('Zero crossing rate: ', calAverage(zeroXrate[0]))

specCenteroid = librosa.feature.spectral_centroid(y, sr)
print('Spectral centroid: ', calAverage(specCenteroid[0]))

bandwidth = librosa.feature.spectral_bandwidth(y, sr)
bandwidth = sorted(bandwidth[0])
print('Bandwidth range: ', bandwidth[0], bandwidth[len(bandwidth) - 1])
print('Bandwidth: ', calAverage(bandwidth))

tempo, beats = librosa.beat.beat_track(y, sr)
print('Tempo: ', tempo)

file = open("data/honda.txt", "a")  # append mode
rms = round(calAverage(rms[0]), 3)
zeroXrate = round(calAverage(zeroXrate[0]), 3)
specCenteroid = round(calAverage(specCenteroid[0]), 3)
bandwidthMin = round(bandwidth[0], 3)
bandwidthMax = round(bandwidth[len(bandwidth) - 1],3)
tempo = round(tempo, 3)

data = "\t"+ str(rms) + "\t\t" + str(zeroXrate) + " \t\t" \
       + str(specCenteroid) + "\t\t" + str(bandwidthMin) + "-" + str(bandwidthMax) + "\t" \
       + str(tempo) + "\n"

print()
file.write(data)
file.close()

file = open("data/honda.txt", "r")
print("Output of Readlines after appending")
print(file.readlines())
print()
file.close()
"""
# Read mp3
# fn_mp3 = 'audio-folder/Ducatisound.mp3'
# x, Fs = librosa.load(fn_mp3, sr=None)
# print_plot_play(x=x, Fs=Fs, text='MP3 file: ')
