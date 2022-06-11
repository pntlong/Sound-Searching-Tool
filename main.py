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

data_dir = './audio-folder/honda'
audio_files = glob(data_dir + '/*.wav')
count = 0
temp = []
# Read wav
for i in range(0, len(audio_files) - 1):
    y, sr = librosa.load(audio_files[5], sr=44100)
    centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]
    print(centroid)
    print(len(centroid))
    break
    # flatness = librosa.feature.spectral_flatness(y)
    # flatness_new = np.ndarray.flatten(flatness)
    # for k in flatness_new:
    #     j = k*10*10*10*10
    #     temp.append(j)
    # plt.plot(range(1, len(temp) + 1), temp)
    # plt.show()
    # temp = []


