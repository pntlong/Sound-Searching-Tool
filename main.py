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

data_dir = './audio-folder/ducati'
audio_files = glob(data_dir + '/*.wav')

# Read wav
for i in range(0, len(audio_files), 1):
    audio_data = audio_files[i]
    y, sr = librosa.load(audio_files[i], sr=44100)
    zcr = librosa.zero_crossings(y)
    print(zcr)
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # times = librosa.times_like(zcr)
    # ax[0].semilogy(times, zcr[0], label='Zero crossing rate')
    # ax[0].set(xticks=[])
    # ax[0].legend()
    # ax[0].label_outer()
    # ax[1].set(title='Zero crossing rate')
    # plt.show()




