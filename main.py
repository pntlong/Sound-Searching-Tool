import librosa
import numpy
import numpy as np
from glob import glob

import scipy.fft
from numpy import linspace, arange
from scipy.fft import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt
from numpy.fft import rfft

# sampFreq, sound = wavfile.read('./audio-folder/ducati/Ducati sound(31).wav')
# signal = sound[:,0]
# fft_spectrum = np.fft.rfft(signal)
# freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
# fft_spectrum_abs = np.abs(fft_spectrum)
# plt.plot(freq[:3000], np.abs(fft_spectrum[:3000]))
# plt.xlabel("frequency, Hz")
# plt.ylabel("Amplitude, units")
# plt.show()
# length = sound.shape[0]/sampFreq
# time = np.linspace,length,sound.shape[0]
# plt.plot( range(1, len(fft_spectrum_abs)+1) , fft_spectrum_abs)
# plt.show()

# data_dir = './audio-folder/yamaha'
# audio_files = glob(data_dir + '/*.wav')
# def calZCR (zcr ) :
#     count = 0
#     for j in range(0, len(zcr) - 1):
#         if zcr[j] != zcr[j+1]:
#             count += 1
#     zcrnew = round((count/len(zcr)),2)
#     return zcrnew
# def calAverage(arr):
#     sum = np.sum(arr)
#     length = len(arr)
#     return round((sum / length),2)
# def calVariant(arr) :
#     average = calAverage(arr)
#     temp = 0
#     for i in arr:
#         temp += pow((i - average),2)
#     variant = round((temp / (len(arr))),2)
#     return variant
# # Read wav
# for i in range(0, len(audio_files), 1):
#     audio_data = audio_files[10]
#     y, sr = librosa.load(audio_files[i], sr=44100)
#
#     # compute rms
#     rms = librosa.feature.rms(y)
#     root_mean_squared = calAverage(rms)
#
#     # compute zcr
#     zcr = librosa.zero_crossings(y)
#     zero_crossing_rate = calZCR(zcr)
#
#     # compute spec_centroid
#     spec_centroid = librosa.feature.spectral_centroid(y)
#     averageCentroid = calAverage(spec_centroid[0])
#     varianCentroid = calVariant(spec_centroid[0])
#     print(averageCentroid)
#     break
#     # compute bandwidth
#     band_width = librosa.feature.spectral_bandwidth(y)
#     band_width[0].sort()
#     minBandWidth = round(band_width[0][0],2)
#     maxBandWidth = round(band_width[0][len(band_width[0]) - 1],2)
#
#     # compute tempo
#     tempo_read_file = librosa.beat.tempo(y)
#     tempoReal = round(tempo_read_file[0],2)
#     # write to file
#     file = open("data/yamaha.txt", "a")
#     data = str(root_mean_squared) + "," + "\t\t" + str(zero_crossing_rate) + "," + "\t\t" + str(averageCentroid) + "," + "\t\t" + str(varianCentroid) \
#             + "," + "\t\t" + str(minBandWidth) + "," + "\t\t" + str(maxBandWidth) + "," + "\t\t" + str(tempoReal) + "\n"
#     print()
#     file.write(data)
#     file.close()

data_dir = './audio-folder/yamaha'
audio_files = glob(data_dir + '/*.wav')
print(audio_files)

sampling, signal = scipy.io.wavfile.read(audio_files[20])
# sampling, signal = scipy.io.wavfile.read('./audio-folder/yamaha/Bản ghi Mới 3.wav')
spectrum = abs(rfft(signal))
# print(signal)

def spectral_centroid(x, samplerate=44100):
    x = x.flatten()
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean

def CalSpectralCentroid( signal ):
    signalFlatten = signal.flatten()
    spectrum = abs(rfft(signalFlatten))
    normalized_spectrum = spectrum / sum(spectrum)
    normalized_frequencies = linspace(0, 1, len(spectrum))
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum)
    return spectral_centroid

def CalFrequency( signal ):
    FFT = abs(scipy.fft.fft(signal))
    freqs = scipy.fft.fftfreq(len(FFT), (1.0 / sampling))
    frequency = freqs[range(len(FFT) // 2)]
    plt.plot(frequency, FFT[range(len(FFT) // 2)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    # plt.show()
    return frequency

def CalFFTSignal( signal ):
    FFT = abs(scipy.fft.fft(signal))
    flattenFFT = FFT.flatten()
    return flattenFFT

def CalRMS(signal):
    arr = signal.flatten()
    rms = np.prod(arr.astype(np.float64))
    for i in arr:
        rms += pow(i,2)
    return round((rms/(len(arr))),4)

def CalZCR(signal):
    count = 0
    arr = signal.flatten()
    for i in range(0,len(arr)-1):
        if arr[i] < 0:
            arr[i] = -1
        elif arr[i] == 0:
            arr[i] = 0
        else:
            arr[i] = 1
    for i in range(1,len(arr)):
        count += abs(arr[i] - arr[i-1])
    return round((count/(2*len(arr))),5)
# print('RMS : ')
print(spectral_centroid(signal))
print(CalSpectralCentroid(signal)*2)

# data_dir = './audio-folder/honda'
# audio_files = glob(data_dir + '/*.wav')
# print(audio_files)

# rate, aud_data = scipy.io.wavfile.read(file)
# sampling, signal = scipy.io.wavfile.read(audio_files[0])
# print(CalFrequency(signal))

length = signal.shape[0] / sampling
time = np.linspace(0, length, signal.shape[0])
plt.plot(time, signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Original signal")
# plt.show()


def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

# frq, X = frequency_spectrum(signal, 1.0/sampling)
# print('x', X)
# plt.plot(frq, X,  'b')
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|X(freq)|')
# plt.show()