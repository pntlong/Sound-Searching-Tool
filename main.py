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

data_dir = './audio-folder/yamaha'
audio_files = glob(data_dir + '/*.wav')

def spectral_centroid(x, samplerate=44100):
    x = x.flatten()
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean

def calFrequency( signal ):
    FFT = abs(scipy.fft.fft(signal))
    freqs = scipy.fft.fftfreq(len(FFT), (1.0 / sampling))
    frequency = freqs[range(len(FFT) // 2)]
    plt.plot(frequency, FFT[range(len(FFT) // 2)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    return frequency

def calFFTSignal( signal ):
    FFT = abs(scipy.fft.fft(signal))
    flattenFFT = FFT.flatten()
    return flattenFFT

def calAvgPower(signal):
    arr = signal.flatten()
    rms = 0.0
    for i in arr:
        rms += pow(i,2)
    return round((rms/(len(arr))),4)

def calZCR(signal):
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

def calBandwidth(signal, sampling):
    freq, power = frequency_spectrum(signal,sampling)
    freq.sort()
    minBandWidth = round(freq[0],2)
    maxBandWidth = round(freq[len(freq) - 1],2)
    return minBandWidth, maxBandWidth

def plotPowerInTime(signal, sampling):
    length = signal.shape[0] / sampling
    time = np.linspace(0, length, signal.shape[0])
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original signal")
    plt.show()

def plotPowerInFrequency(signal, sampling):
    frq, X = frequency_spectrum(signal, sampling)
    plt.plot(frq, X, 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|X(freq)|')
    plt.show()

def processApp():
    file = input("Enter audio file: ")
    sampling, signal = scipy.io.wavfile.read(file)
    plotPowerInTime(signal, sampling)

for i in range(0, len(audio_files), 1):
    sampling, signal = scipy.io.wavfile.read(audio_files[i])
    file = open("data/yamaha.txt", "a")  # append mode
    data = "\t" + str(calAvgPower(signal)) + "\t\t" + str(calZCR(signal)) + " \t\t" \
           + str(spectral_centroid(signal)) + "\t\t" + str(calBandwidth(signal,sampling)) + "\n"
    file.write(data)
    file.close()


