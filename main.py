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

data_dir = './audio-folder/honda'
audio_files = glob(data_dir + '/*.wav')
print(audio_files)

sampling, signal = scipy.io.wavfile.read(audio_files[0])
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

def CalAvgPower(signal):
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

def CalBandwidth(signal, sampling):
    freq, power = frequency_spectrum(signal,sampling)
    freq.sort()
    print('freq', freq)
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

# print('RMS : ')
# print(spectral_centroid(signal))
# print(CalSpectralCentroid(signal)*2)

# data_dir = './audio-folder/honda'
# audio_files = glob(data_dir + '/*.wav')
# print(audio_files)

# rate, aud_data = scipy.io.wavfile.read(file)
# sampling, signal = scipy.io.wavfile.read(audio_files[0])
# print(CalFrequency(signal))

def processApp():
    file = input("Enter audio file: ")
    sampling, signal = scipy.io.wavfile.read(file)
    plotPowerInTime(signal, sampling)

