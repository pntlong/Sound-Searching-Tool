from cmath import sqrt
import numpy as np
from glob import glob
import scipy.fft
from numpy import linspace, arange
from scipy.fft import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt
from numpy.fft import rfft

#./audio-folder/ducati/Ducati Panigale V4 Speciale Sound Check Akrapovic full Titanium Racing Exhaust (mp3cut.net) (1).wav
data_dir = './audio-folder/honda'
audio_files = glob(data_dir + '/*.wav')
lenAudioFile = len(audio_files)
ducatiCentral = [41972727.272, 0.032, 3142.55, 0, 22985.751]
hondaCentral = [35196327.213, 0.197, 6567.135, 0, 22049.741]
yamahaCentral = [3740977.917, 0.042, 3078.08, 0, 22049.742]
maxDistanceYamaha = 13825172.380770255
maxDistanceHonda = 168620860.68438935
maxDistanceDucati = 214900453.3454481
def spectral_centroid(x, sampling):
    x = x.flatten()
    magnitudes = np.abs(np.fft.rfft(x))
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/sampling)[:length//2+1])
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)

def calFrequency( signal, sampling ):
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
    x = x - np.average(x)  # zero-centering
    n = len(x)
    k = arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft(x) / n # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)

def calDistance(feature, central):
    tempDistance = 0.0
    for i in range(0, len(feature)- 1):
        tempDistance += pow((feature[i] - central[i]),2)
    tempDistance = sqrt(tempDistance)
    return tempDistance

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
    featureAudio = []
    distanceHonda = 0.0
    distanceYamaha = 0.0
    distanceDucati = 0.0
    file = input("Enter audio file: ")
    sampling, signal = scipy.io.wavfile.read(file)
    minBandwidth, maxBandwidth = calBandwidth(signal, sampling)
    featureAudio.extend([calAvgPower(signal), calZCR(signal), spectral_centroid(signal,sampling), minBandwidth, maxBandwidth])
    for i in range(0, len(featureAudio) - 1):
        distanceHonda += pow((featureAudio[i] - hondaCentral[i]),2)
        distanceDucati += pow((featureAudio[i] - ducatiCentral[i]), 2)
        distanceYamaha += pow((featureAudio[i] - yamahaCentral[i]),2)
    distanceHonda = sqrt(distanceHonda)
    # print('Khoang cach den tam cua cum Honda la : ' + str(distanceHonda))
    distanceYamaha = sqrt(distanceYamaha)
    # print('Khoang cach den tam cua cum Yamaha la : ' + str(distanceYamaha))
    distanceDucati = sqrt(distanceDucati)
    # print('Khoang cach den tam cua cum Ducati la : ' + str(distanceDucati))
    minDistance = 0.0
    if(abs(distanceYamaha) < abs(distanceHonda) and abs(distanceYamaha) < abs(distanceDucati)):
        minDistance = abs(distanceYamaha)
        if(minDistance < abs(maxDistanceYamaha)):
            print('Hang cua xe la : Yamaha')
        else: print('Khong thuoc 3 hang xe tren')
    elif(abs(distanceDucati) < abs(distanceHonda) and abs(distanceDucati) < abs(distanceYamaha)):
        minDistance = abs(distanceDucati)
        if (minDistance < abs(maxDistanceDucati)):
            print('Hang cua xe la : Ducati')
        else: print('Khong thuoc 3 hang xe tren')
    elif (abs(distanceHonda) < abs(distanceDucati) and abs(distanceHonda) < abs(distanceYamaha)):
        minDistance = abs(distanceHonda)
        if (minDistance < abs(maxDistanceHonda)):
            print('Hang cua xe la : Honda')
        else:print('Khong thuoc 3 hang xe tren')
    # print(minDistance)
    # plotPowerInTime(signal, sampling)
def calCentral(arrFeature ):
    tempZCR = 0.0
    tempRMS = 0.0
    tempSpectralCentroid = 0.0
    tempBandwidth = 0.0
    for i in range(0, lenAudioFile, 1):
        sampling, signal = scipy.io.wavfile.read(audio_files[i])
        tempRMS += calAvgPower(signal)
        tempZCR += calZCR(signal)
        tempSpectralCentroid += spectral_centroid(signal, sampling)
        minBandwidth, maxBandwidth = calBandwidth(signal, sampling)
        tempBandwidth += maxBandwidth
    arrFeature.extend([ tempRMS/lenAudioFile, tempZCR/lenAudioFile, tempSpectralCentroid/lenAudioFile , 0 , tempBandwidth/lenAudioFile ])
    return arrFeature
processApp()
# distance = 0.0
# for i in range(0, lenAudioFile, 1):
#     processApp(audio_files[i])
#     sampling, signal = scipy.io.wavfile.read(audio_files[i])
#     minBandwidth, maxBandwidth = calBandwidth(signal, sampling)
#     feature = []
#     feature.extend( [ calAvgPower(signal), calZCR(signal), spectral_centroid(signal, sampling) , minBandwidth , maxBandwidth ])
#     distanceAudio = calDistance(feature, yamahaCentral)
#     if abs(distanceAudio) > abs(distance):
#         distance = distanceAudio
    # file = open("data/honda.txt", "a")  # append mode
    # data = "\t" + str(calAvgPower(signal)) + "\t\t" + str(calZCR(signal)) + " \t\t" \
    #        + str(spectral_centroid(signal)) + "\t\t" + str(calBandwidth(signal,sampling)) + "\n"
    # file.write(data)
    # file.close()

# print(distance)
