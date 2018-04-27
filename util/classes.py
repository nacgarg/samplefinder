import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fftpack import fft, rfft
import time

class AudioData():
    def __init__(self, sample_rate, audio_data):
        self.sample_rate = sample_rate
        self.audio_data = audio_data

    def split(self, split_size_samples): # TODO add overlap
        num_pad = split_size_samples - (self.audio_data.shape[0] % split_size_samples)
        padded = np.append(self.audio_data, np.zeros(num_pad))
        split_num = round(len(padded)/split_size_samples)
        return np.split(padded, split_num)
        # To add 50% overlap, just delete the first split_size_samples/2, add the same number of zeros at the end, and split again
        offset = np.delete(padded, range(int(split_size_samples/2)))
        offset = np.append(offset, np.zeros(int(split_size_samples/2)))
        return np.concatenate((np.split(padded, split_num), np.split(offset, split_num)))

class Analysis():
    # Store calculated amplitude_envelope and fft in here for pickling
    def __init__(self, amplitude_envelope, fft):
        self.amplitude_envelope = amplitude_envelope
        self.fft = fft

def amplitude_envelope(sample):
    # Calculate amplitude envelope using a low-pass filtered Hilbert transform
    N = 1000 # Moving average window
    h = hilbert(sample)
    return np.abs(h)
    return np.convolve(np.abs(h), np.ones((N))/float(N))

def fourier(a):
    N = len(a)
    yf = rfft(a)
    return 2.0/N * np.abs(yf[:N//2])

def visual_similarity(a, b):
    plt.plot(amplitude_envelope(a), label="a")
    plt.plot(amplitude_envelope(b), label="b")
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    line, = ax.plot(fourier(a), color='blue', lw=1)
    line, = ax.plot(fourier(b), color='red', lw=1)
    ax.set_xscale('log')
    plt.show()


def sample_similarity(b, amp, ft, debug): # a and b are Analysis
    fft_score = ((ft - fourier(b)) ** 2).mean(axis=0)
    if fft_score < 2000:
        amp_score = np.mean(np.abs(amp-amplitude_envelope(b)))
    else: 
        amp_score = fft_score
    
    if debug:
        print("FFT Score:", fft_score)
        print("Amp Score:", amp_score)
    return (amp_score, fft_score)
    # ideas: envelope, FFT
    # for FFT: take into account pitch offset? or not?