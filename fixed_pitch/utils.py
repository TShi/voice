from sys import byteorder
from array import array
from struct import pack
import glob
import numpy as np
import pyaudio
import wave
import re,sys,os
from sklearn.cluster import KMeans
import scipy.io.wavfile
import scipy.signal

import matplotlib.pyplot as plt
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
from numpy.fft import rfft, irfft

from numpy import argmax, sqrt, mean, diff, log

from scipy.interpolate import interp1d

from sklearn.svm import SVC

THRESHOLD = 500
CHUNK_SIZE = 8192
FORMAT = pyaudio.paInt16
RATE = 44100

F_RANGE = np.arange(0, 5000)

DATA_DIR = "samples/"

def play_tone(frequency, amplitude, duration, fs, stream):
    N = int(fs / frequency)
    T = int(frequency * duration)  # repeat for T cycles
    dt = 1.0 / fs
    # 1 cycle
    tone = (amplitude * np.sin(2 * np.pi * frequency * n * dt)
            for n in xrange(N))
    # todo: get the format from the stream; this assumes Float32
    data = ''.join(pack('f', samp) for samp in tone)
    for n in xrange(T):
        stream.write(data)

def playNote(freq, fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    play_tone(freq, 0.5, 1.0, fs, stream)


def getFundFreq(label, index):
    fs, signal = scipy.io.wavfile.read(DATA_DIR+"fixed_pitch/%s_%d.wav"
                                       % (label, index))
    fund_freq = freq_from_autocorr(signal, fs)
    return fund_freq, fs


def flatten(l): return [item for sublist in l for item in sublist]


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
    	if i+n <= len(l):
	        yield l[i:i+n]


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation

    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]

    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px


def freq_from_fft(signal, fs):
    """Estimate frequency from peak of FFT

    """
    # Compute Fourier transform of windowed signal
    windowed = signal * blackmanharris(len(signal))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def get_dataname(filepath):
    return filepath.split("/")[-1].split(".")[0]
