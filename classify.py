import scipy.io.wavfile
import scipy.signal

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.mlab import find
from record import *
from scipy.signal import blackmanharris, fftconvolve
from numpy.fft import rfft, irfft

from numpy import argmax, sqrt, mean, diff, log

RATE = 44100

def get():
	for filename in glob.glob("samples/*.wav"):
		out_name = "spectra/"+filename[8:-4]
		if os.path.isfile(out_name): continue
		print filename
		fs,y=scipy.io.wavfile.read(filename)
		# Spectrum
		f,Pxx_den=scipy.signal.periodogram(y,fs=fs)
		okay_indices = np.where((f>30)&(f<3000))
		f1,Pxx_den1=f[okay_indices],Pxx_den[okay_indices]
		plt.clf()
		plt.semilogy(f1, Pxx_den1)
		plt.xlabel('frequency [Hz]')
		plt.ylabel('PSD [V**2/Hz]')
		plt.savefig(out_name+".pdf",format="pdf")
		# # FFT
		# fourier = abs(np.fft.fft(y))
		# freq = np.fft.fftfreq(len(y), d=1./fs)
		# okay_indices = np.where((f>30)&(f<3000))
		# f2,fourier2 = freq[okay_indices],fourier[okay_indices]
		# plt.clf()
		# plt.semilogy(f2, fourier2)
		# plt.xlabel('frequency [Hz]')
		# plt.ylabel('Coef')
		# plt.savefig(out_name+"_f.pdf",format="pdf")


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

# for filename in glob.glob("samples/*.wav"):
# 	# out_name = "spectra/"+filename[8:-4]+".pdf"
# 	# if os.path.isfile(out_name): continue
# 	fs,y=scipy.io.wavfile.read(filename)
# 	print filename,freq_from_autocorr(y,fs)

from scipy.interpolate import interp1d

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
    	if i+n <= len(l):
	        yield l[i:i+n]

F_RANGE = np.arange(0,10000)

def get_features(label):
	BASE_FREQ = 250.
	fs,y=scipy.io.wavfile.read("samples/%s.wav" % label)
	fund_freq = freq_from_autocorr(y,fs)
	fund_freq_2 = freq_from_fft(y,fs)
	okay = fund_freq < fund_freq_2 * 1.2 and fund_freq_2 < fund_freq * 1.2
	print "%s\t%.1f\t%.1f\t%s" % (label,fund_freq, fund_freq_2,
		"" if okay else "<---")
	for yy in chunks(y, RATE * 1/2):
		f,Pxx_den=scipy.signal.periodogram(yy,fs=fs)
		fnew = f/fund_freq*BASE_FREQ
		f = interp1d(fnew, Pxx_den)
		features=f(F_RANGE)
		yield features

def plot_features(label):
	count = 0
	for features in get_features(label):
		plt.clf()
		plt.semilogy(F_RANGE, features)
		# plt.plot(f_range, features)
		plt.xlabel('frequency [Hz]')
		plt.ylabel('PSD [V**2/Hz]')
		plt.savefig("spectra/%s_%d.pdf" % (label,count),format="pdf")
		count += 1

def process():
	for filename in glob.glob("samples/*.wav"):
		label = filename[8:-4]
		# if os.path.isfile("spectra/%s.pdf"%label): continue
		plot_features(label)


from sklearn.svm import SVC

X = []
y = []
for filename in glob.glob("samples/*.wav"):
	label = filename[8:-4]
	person,num = label.split("_")
	for features in get_features(label):
		X.append(features)
		# y.append(1 if person == 'ts' else 0)
		y.append(person)

data = zip(X,y)
np.random.shuffle(data)
X,y = zip(*data)
X,y = np.array(X),np.array(y)
logX = log(X)

n=len(y)
train=int(0.9*n)

clf = SVC(probability=True)
clf.fit(logX[:train],y[:train])
label_to_ind = dict()
for i,l in enumerate(clf.classes_):
	label_to_ind[l]=i

print "True\tScore\tPredict"
for i in range(train,n):
	# print "%d\t%.2f\t%d" % (y[i],clf.predict_proba(logX[i])[0,1],clf.predict(logX[i])[0])
	print "%s\t%s\t%.3f" % (y[i],clf.predict(logX[i])[0],clf.predict_proba(logX[i])[0,label_to_ind[y[i]]])
	# print clf.predict_proba(logX[i])




