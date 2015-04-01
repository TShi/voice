from __future__ import division
from scikits.audiolab import wavread
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
from time import time
import sys
 

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
 
def freq_from_crossings(sig, fs):
    """Estimate frequency by counting zero crossings
    
    """
    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    #crossings = indices
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better. Spline, cubic, whatever
    
    return fs / mean(diff(crossings))
 
def freq_from_fft(sig, fs):
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
 
def freq_from_HPS(sig, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    
    """
    windowed = signal * blackmanharris(len(signal))
 
    from pylab import subplot, plot, log, copy, show
 
    #harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms,1,1)
    plot(log(c))
    for x in range(2,maxharms):
        a = copy(c[::x]) #Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        print 'Pass %d: %f Hz' % (x, fs * true_i / len(windowed))
        c *= a
        subplot(maxharms,1,x)
        plot(log(c))
    show()
 
import glob

if len(sys.argv) > 1:

    filename = sys.argv[1]
 
    print 'Reading file "%s"\n' % filename
    signal, fs, enc = wavread(filename)
     
    print 'Calculating frequency from FFT:',
    start_time = time()
    print '%f Hz'   % freq_from_fft(signal, fs)
    print 'Time elapsed: %.3f s\n' % (time() - start_time)
     
    print 'Calculating frequency from zero crossings:',
    start_time = time()
    print '%f Hz' % freq_from_crossings(signal, fs)
    print 'Time elapsed: %.3f s\n' % (time() - start_time)
     
    print 'Calculating frequency from autocorrelation:',
    start_time = time()
    print '%f Hz' % freq_from_autocorr(signal, fs)
    print 'Time elapsed: %.3f s\n' % (time() - start_time)
     
    print 'Calculating frequency from harmonic product spectrum:'
    start_time = time()
    #freq_from_HPS(signal, fs)
    print 'Time elapsed: %.3f s\n' % (time() - start_time)
else:
    print "Label\tFFT\tAutoCorr"
    for filename in glob.glob("samples/*.wav"):
    	label = filename[8:-4]
    	signal, fs, enc = wavread(filename)
    	fft_freq = freq_from_fft(signal, fs)
    	autocorr_freq = freq_from_autocorr(signal, fs)
    	okay = fft_freq < autocorr_freq * 1.2 and autocorr_freq < fft_freq * 1.2
    	print "%s\t%.1f\t%.1f\t%s" % (label,fft_freq,autocorr_freq,
    		"" if okay else "<---")
