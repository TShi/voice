from utils import *
for filename in glob.glob(DATA_DIR+"speech/*.wav"):
	label=filename[7:-4]
	print label
	fs,y=scipy.io.wavfile.read(filename)
	fund_freq = freq_from_autocorr(y,fs)
	f,Pxx_den=scipy.signal.periodogram(y,fs=fs)
	plt.clf()
	plt.semilogy(f, Pxx_den)
	plt.savefig(DATA_DIR+"speech_spectra/%s.pdf" % (label),format="pdf")