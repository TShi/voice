from utils import *

for filename in glob.glob("samples/*.wav"):
	fs,y=scipy.io.wavfile.read(filename)
	label = filename[8:-4]
	fund_freq = freq_from_autocorr(y,fs)
	f,Pxx_den=scipy.signal.periodogram(y,fs=fs)
	f_interp = interp1d(f, Pxx_den)
	features = []
	for i in range(1,11):
		features.append(log(1+f_interp(fund_freq*i)))
	# print features
	plt.clf()
	plt.plot(features)
	plt.savefig("overtones/%s.pdf"%label,format="pdf")