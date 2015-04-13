from utils import *

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


# for filename in glob.glob("samples/*.wav"):
# 	# out_name = "spectra/"+filename[8:-4]+".pdf"
# 	# if os.path.isfile(out_name): continue
# 	fs,y=scipy.io.wavfile.read(filename)
# 	print filename,freq_from_autocorr(y,fs)



def get_features(fs,signal):
	BASE_FREQ = 250.
	fund_freq = freq_from_autocorr(signal,fs)
	# fund_freq_2 = freq_from_fft(y,fs)
	# okay = fund_freq < fund_freq_2 * 1.2 and fund_freq_2 < fund_freq * 1.2
	# print "%s\t%.1f\t%.1f\t%s" % (label,fund_freq, fund_freq_2,
	# 	"" if okay else "<---")
	f,Pxx_den=scipy.signal.periodogram(signal,fs=fs)
	fnew = f/fund_freq*BASE_FREQ
	f = interp1d(fnew, Pxx_den)
	features=f(F_RANGE)
	yield features

def get_features_2(fs,signal):
	fund_freq = freq_from_autocorr(signal,fs)
	# fund_freq_2 = freq_from_fft(y,fs)
	# okay = fund_freq < fund_freq_2 * 1.2 and fund_freq_2 < fund_freq * 1.2
	for yy in chunks(signal, RATE * 1/2):
		f,Pxx_den=scipy.signal.periodogram(signal,fs=fs)
		f_interp = interp1d(f, Pxx_den)
		features = []
		for i in range(1,5):
			features.append(np.log(1+f_interp(fund_freq*i)))
		yield features

def get_features_20(fs,signal):
	fund_freq = freq_from_autocorr(signal,fs)
	# fund_freq_2 = freq_from_fft(y,fs)
	# okay = fund_freq < fund_freq_2 * 1.2 and fund_freq_2 < fund_freq * 1.2
	f,Pxx_den=scipy.signal.periodogram(signal,fs=fs)
	f_interp = interp1d(f, Pxx_den)
	features = [np.log(fund_freq)]
	if fund_freq > 880: return
	for i in np.arange(1,20,0.5):
		features.append(np.log(1+f_interp(fund_freq*i)))
	yield features

# def plot_features(label):
# 	count = 0
# 	for features in get_features(label):
# 		plt.clf()
# 		plt.semilogy(F_RANGE, features)
# 		# plt.plot(f_range, features)
# 		plt.xlabel('frequency [Hz]')
# 		plt.ylabel('PSD [V**2/Hz]')
# 		plt.savefig("spectra/%s_%d.pdf" % (label,count),format="pdf")
# 		count += 1

# def process():
# 	for filename in glob.glob("samples/*.wav"):
# 		label = filename[8:-4]
# 		# if os.path.isfile("spectra/%s.pdf"%label): continue
# 		plot_features(label)


print "Preparing features"
X = []
y = []
frequencies = []
for filename in glob.glob("samples/*.wav"):
	label = filename[8:-4]
	person,num = label.split("_")
	features_group = [] 
	persons_group = []
	fs,signal=scipy.io.wavfile.read("samples/%s.wav" % label)
	fund_freq = freq_from_autocorr(signal,fs)
	print label,fund_freq
	frequencies.append(fund_freq)
	for features in get_features_20(fs,signal):
		features_group.append(features)
		persons_group.append(person)
	X.append(features_group)
		# y.append(1 if person == 'ts' else 0)
	y.append(persons_group)


data = zip(X,y,frequencies)
np.random.shuffle(data)
X,y,frequencies = zip(*data)

n=len(y)
train=int(0.9*n)

X_train,y_train = np.array(flatten(X[:train])),np.array(flatten(y[:train]))
X_test,y_test = np.array(flatten(X[train:])),np.array(flatten(y[train:]))
frequencies_test = np.array(frequencies[train:])
print "Training"

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()


clf = SVC(probability=True,C=25,gamma=0.025)
a = clf.fit(X_train, y_train)

print "Testing"
label_to_ind = dict()
for i,l in enumerate(clf.classes_):
	label_to_ind[l]=i

bad = 0
print "True\tPredict\tScore\tFreq"
for i in range(len(y_test)):
	y_pred = clf.predict(X_test[i])[0]
	if y_pred!=y_test[i]: bad += 1
	print "%s\t%s\t%.3f\t%.1f\t%s" % (
		y_test[i],
		y_pred,
		clf.predict_proba(X_test[i])[0,label_to_ind[y_test[i]]],
		frequencies_test[i],
		"<--" if y_pred!=y_test[i] else "")




