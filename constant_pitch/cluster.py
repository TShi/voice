from utils import *

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

def get_features_2(label):
	fs,y=scipy.io.wavfile.read(filename)
	label = filename[8:-4]
	fund_freq = freq_from_autocorr(y,fs)
	# fund_freq_2 = freq_from_fft(y,fs)
	# okay = fund_freq < fund_freq_2 * 1.2 and fund_freq_2 < fund_freq * 1.2
	for yy in chunks(y, RATE * 1/2):
		f,Pxx_den=scipy.signal.periodogram(y,fs=fs)
		f_interp = interp1d(f, Pxx_den)
		features = []
		for i in range(1,51):
			features.append(np.log(1+f_interp(fund_freq*i)))
		yield features



X = []
y = []
for filename in glob.glob("samples/*.wav"):
	label = filename[8:-4]
	person,num = label.split("_")
	for features in get_features_2(label):
		X.append(features)
		# y.append(1 if person == 'ts' else 0)
		y.append(person)


data = zip(X,y)
np.random.shuffle(data)
X,y = zip(*data)
X,y = np.array(X),np.array(y)
logX = log(X)

#----------------------------------------------------------------------
# Clustering
from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# logX_pca = pca.fit_transform(logX)
# from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
y_pred = kmeans.fit_predict(logX)
from collections import Counter
from itertools import groupby

for person,actual in groupby(sorted(zip(list(y),y_pred)),key=lambda x:x[0]):
	c = Counter(map(lambda x:x[1], actual))
	print person,c.most_common()
