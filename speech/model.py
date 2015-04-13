from utils import *
from sklearn import cross_validation
from threading import Thread

class VoiceClassifier(object):
	def __init__(self):
		self.clf = SVC(probability=True)
		self.label_to_ind = dict()
	def fit(self,X,y):
		self.clf.fit(X,y)
		self.label_to_ind = dict()
		for i,l in enumerate(self.clf.classes_):
			self.label_to_ind[l]=i
	def predict(self,X):
		pred = self.clf.predict(X)[0]
		prob = self.clf.predict_proba(X)[0,self.label_to_ind[pred]]
		return (pred, prob)

class VoiceManager(object):
	def __init__(self):
		self.load_from_disk()
	def get_features(self,fs,signal):
		"""
		Return (fund_freq, [features])
		"""
		fund_freq = freq_from_autocorr(signal,fs)
		if fund_freq > 880 or fund_freq < 60:
			return (fund_freq,[])
		f,Pxx_den=scipy.signal.periodogram(signal,fs=fs)
		f_interp = interp1d(f, Pxx_den)
		return (fund_freq,
			   [np.log(1+f_interp(fund_freq*i)) for i in np.arange(1,20,0.5)])
	def load_from_disk(self):
		self.X = []
		self.y = []
		self.labels = []
		self.freqs = []
		for filename in glob.glob(DATA_DIR+"speech/*.wav"):
			label = get_dataname(filename)
			person,num = label.split("_")
			fs,signal=scipy.io.wavfile.read(DATA_DIR+"speech/%s.wav" % label)
			fund_freq,X = self.get_features(fs,signal)
			if not X:
				print "Err: %s, %.1f" % (label,fund_freq)
				continue
			else:
				print label,fund_freq
			self.labels.append(label)
			self.freqs.append(fund_freq)
			self.X.append(X)
			self.y.append(person)
	def add(self,fs,signal,label):
		person,num = label.split("_")
		fund_freq,features = self.get_features(fs,signal)
		self.X.append(features)
		self.y.append(person)
		self.labels.append(label)
		self.freqs.append(fund_freq)
	def get_snapshot(self,full=False):
		if full:
			return self.X, self.y, self.labels, self.freqs
		else:
			return self.X, self.y


class Recorder(object):
	def __init__(self):
		self.p = pyaudio.PyAudio()
		self.threshold = 500
		self.chunk_size = 1024
		self.format = pyaudio.paInt16
		self.sample_width = self.p.get_sample_size(self.format)
		self.fs = 44100
	def is_silent(self, snd_data):
		"Returns 'True' if below the 'silent' threshold"
		return np.mean(map(abs,snd_data)) < self.threshold
	def normalize(self,snd_data):
		"Average the volume out"
		MAXIMUM = 16384
		times = float(MAXIMUM)/max(abs(i) for i in snd_data)
		r = array('h')
		for i in snd_data:
			r.append(int(i*times))
		return r
	def record(self,min_sec=2.):
		"""
		Record a word or words from the microphone and 
		return the data as an array of signed shorts.
		"""
		stream = self.p.open(format=FORMAT, channels=1, rate=RATE,
			input=True, output=True,
			frames_per_buffer=CHUNK_SIZE)
		num_silent = 0
		snd_started = False
		r = array('h')
		# print "Go!"
		num_periods = 0
		while 1:
			# little endian, signed short
			snd_data = array('h', stream.read(CHUNK_SIZE))
			if byteorder == 'big':
				snd_data.byteswap()
			# print np.mean(map(abs,snd_data)), is_silent(snd_data)
			silent = self.is_silent(snd_data)
			if silent and snd_started:
				if num_periods <= 5 :
					# print "Too short, resampling"
					snd_started = False
					r = array('h')
					num_periods = 0
					continue
				else:
					break
			elif silent and not snd_started: # hasn't started yet
				continue
			elif not silent and snd_started: # okay
				r.extend(self.normalize(snd_data))
				num_periods += 1
				# print num_periods,len(r)
			else: # sound just started
				# print "Start recording"
				snd_started = True
		# print "Finish"
		r = r[:-CHUNK_SIZE]
		stream.stop_stream()
		stream.close()
		return r
	def __del__(self):
		self.p.terminate()
	def findmax(self,label):
		largest = -1
		for filename in glob.glob(DATA_DIR+"speech/%s_*.wav" % label):
			largest = max(largest,int(re.findall(DATA_DIR+"speech/%s_(\d+).wav" % label,filename)[0]))
		return largest
	def save(self,signal,label):
		"Records from the microphone and outputs the resulting data to 'path'"
		signal = pack('<' + ('h'*len(signal)), *signal)
		next_id = self.findmax(label) + 1
		recording_name = "%s_%d" % (label,next_id)
		wf = wave.open(DATA_DIR+"speech/%s.wav" % recording_name, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(self.sample_width)
		wf.setframerate(self.fs)
		wf.writeframes(signal)
		wf.close()
		return recording_name



voice_manager = VoiceManager()
recorder = Recorder()
voice_clf = VoiceClassifier()
voice_clf.fit(*voice_manager.get_snapshot())

while True:
	signal = recorder.record()
	fund_freq,X = voice_manager.get_features(recorder.fs,signal)
	if not X:
		print "what?"
		continue
	print voice_clf.predict(X)


