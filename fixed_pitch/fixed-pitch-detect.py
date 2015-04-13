from utils import *
from record import is_silent,normalize
def record():
	"""
	Record a word or words from the microphone and
	return the data as an array of signed shorts.

	Normalizes the audio, trims silence from the
	start and end, and pads with 0.5 seconds of
	blank sound to make sure VLC et al can play
	it without getting chopped off.
	"""
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT, channels=1, rate=RATE,
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
		silent = is_silent(snd_data)
		if silent and snd_started:
			if num_periods <= RATE / CHUNK_SIZE / 2:
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
			r.extend(normalize(snd_data))
			num_periods += 1
			# if num_periods == RATE / CHUNK_SIZE : break
			# print num_periods
		else: # sound just started
			# print "Start recording"
			snd_started = True
	# print "Finish"
	r = r[:-CHUNK_SIZE]
	sample_width = p.get_sample_size(FORMAT)
	stream.stop_stream()
	stream.close()
	p.terminate()
	return sample_width, r

execfile("fixed-pitch-classify.py")

clf.fit(np.concatenate((X_train,X_test)), np.concatenate((y_train,y_test)))

while True:
    label = raw_input("Who are you? Enter your label:   ")
    if (os.path.isfile("fixed-pitch/%s_0.wav" % label)):
        fund_freq, fs = getFundFreq(label, 0)
        print "Listen, here's your pitch"
        playNote(fund_freq, fs)
        print "Now duplicate it!"
	for X_test in get_features_20(RATE,record()[1]):
		y_pred = clf.predict(X_test)[0]
		print "%s (%.2f)" % (y_pred, clf.predict_proba(X_test)[0,label_to_ind[y_pred]])
