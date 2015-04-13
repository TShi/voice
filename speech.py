from utils import *

def is_silent(snd_data):
	"Returns 'True' if below the 'silent' threshold"
	# return max(snd_data) < THRESHOLD
	return np.mean(map(abs,snd_data)) < THRESHOLD

def normalize(snd_data):
	"Average the volume out"
	MAXIMUM = 16384
	times = float(MAXIMUM)/max(abs(i) for i in snd_data)
	r = array('h')
	for i in snd_data:
		r.append(int(i*times))
	return r

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
	print "Go!"
	num_periods = 0
	while 1:
		# little endian, signed short
		snd_data = array('h', stream.read(CHUNK_SIZE))
		if byteorder == 'big':
			snd_data.byteswap()
		# print np.mean(map(abs,snd_data)), is_silent(snd_data)
		silent = is_silent(snd_data)
		if silent and snd_started:
			if num_periods <= 4:
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
			print num_periods,len(r)
		else: # sound just started
			print "Start recording"
			snd_started = True
	print "Finish"
	r = r[:-CHUNK_SIZE]
	sample_width = p.get_sample_size(FORMAT)
	stream.stop_stream()
	stream.close()
	p.terminate()
	return sample_width, r

def findmax(label):
	largest = -1
	for filename in glob.glob("speech/%s_*.wav" % label):
		largest = max(largest,int(re.findall("speech/%s_(\d+).wav" % label,filename)[0]))
	return largest

def record_to_file_full(label):
	"Records from the microphone and outputs the resulting data to 'path'"
	sample_width, data = record()
	data = pack('<' + ('h'*len(data)), *data)
	seq = findmax(label) + 1
	wf = wave.open("speech/%s_%d.wav" % (label,seq), 'wb')
	wf.setnchannels(1)
	wf.setsampwidth(sample_width)
	wf.setframerate(RATE)
	wf.writeframes(data)
	wf.close()

def record_to_file(label):
	"Records from the microphone and outputs the resulting data to 'path'"
	sample_width, data = record()
	data = pack('<' + ('h'*len(data)), *data)
	seq = findmax(label) + 1
	for data_chunk in chunks(data,RATE * 1): # 1s chunks
		wf = wave.open("speech/%s_%d.wav" % (label,seq), 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(sample_width)
		wf.setframerate(RATE)
		wf.writeframes(data_chunk)
		wf.close()
		seq += 1

if __name__ == '__main__':
	label = sys.argv[1]
	print "label: %s" % label
	print("please speak a word into the microphone")
	while True:
		record_to_file_full(label)
	print("done - result written to %s" % label)