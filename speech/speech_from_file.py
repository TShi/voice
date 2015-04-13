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


def record(filename):
	fs,y = scipy.io.wavfile.read(filename)

	num_silent = 0
	snd_started = False

	r = array('h')
	print "Go!"
	num_periods = 0
	for snd_data in chunks(y,CHUNK_SIZE):
		# print np.mean(map(abs,snd_data)), is_silent(snd_data)
		snd_data = snd_data[:,0]
		silent = is_silent(snd_data)
		if silent and snd_started:
			if num_periods <= 4:
				# print "Too short, resampling"
				snd_started = False
				r = array('h')
				num_periods = 0
				continue
			else:
				yield r[:-CHUNK_SIZE]
				r=array('h')
				num_periods = 0
				num_silent = 0
				snd_started = False
				continue
		elif silent and not snd_started: # hasn't started yet
			continue
		elif not silent and snd_started: # okay
			r.extend(normalize(snd_data))
			num_periods += 1
			print num_periods,len(r)
		else: # sound just started
			print "Start recording"
			snd_started = True

def findmax(label):
	largest = -1
	for filename in glob.glob("speech/%s_*.wav" % label):
		largest = max(largest,int(re.findall("speech/%s_(\d+).wav" % label,filename)[0]))
	return largest


def record_to_file(filename,label):
	"Records from the microphone and outputs the resulting data to 'path'"
	seq = findmax(label) + 1
	for data_chunk in record(filename): # 1s chunks
		wf = scipy.io.wavfile.write("speech/%s_%d.wav" % (label,seq), RATE, np.array(data_chunk,dtype="int16"))
		seq += 1

if __name__ == '__main__':
	filename = sys.argv[1]
	label = sys.argv[2]
	print "label: %s" % label
	print("please speak a word into the microphone")
	record_to_file(filename,label)
	print("done - result written to %s" % label)

