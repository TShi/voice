from model import *

# Interactive Mode 

recorder = Recorder()

def ask_name():
	cmd = raw_input("n - change name, Enter - continue.")
	if cmd == 'n':
		return raw_input("Your name: ")
	elif cmd != "":
		print "Unknown command"
		return ask_name()
	else:
		return ""

person = ask_name()
test_mode = person == ""
while True:
	signal = recorder.record()
	fund_freq,X = voice_manager.get_features(recorder.fs,signal)
	print voice_clf.predict(X)
	if test_mode:
		cmd = raw_input("Enter /<name> to train. Enter .<name> to switch to Train mode.")
		if len(cmd)>=2 and cmd[0] == "/":
			voice_manager.add(recorder.fs,signal,recorder.save(signal,cmd[1:]))
		elif len(cmd)>=2 and cmd[0] == ".":
			test_mode = False
			person = cmd[1:]
			voice_manager.add(recorder.fs,signal,recorder.save(signal,person))
	else: # train mode
		cmd = raw_input("Enter / to Train. Enter . to switch to Test mode.")
		if cmd == "/":
			voice_manager.add(recorder.fs,signal,recorder.save(signal,person))
		elif cmd == ".":
			person = ""
			test_mode = True
