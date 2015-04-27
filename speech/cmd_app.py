from model import *

voice_manager = VoiceManager(FeatureExtractor)
voice_clf = VoiceClassifier(clf=SVC(probability=True))

recorder = Recorder()


voice_clf.fit(voice_manager.X,voice_manager.y)

while True:
	signal = recorder.record()
	fund_freq,X = FeatureExtractor.get_features(recorder.fs,signal)
	if not X:
		print "what?"
		continue
	print voice_clf.predict(X)