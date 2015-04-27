from model import *

voice_manager = VoiceManager(FeatureExtractor)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


models = {
	"LogisticReg": LogisticRegression(),
	"SVM": SVC(probability=True),
	"RandomForest": RandomForestClassifier(),
	"GradientBoost": GradientBoostingClassifier(),
	"AdaBoost": AdaBoostClassifier(),
	"KNN": KNeighborsClassifier()
}
for model_name,model in models.iteritems():
	voice_clf = VoiceClassifier(clf=model)
	scores = cross_validate(voice_manager,voice_clf,shuffle=True,verbose=0,n_trials=10)
	print "%s Accuracy: %.3f (%.3f)" % (model_name, np.mean(scores), np.std(scores))


# while True:
# 	signal = recorder.record()
# 	fund_freq,X = voice_manager.get_features(recorder.fs,signal)
# 	if not X:
# 		print "what?"
# 		continue
# 	print voice_clf.predict(X)