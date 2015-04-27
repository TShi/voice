from model import *

voice_manager = VoiceManager(FeatureExtractor)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
	"SVM": SVC(probability=True),
	"KNN": KNeighborsClassifier(),
	"GradientBoost": GradientBoostingClassifier(),
	"RandomForest": RandomForestClassifier(),
	"Bagging": BaggingClassifier(),
	"LogisticReg": LogisticRegression(),
	"AdaBoost": AdaBoostClassifier()
}

print "Model & Train & Test \\\\"
for model_name,model in models.iteritems():
	voice_clf = VoiceClassifier(clf=model)
	train_scores,test_scores = cross_validate(voice_manager,voice_clf,shuffle=True,verbose=0,n_trials=5)
	print "%s & %.3f (%.3f) & %.3f (%.3f) \\\\" % (
		model_name, np.mean(train_scores), np.std(train_scores),
		np.mean(test_scores), np.std(test_scores)
		)
