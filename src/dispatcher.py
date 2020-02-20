from sklearn import ensemble
from sklearn.linear_model import LogisticRegression


MODELS = {
	"randomforest": ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1, min_samples_leaf = 5, max_features = "sqrt", verbose = 2),
	"extratrees": ensemble.ExtraTreesClassifier(n_estimators = 100, n_jobs = -1, min_samples_leaf = 5, max_features = "sqrt", verbose = 2),
	"gradientboost": ensemble.GradientBoostingClassifier(n_estimators = 100, min_samples_leaf = 5, max_features = "sqrt", verbose = 2),
	"logreg": LogisticRegression(random_state=0)
}