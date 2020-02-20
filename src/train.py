import os
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd 
import numpy as np
from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}


if __name__ == "__main__":
	df = pd.read_csv(TRAINING_DATA).dropna()
	test_df = pd.read_csv(TEST_DATA).fillna(method='pad')


	train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
	valid_df = df[df.kfold==FOLD].reset_index(drop=True)

	ytrain = train_df.Survived.values
	yvalid = valid_df.Survived.values

	print("nclass: ", len(train_df.Survived.unique()))

	train_df = train_df.drop(["PassengerId", "Survived", "kfold"], axis = 1)
	valid_df = valid_df.drop(["PassengerId", "Survived", "kfold"], axis = 1)
	test_PassengerID = test_df["PassengerId"]
	test_df = test_df.drop(["PassengerId"], axis = 1)

	valid_df = valid_df[train_df.columns]
	test_df = test_df[train_df.columns]


	label_encoders = {}
	for c in train_df.columns:
		lbl = preprocessing.LabelEncoder()
		lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + test_df[c].values.tolist())
		train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
		valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
		test_df.loc[:, c] = lbl.transform(test_df[c].values.tolist())
		label_encoders[c] = lbl

	#data is ready to train
	clf = dispatcher.MODELS[MODEL]
	clf.fit(train_df, ytrain)
	preds = clf.predict_proba(valid_df)[:, 1]
	#print(preds)
	print("Validation score: ", metrics.roc_auc_score(yvalid, preds))
	test_preds = clf.predict(test_df)
	print(test_preds)

	final_results = pd.DataFrame()
	final_results["PassengerId"] = test_PassengerID
	final_results["Survived"] = test_preds
	final_results.to_csv(f"data/test_preds_{MODEL}.csv")
	#np.savetxt("data/test_preds.csv", test_preds, delimiter=",")
	joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
	joblib.dump(clf, f"models/{MODEL}.pkl")

	print("Done!")