export TRAINING_DATA=data/train_folds.csv 
export TEST_DATA=data/test.csv
export FOLD=0
export MODEL=$1

python3 -m src.train