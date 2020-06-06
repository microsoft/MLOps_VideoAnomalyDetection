import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import argparse
import hickle as hkl

from azureml.core import Run

parser = argparse.ArgumentParser(description="Process input arguments")
parser.add_argument(
    "--preprocessed_data",
    default="./data/preprocessed/",
    type=str,
    dest="preprocessed_data",
    help=("path to preprocessed_data"),
)
parser.add_argument(
    "--scored_data",
    default="./data/scored_data/",
    type=str,
    dest="scored_data",
    help=("path to scored_data"),
)
parser.add_argument(
    "--model_path",
    default="./outputs/",
    type=str,
    dest="model_path",
    help=("path to model"),
)
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="UCSDped1",
    help="the dataset that we are using",
    type=str,
    required=False,
)
args = parser.parse_args()

run = Run.get_context()

# run batch_scoring if this data frame is missing
df = pd.read_pickle(os.path.join(args.scored_data, "df.pkl.gz"))

# create feature set
# X = df.loc[
#     :, ["model_std"]
# ]
# X = df
# y = hkl.load(os.path.join(args.preprocessed_data, args.dataset, "y_test.hkl"))
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

clf = (
    # LogisticRegression()
    RandomForestClassifier()
)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])

pipeline.fit(X_train, y_train)

run.log("train accuracy: ", pipeline.score(X_train, y_train))

# save the model as a .pkl file
os.makedirs(args.model_path, exist_ok=True)
with open(os.path.join(args.model_path, "model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
run.log("test accuracy: ", acc)

roc = roc_auc_score(y_test, y_pred)
run.log("test ROC: ", roc)

precision = precision_score(y_test, y_pred)
run.log("test precision: ", precision)

recall = recall_score(y_test, y_pred)
run.log("test recall: ", recall)

cm = confusion_matrix(y_test, y_pred)
run.log("comnfusion Matrix:\n", cm)
