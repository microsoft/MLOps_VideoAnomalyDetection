import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice
import hickle as hkl
import requests
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# Input for Model with all features
# load the dataset 
# X_test_file = os.path.join('.', 'data', 'preprocessed', 'UCSDped1', 'X_test.hkl')
# y_test_file = os.path.join('.', 'data', 'preprocessed', 'UCSDped1', 'y_test.hkl')
X_test_file = os.path.join('.', 'deployment', 'test_data', 'X_test.hkl')
y_test_file = os.path.join('.', 'deployment', 'test_data', 'y_test.hkl')

X_test = hkl.load(X_test_file)
y_test = hkl.load(y_test_file)

first_anomaly = np.where(y_test == 1)[0][0]

X_test_s = X_test[(first_anomaly - 5):(first_anomaly + 5)]
y_test_s = y_test[(first_anomaly - 5):(first_anomaly + 5)]

json_data = json.dumps({"data": X_test_s.tolist(), "id": "UCSDped1"})
json_data = bytes(json_data, encoding='utf8')

# test local docker
headers = {'Content-Type':'application/json'}
r = requests.post('http://localhost:8002/score', data=json_data, headers=headers)

cm = confusion_matrix(y_test_s.tolist(), r.json()[0])
acc = accuracy_score(y_test_s.tolist(), r.json()[0])
print("accuracy (chance: 0.5):", acc)
