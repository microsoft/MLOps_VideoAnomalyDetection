import os
import json
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice
from azureml.core.authentication import ServicePrincipalAuthentication
import hickle as hkl
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

base_dir = "."

config_json = os.path.join(base_dir, "config.json")
with open(config_json, "r") as f:
    config = json.load(f)

auth = ServicePrincipalAuthentication(
    tenant_id=config["tenant_id"],
    service_principal_id=config["service_principal_id"],
    service_principal_password=config["service_principal_password"],
)


# Get workspace
ws = Workspace.from_config(auth=auth)

service = AksWebservice(ws, "videoanom-service")

# load the dataset 
X_test_file = os.path.join('.', 'data', 'preprocessed', 'UCSDped1', 'X_test.hkl')
y_test_file = os.path.join('.', 'data', 'preprocessed', 'UCSDped1', 'y_test.hkl')
# X_test_file = os.path.join('.', 'deployment', 'test_data', 'X_test.hkl')
# y_test_file = os.path.join('.', 'deployment', 'test_data', 'y_test.hkl')
X_test = hkl.load(X_test_file)
y_test = hkl.load(y_test_file)

anoms = np.where(y_test == 1)[0]
random_anom = np.random.choice(anoms)
print(random_anom)
X_test = X_test[(random_anom - 10):(random_anom + 10)]
y_test = y_test[(random_anom - 10):(random_anom + 10)]

json_data = json.dumps({"data": X_test.tolist(), "id": "UCSDped1"})
json_data = bytes(json_data, encoding='utf8')

print("Service URL:", service.scoring_uri)

try:
    prediction = service.run(json_data)
except Exception as e:
    result = str(e)
    print(result)
    raise Exception('web service is not working as expected')

cm = confusion_matrix(y_test.tolist(), prediction)
acc = accuracy_score(y_test.tolist(), prediction)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# if acc < .75:
#     raise Exception("The accuracy of this service is too low!")
