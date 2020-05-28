import os
import json
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice
import hickle as hkl
from sklearn.metrics import confusion_matrix, accuracy_score

# Get workspace
ws = Workspace.from_config()

service = AksWebservice(ws, "videoanom-service")

# load the dataset 
X_test_file = os.path.join('.', 'deployment', 'test_data', 'X_test.hkl')
y_test_file = os.path.join('.', 'deployment', 'test_data', 'y_test.hkl')
X_test = hkl.load(X_test_file)
y_test = hkl.load(y_test_file)

json_data = json.dumps({"data": X_test.tolist(), "id": "UCSDped1"})
json_data = bytes(json_data, encoding='utf8')

print("Service URL:", service.scoring_uri)

try:
    prediction = service.run(json_data)
except Exception as e:
    result = str(e)
    print(result)
    raise Exception('web service is not working as expected')

cm = confusion_matrix(y_test.tolist(), prediction[1])
acc = accuracy_score(y_test.tolist(), prediction[1])

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

if acc < .75:
    raise Exception("The accuracy of this service is too low!")
