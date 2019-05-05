import numpy
import os, json, datetime, sys
from operator import attrgetter
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice
import hickle as hkl

# Get workspace
ws = Workspace.from_config()

# Get the ACI Details
try:
    with open("aml_config/aci_webservice.json") as f:
        config = json.load(f)
except:
    print('No new model, thus no deployment on ACI')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(0)

service_name = config['aci_name']
print("Service :", service_name)
# Get the hosted web service
service = AciWebservice(ws, service_name)

# Input for Model with all features
# load the dataset 
test_file = os.path.join('..', 'data', 'preprocessed', 'X_test.hkl')
X = hkl.load(test_file)
X_test = X[:10]

json_data = json.dumps({"data": X_test.tolist()})
json_data = bytes(json_data, encoding='utf8')

print("Service URL:", service.scoring_uri)

try:
    prediction = service.run(json_data)
    print(prediction)
except Exception as e:
    result = str(e)
    print(result)
    raise Exception('ACI service is not working as expected')
 
