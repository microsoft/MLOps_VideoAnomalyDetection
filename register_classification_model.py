import argparse
import os
import json
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', dest="model_dir", default = "models")

args = parser.parse_args()
print("all args: ", args)

with open('config.json', 'r') as f:
    config = json.load(f)

try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])
except KeyError as e:
    print("Getting Service Principal Authentication from Azure Devops")
    svr_pr = None
    pass
    
ws = Workspace.from_config(auth=svc_pr)

model = Model.register(ws, os.path.join(args.model_dir, "logistic_regression", "model.pkl"), model_name="logistic_regression")

