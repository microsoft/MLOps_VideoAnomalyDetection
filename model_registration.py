import argparse
import os
import json
from azureml.core import Workspace, Run, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', dest="input_dir", default = "output")
parser.add_argument('--output_dir', dest="output_dir", default = "output")

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

input_dir = os.path.dirname(args.input_dir)

with open(os.path.join(input_dir, 'data_metrics')) as f:
    metrics = json.load(f)

best_loss = 1.0
best_run_id = None

print(metrics)
for run in metrics.keys():
    try:
        loss = metrics[run]['val_loss'][-1]
        if loss < best_loss:
            best_loss = loss
            best_run_id = run
    except Exception as e:
        print("WARNING: Could get val_los for run_id", run)
        pass

print("best run", best_run_id, best_loss)

from azureml.core import Run

# start an Azure ML run
run = Run.get_context()
run_details = run.get_details()

experiment_name = run_details['runDefinition']['environment']['name'].split()[1]

with open('config.json', 'r') as f:
    config = json.load(f)

svc_pr = ServicePrincipalAuthentication(
    tenant_id=config['tenant_id'],
    service_principal_id=config['service_principal_id'],
    service_principal_password=config['service_principal_password'])

ws = Workspace.from_config(auth=svc_pr)

exp = Experiment(ws, name=experiment_name)
best_run = Run(exp, best_run_id)

# register the model
if best_run_id:
    tags = {}
    tags['run_id'] = best_run_id
    tags['val_loss'] = metrics[best_run_id]['val_loss'][-1]
    model = best_run.register_model(model_name=experiment_name, 
                                    model_path='outputs',
                                    tags=tags)

    # # Writing the registered model details to /aml_config/model.json
    # model_json = {}
    # model_json['model_name'] = model.name
    # model_json['model_version'] = model.version
    # model_json['run_id'] = best_run_id

    # os.makedirs('aml_config', exist_ok=True)
    # with open('aml_config/model.json', 'w') as outfile:
    #     json.dump(model_json, outfile)
else:
    print("Couldn't not find a model to register.  Probably because no run completed")
    raise BaseException