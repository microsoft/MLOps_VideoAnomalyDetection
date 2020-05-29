from azureml.core.model import Model
from azureml.core import Workspace, Environment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import InferenceConfig
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice
import json
import os
import shutil

base_dir = "."

source_directory = 'deployment_assets'

if os.path.exists(source_directory):
    print("Deleting:", source_directory)
    shutil.rmtree(source_directory)
os.makedirs(source_directory)

shutil.copytree(
    os.path.join(base_dir, "models"),
    os.path.join(base_dir, source_directory, "models"))
shutil.copy('deployment/score.py', source_directory)

config_json = os.path.join(base_dir, "config.json")
with open(config_json, "r") as f:
    config = json.load(f)

auth = ServicePrincipalAuthentication(
    tenant_id=config["tenant_id"],
    service_principal_id=config["service_principal_id"],
    service_principal_password=config["service_principal_password"],
)

ws = Workspace(
    config["subscription_id"],
    config["resource_group"],
    config["workspace_name"],
    auth=auth
)

models_init = Model.list(ws)

prednet_model_names = []
clf_model_names = []
models = []

for model in models_init:
    if model.name not in clf_model_names + prednet_model_names:
        if model.name.startswith("prednet_"):
            prednet_model_names.append(model.name)
        elif model.name.startswith("clf_"):
            clf_model_names.append(model.name)
        models.append(model)

with open(os.path.join(source_directory, 'models.json'), 'w') as f:
    json.dump({
        "prednet_model_names": prednet_model_names,
        "clf_model_names": clf_model_names
        }, f)

env = Environment.get(ws, 'prednet')
env.python.conda_dependencies.add_pip_package('azureml-defaults')

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env,
    source_directory=source_directory)

prov_config = AksCompute.provisioning_configuration(cluster_purpose="DevTest")

aks_name = 'videoanom-aks'

try:
    aks_target = AksCompute(ws, aks_name)
except Exception:
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws,
        name=aks_name,
        provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)

aks_config = AksWebservice.deploy_configuration()

aks_service_name = 'videoanom-service'

aks_service = Model.deploy(
    workspace=ws,
    name=aks_service_name,
    models=models,
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=aks_target,
    overwrite=True)

aks_service.wait_for_deployment(show_output=True)

print(aks_service.state)
