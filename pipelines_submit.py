import os
import azureml
import shutil
import socket
import json
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveRunConfig, PrimaryMetricGoal
from azureml.pipeline.steps import HyperDriveStep
from azureml.pipeline.core import PublishedPipeline
from azureml.train.hyperdrive import choice, loguniform
from azureml.train.dnn import TensorFlow
from azure.storage.blob import BlockBlobService
from azureml.core.authentication import ServicePrincipalAuthentication

# read AML configuration from json file
config_json = 'config.json'
with open(config_json, 'r') as f:
    config = json.load(f)

# set up service principal for non-interactive authentication
try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])
except KeyError as e:
    print("Getting Service Principal Authentication from Azure DevOps")
    svc_pr = None
    pass

# attach to existing AML workspace    
ws = Workspace.from_config(path=config_json, auth=svc_pr)

print("Get all published pipeline objects in the workspace")
all_pub_pipelines = PublishedPipeline.get_all(ws)
all_pub_pipelines.reverse()

print("Collecting list of published pipelines")
prednet_pipelines = {}
for p in all_pub_pipelines:
    # print(p.name)
    if p.name.startswith("prednet_UCSD"):
        prednet_pipelines[p.name] = p
        print("Found pipeline:", p.name)

print()
experiments = {}
runs = []
for n, p in prednet_pipelines.items():
    print("submitting pipeline:", p.name)
    _ = p.submit(ws, p.name)


print("done")