import os
import azureml
import shutil
import socket
from azure.storage.blob import BlockBlobService
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import DEFAULT_GPU_IMAGE, DEFAULT_CPU_IMAGE
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveRunConfig, PrimaryMetricGoal
from azureml.pipeline.steps import HyperDriveStep
from azureml.train.hyperdrive import choice, loguniform
from azureml.train.dnn import TensorFlow
from azureml.core.authentication import ServicePrincipalAuthentication
from utils import *
import json


base_dir = '.'

config_json = os.path.join(base_dir, 'config.json')
with open(config_json, 'r') as f:
    config = json.load(f)

try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])
except KeyError as e:
    print("Getting Service Principal Authentication from Azure Devops")
    svc_pr = None
    pass

ws = Workspace.from_config(path=config_json, auth=svc_pr)
    
# folder for scripts that need to be uploaded to Aml compute target
script_folder = './scripts/'
try:
    os.makedirs(script_folder)
except OSError as e:
    for f in os.listdir(script_folder):
        os.unlink(os.path.join(script_folder, f))

cpu_compute_name = config['cpu_compute']
try:
        cpu_compute_target = AmlCompute(ws, cpu_compute_name)
        print("found existing compute target: %s" % cpu_compute_name)
except ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                                max_nodes=4,
                                                                idle_seconds_before_scaledown=1800)    
    cpu_compute_target = ComputeTarget.create(ws, cpu_compute_name, provisioning_config)
    cpu_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
# use get_status() to get a detailed status for the current cluster. 
print(cpu_compute_target.get_status().serialize())

# conda dependencies for compute targets
cpu_cd = CondaDependencies.create(pip_indexurl='https://azuremlsdktestpypi.azureedge.net/sdk-release/Candidate/604C89A437BA41BD942B4F46D9A3591D', pip_packages=["azure-storage-blob", "hickle==3.4.3", "requests==2.21.0", "sklearn", "pandas==0.24.2", "azureml-sdk", "numpy==1.16.2"])

# Runconfigs
cpu_compute_run_config = RunConfiguration(conda_dependencies=cpu_cd)
cpu_compute_run_config.environment.docker.enabled = True
cpu_compute_run_config.environment.docker.gpu_support = False
cpu_compute_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
cpu_compute_run_config.environment.spark.precache_packages = False

shutil.copy(os.path.join(base_dir, 'pipelines_submit.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'pipelines_build.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'train.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'data_utils.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'prednet.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'keras_utils.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'video_decoding.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'data_preparation.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'model_registration.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'config.json'), script_folder)
shutil.copy(os.path.join(base_dir, '.azureml'), script_folder)
    
hash_paths = os.listdir(script_folder)

build_pipelines = PythonScriptStep(
    name='build pipelines',
    script_name="pipelines_build.py", 
    compute_target=cpu_compute_target, 
    source_directory=script_folder,
    runconfig=cpu_compute_run_config,
    allow_reuse=False,
    hash_paths=["."]
)
print("pipeline building step created")


# step 2, submit pipelines
submit_pipelines = PythonScriptStep(
    name='submit pipelines',
    script_name="pipelines_submit.py", 
    # arguments=["--overwrite_published_pipelines", overwrite_published_pipelines],
    compute_target=cpu_compute_target, 
    source_directory=script_folder,
    runconfig=cpu_compute_run_config,
    allow_reuse=False,
    hash_paths=["."]
)
print("pipeline submit step created")

submit_pipelines.run_after(build_pipelines)

pipeline = Pipeline(workspace=ws, steps=[build_pipelines, submit_pipelines])
print ("Pipeline created")

pipeline.validate()
print("Validation complete") 

pipeline_name = 'prednet_master'
published_pipeline = pipeline.publish(name=pipeline_name)
print("pipeline id: ", published_pipeline.id)

datastore = ws.get_default_datastore()

schedule = Schedule.create(workspace=ws, name=pipeline_name + "_sch",
                           pipeline_id=published_pipeline.id, 
                           experiment_name='Schedule_Run',
                           datastore=datastore,
                           wait_for_provisioning=True,
                           description="Datastore scheduler for Pipeline" + pipeline_name
                           )

print("Created schedule with id: {}".format(schedule.id))

published_pipeline.submit(ws, published_pipeline.name)
