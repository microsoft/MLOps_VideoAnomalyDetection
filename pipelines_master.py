import os
import azureml
import shutil
import socket
from azure.storage.blob import BlockBlobService
from azureml.core import Workspace, Run, Experiment, Datastore, Environment
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import DEFAULT_GPU_IMAGE, DEFAULT_CPU_IMAGE
from azureml.core.compute import ComputeTarget

# from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BanditPolicy,
    HyperDriveRunConfig,
    PrimaryMetricGoal,
)
from azureml.pipeline.steps import HyperDriveStep
from azureml.train.hyperdrive import choice, loguniform
from azureml.train.dnn import TensorFlow
from azureml.core.authentication import ServicePrincipalAuthentication

# from utils import *
import json

from utils import disable_pipeline

from azureml.core import VERSION

print("azureml.core.VERSION", VERSION)

base_dir = "."

config_json = os.path.join(base_dir, "config.json")
with open(config_json, "r") as f:
    config = json.load(f)

auth = ServicePrincipalAuthentication(
    tenant_id=config["tenant_id"],
    service_principal_id=config["service_principal_id"],
    service_principal_password=config["service_principal_password"],
)

ws = Workspace.create(
    config["workspace_name"],
    location=config["workspace_region"],
    resource_group=config["resource_group"],
    subscription_id=config["subscription_id"],
    auth=auth,
    exist_ok=True
)

print(ws.get_details)


# folder for scripts that need to be uploaded to Aml compute target
script_folder = "./scripts/"
if os.path.exists(script_folder):
    print("Deleting:", script_folder)
    shutil.rmtree(script_folder)
os.makedirs(script_folder)


shutil.copy(os.path.join(base_dir, "utils.py"), script_folder)
# shutil.copy(os.path.join(base_dir, "pipelines_submit.py"), script_folder)
shutil.copy(os.path.join(base_dir, "pipelines_create.py"), script_folder)
shutil.copy(os.path.join(base_dir, "train.py"), script_folder)
shutil.copytree(
    os.path.join(base_dir, "models"),
    os.path.join(base_dir, script_folder, "models"))
# shutil.copy(os.path.join(model_dir, "prednet.py"), script_folder)
# shutil.copy(os.path.join(base_dir, "keras_utils.py"), script_folder)
shutil.copy(os.path.join(base_dir, "data_preparation.py"), script_folder)
shutil.copy(os.path.join(base_dir, "register_prednet.py"), script_folder)
shutil.copy(
    os.path.join(base_dir, "register_classification_model.py"), script_folder
)
shutil.copy(os.path.join(base_dir, "config.json"), script_folder)

cpu_compute_name = config["cpu_compute"]
try:
    cpu_compute_target = AmlCompute(ws, cpu_compute_name)
    print("found existing compute target: %s" % cpu_compute_name)
except Exception:  # ComputeTargetException:
    print("creating new compute target")

    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",
        max_nodes=4,
        idle_seconds_before_scaledown=1800,
    )
    cpu_compute_target = ComputeTarget.create(
        ws, cpu_compute_name, provisioning_config
    )
    cpu_compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20
    )


# choose a name for your cluster
gpu_compute_name = config["gpu_compute"]
try:
    gpu_compute_target = AmlCompute(workspace=ws, name=gpu_compute_name)
    print("found existing compute target: %s" % gpu_compute_name)
except Exception as e:
    print("Creating a new compute target...")
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_NC6",
        max_nodes=10,
        idle_seconds_before_scaledown=1800,
    )

    # create the cluster
    gpu_compute_target = ComputeTarget.create(
        ws, gpu_compute_name, provisioning_config
    )

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it uses the scale settings for the
    # cluster
    gpu_compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20
    )

# use get_status() to get a detailed status for the current cluster.
print(cpu_compute_target.get_status().serialize())

cd = CondaDependencies.create()

# conda dependencies for compute targets
conda_dependencies = CondaDependencies.create(
    conda_packages=["cudatoolkit=10.1"],
    pip_packages=[
        "azure-storage-blob==2.1.0",
        "azureml-sdk",
        # "azureml-defaults==1.1.5",
        "hickle==3.4.3",
        "requests==2.21.0",
        "sklearn",
        "pandas",
        "numpy",
        "pillow==6.0.0",
        "tensorflow-gpu==1.15",
        "keras",
        "matplotlib",
        "seaborn",
    ]
)

env = Environment("prednet")
env.python.conda_dependencies = conda_dependencies
env.docker.enabled = True
env.register(ws)

# Runconfigs
runconfig = RunConfiguration(conda_dependencies=env.python.conda_dependencies)
# runconfig.environment = env
# conda_dependencies=env.python.conda_dependencies)
runconfig.environment.docker.enabled = True
runconfig.environment.docker.gpu_support = False
runconfig.environment.docker.base_image = DEFAULT_CPU_IMAGE
runconfig.environment.spark.precache_packages = False


create_pipelines = PythonScriptStep(
    name="create pipelines",
    script_name="pipelines_create.py",
    compute_target=cpu_compute_target,
    source_directory=script_folder,
    runconfig=runconfig,
    allow_reuse=False,
)
print("pipeline building step created")


# step 2, submit pipelines
# submit_pipelines = PythonScriptStep(
#     name="submit pipelines",
#     script_name="pipelines_submit.py",
#     compute_target=cpu_compute_target,
#     source_directory=script_folder,
#     runconfig=runconfig,
#     allow_reuse=False,
# )
# print("pipeline submit step created")

# submit_pipelines.run_after(create_pipelines)

pipeline = Pipeline(workspace=ws, steps=[create_pipelines])
print("Pipeline created")

pipeline.validate()
print("Validation complete")

pipeline_name = "prednet_master"
disable_pipeline(pipeline_name=pipeline_name, dry_run=False)
published_pipeline = pipeline.publish(name=pipeline_name)

print("pipeline id: ", published_pipeline.id)

datastore = ws.get_default_datastore()

with open("placeholder.txt", "w") as f:
    f.write(
        "This is just a placeholder to ensure "
        "that this path exists in the blobstore.\n"
    )

datastore.upload_files(
    [os.path.join(os.getcwd(), "placeholder.txt")],
    target_path="prednet/data/proprocessed/",
)

schedule = Schedule.create(
    workspace=ws,
    name=pipeline_name + "_sch",
    pipeline_id=published_pipeline.id,
    experiment_name="prednet_master",
    datastore=datastore,
    wait_for_provisioning=True,
    description="Datastore scheduler for Pipeline" + pipeline_name,
    path_on_datastore="prednet/data/preprocessed",
    polling_interval=60,
)

print("Created schedule with id: {}".format(schedule.id))

published_pipeline.submit(ws, published_pipeline.name)
