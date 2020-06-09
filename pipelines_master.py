import json
import os
import shutil
from utils import disable_pipeline

from azureml.core import Workspace, Environment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import (
    CondaDependencies,
    RunConfiguration
)
from azureml.core.runconfig import DEFAULT_GPU_IMAGE

from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core.schedule import Schedule

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

ws = Workspace(
    config["subscription_id"],
    config["resource_group"],
    config["workspace_name"],
    auth=auth
)

print(ws.get_details)

keyvault = ws.get_default_keyvault()
keyvault.set_secret("tenantID", config["tenant_id"])
keyvault.set_secret("servicePrincipalId", config["service_principal_id"])
keyvault.set_secret(
    "servicePrincipalPassword",
    config["service_principal_password"])

# folder for scripts that need to be uploaded to Aml compute target
script_folder = "./scripts/"
if os.path.exists(script_folder):
    print("Deleting:", script_folder)
    shutil.rmtree(script_folder)
os.makedirs(script_folder)

shutil.copy(os.path.join(base_dir, "utils.py"), script_folder)
shutil.copy(os.path.join(base_dir, "pipelines_slave.py"), script_folder)
shutil.copy(os.path.join(base_dir, "train.py"), script_folder)
shutil.copytree(
    os.path.join(base_dir, "models"),
    os.path.join(base_dir, script_folder, "models"))
shutil.copy(os.path.join(base_dir, "data_preparation.py"), script_folder)
shutil.copy(os.path.join(base_dir, "register_prednet.py"), script_folder)
shutil.copy(os.path.join(base_dir, "batch_scoring.py"), script_folder)
shutil.copy(os.path.join(base_dir, "train_clf.py"), script_folder)
shutil.copy(os.path.join(base_dir, "register_clf.py"), script_folder)

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
        vm_priority="lowpriority"
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
except Exception:
    print("Creating a new compute target...")
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_NC6",
        max_nodes=10,
        idle_seconds_before_scaledown=1800,
        vm_priority="lowpriority"
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
    conda_packages=["cudatoolkit=10.0"],
    pip_packages=[
        "azure-storage-blob==2.1.0",
        "azureml-sdk",
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
env.docker.base_image = DEFAULT_GPU_IMAGE
env.register(ws)

# Runconfigs
runconfig = RunConfiguration()
runconfig.environment = env
print("PipelineData object created")


create_pipelines = PythonScriptStep(
    name="create pipelines",
    script_name="pipelines_slave.py",
    compute_target=cpu_compute_target,
    arguments=[
        "--cpu_compute_name",
        cpu_compute_name,
        "--gpu_compute_name",
        gpu_compute_name
    ],
    source_directory=script_folder,
    runconfig=runconfig,
    allow_reuse=False,
)
print("pipeline building step created")

pipeline = Pipeline(workspace=ws, steps=[create_pipelines])
print("Pipeline created")

pipeline.validate()
print("Validation complete")

pipeline_name = "prednet_master"
disable_pipeline(pipeline_name=pipeline_name, dry_run=False)
disable_pipeline(pipeline_name="prednet_UCSDped1", dry_run=False)
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
    target_path="prednet/data/raw_data/",
)

schedule = Schedule.create(
    workspace=ws,
    name=pipeline_name + "_sch",
    pipeline_id=published_pipeline.id,
    experiment_name="prednet_master",
    datastore=datastore,
    wait_for_provisioning=True,
    description="Datastore scheduler for Pipeline" + pipeline_name,
    path_on_datastore="prednet/data/raw_data",
    polling_interval=5,
)

print("Created schedule with id: {}".format(schedule.id))

published_pipeline.submit(ws, published_pipeline.name)
