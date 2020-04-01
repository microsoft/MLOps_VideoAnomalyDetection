import os
import json
import azureml
import shutil
from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig,
    PrimaryMetricGoal,
)
from azureml.train.hyperdrive import choice, uniform
from azureml.train.dnn import TensorFlow
from azureml.train.estimator import Estimator

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

base_dir = "."
config_json = os.path.join(base_dir, "config.json")
with open(config_json, "r") as f:
    config = json.load(f)

# initialize workspace from config.json
ws = Workspace.from_config()

print(
    "Workspace name: " + ws.name,
    "Azure region: " + ws.location,
    "Subscription id: " + ws.subscription_id,
    "Resource group: " + ws.resource_group,
    sep="\n",
)

data_folder = "./data"

# folder for scripts that need to be uploaded to Aml compute
# target
script_folder = "./scripts"
if os.path.exists(script_folder):
    print("Deleting:", script_folder)
    shutil.rmtree(script_folder)
os.makedirs(script_folder)
# the training logic is in the keras_mnist.py file.
shutil.copy("./train.py", script_folder)
# shutil.copy("./data_utils.py", script_folder)
# shutil.copy("./prednet.py", script_folder)
# shutil.copy("./keras_utils.py", script_folder)
shutil.copytree(
    os.path.join(base_dir, "models"),
    os.path.join(base_dir, script_folder, "models"))


# create AML experiment
exp = Experiment(workspace=ws, name="prednet")

# upload data to default datastore
ds = ws.get_default_datastore()
ds.upload(
    src_dir="./data/preprocessed/UCSDped1",
    target_path="prednet/data/preprocessed/",
    overwrite=False,
    show_progress=True,
)

# choose a name for your cluster
cluster_name = config["gpu_compute"]

try:
    compute_target = AmlCompute(workspace=ws, name=cluster_name)
    print("Found existing compute target")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_NC6", max_nodes=10
    )

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout.
    # if no min node count is provided it uses the scale settings for
    # the cluster.
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20
    )

# use get_status() to get a detailed status for the current cluster.
print(compute_target.get_status().serialize())


# conda dependencies for compute targets
conda_dependencies = CondaDependencies.create(
    conda_packages=["cudatoolkit=10.1"],
    pip_packages=[
        "azure-storage-blob==2.1.0",
        "azureml-sdk==1.1.5",
        "azureml-defaults==1.1.5",
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

script_params = {
    "--data-folder": ds.path("prednet/data/preprocessed/UCSDped1").as_mount(),
    # "--compute_target": cluster_name,
}

est = Estimator(
    source_directory=script_folder,
    compute_target=compute_target,
    entry_script="train.py",
    # use_gpu=True,
    node_count=1,
    # conda_packages=gpu_cd.conda_packages,
    # pip_packages=gpu_cd.pip_packages,
    environment_definition=env,
    script_params=script_params,
)


ps = BayesianParameterSampling(
    {
        "--batch_size": choice(1, 2, 4, 10),
        "--filter_sizes": choice("3, 3, 3", "4, 4, 4", "5, 5, 5"),
        "--stack_sizes": choice(
            "48, 96, 192", "36, 72, 144", "12, 24, 48"
        ),
        "--learning_rate": uniform(1e-6, 1e-3),
        "--lr_decay": uniform(1e-9, 1e-2),
        "--freeze_layers": choice(
            "0, 1, 2", "1, 2, 3", "0, 1", "1, 2", "2, 3", "0", "3"
        ),
        "--transfer_learning": choice("True", "False"),
    }
)


hdc = HyperDriveConfig(
    estimator=est,
    hyperparameter_sampling=ps,
    primary_metric_name="val_loss",
    primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
    max_total_runs=30,
    max_concurrent_runs=5,
    max_duration_minutes=60 * 6,
)

hdr = exp.submit(config=hdc)

hdr.wait_for_completion(show_output=True)

best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(best_run)

# Writing the run id to /aml_config/run_id.json for use by a DevOps pipeline.
run_id = {}
run_id["run_id"] = best_run.id
run_id["experiment_name"] = best_run.experiment.name

# save run info
os.makedirs("aml_config", exist_ok=True)
with open("aml_config/run_id.json", "w") as outfile:
    json.dump(run_id, outfile)
