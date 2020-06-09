import os
import shutil
import argparse

from azureml.core import Workspace, Environment, Run
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.core.schedule import Schedule
from azureml.pipeline.steps import PythonScriptStep, HyperDriveStep
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig,
    PrimaryMetricGoal,
    choice,
    uniform,
)
from azureml.train.estimator import Estimator
from azure.storage.blob import BlockBlobService

from azureml.core import VERSION
print("azureml.core.VERSION", VERSION)


def build_prednet_pipeline(dataset, ws):
    print(
        "building pipeline for dataset %s in workspace %s" % (dataset, ws.name)
    )

    base_dir = "."

    def_blob_store = ws.get_default_datastore()

    # folder for scripts that need to be uploaded to Aml compute target
    script_folder = "./scripts"
    os.makedirs(script_folder, exist_ok=True)

    shutil.copytree(
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, script_folder, "models"))
    shutil.copy(os.path.join(base_dir, "train.py"), script_folder)
    shutil.copy(os.path.join(base_dir, "data_preparation.py"), script_folder)
    shutil.copy(os.path.join(base_dir, "register_prednet.py"), script_folder)
    shutil.copy(os.path.join(base_dir, "batch_scoring.py"), script_folder)
    shutil.copy(os.path.join(base_dir, "train_clf.py"), script_folder)
    shutil.copy(os.path.join(base_dir, "register_clf.py"), script_folder)

    cpu_compute_name = args.cpu_compute_name
    cpu_compute_target = AmlCompute(ws, cpu_compute_name)
    print("found existing compute target: %s" % cpu_compute_name)

    # use get_status() to get a detailed status for the current cluster.
    print(cpu_compute_target.get_status().serialize())

    # choose a name for your cluster
    gpu_compute_name = args.gpu_compute_name

    gpu_compute_target = AmlCompute(workspace=ws, name=gpu_compute_name)
    print(gpu_compute_target.get_status().serialize())

    env = Environment.get(ws, "prednet")

    # Runconfigs
    runconfig = RunConfiguration()
    runconfig.environment = env
    print("PipelineData object created")

    # DataReference to where raw data is stored.
    raw_data = DataReference(
        datastore=def_blob_store,
        data_reference_name="raw_data",
        path_on_datastore=os.path.join("prednet", "data", "raw_data"),
    )
    print("DataReference object created")

    # Naming the intermediate data as processed_data and assigning it to the
    # variable processed_data.
    preprocessed_data = PipelineData(
        "preprocessed_data", datastore=def_blob_store
    )
    data_metrics = PipelineData("data_metrics", datastore=def_blob_store)
    hd_child_cwd = PipelineData(
        "prednet_model_path",
        datastore=def_blob_store)
    # prednet_path = PipelineData("outputs", datastore=def_blob_store)
    scored_data = PipelineData("scored_data", datastore=def_blob_store)
    model_path = PipelineData("model_path", datastore=def_blob_store)

    # prepare dataset for training/testing recurrent neural network
    data_prep = PythonScriptStep(
        name="prepare_data",
        script_name="data_preparation.py",
        arguments=[
            "--raw_data",
            raw_data,
            "--preprocessed_data",
            preprocessed_data,
            "--dataset",
            dataset,
        ],
        inputs=[raw_data],
        outputs=[preprocessed_data],
        compute_target=cpu_compute_target,
        source_directory=script_folder,
        runconfig=runconfig,
        allow_reuse=True,
    )
    # data_prep.run_after(video_decoding)

    print("data_prep step created")

    est = Estimator(
        source_directory=script_folder,
        compute_target=gpu_compute_target,
        entry_script="train.py",
        node_count=1,
        environment_definition=env,
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
            # "--fine_tuning": choice("True", "False"),
        }
    )

    hdc = HyperDriveConfig(
        estimator=est,
        hyperparameter_sampling=ps,
        primary_metric_name="val_loss",
        primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
        max_total_runs=3,
        max_concurrent_runs=3,
        max_duration_minutes=60 * 6,
    )

    train_prednet = HyperDriveStep(
        "train_w_hyperdrive",
        hdc,
        estimator_entry_script_arguments=[
            "--preprocessed_data",
            preprocessed_data,
            "--remote_execution",
            "--dataset",
            dataset,
        ],
        inputs=[preprocessed_data],
        outputs=[hd_child_cwd],
        metrics_output=data_metrics,
        allow_reuse=True,
    )
    train_prednet.run_after(data_prep)

    register_prednet = PythonScriptStep(
        name="register_prednet",
        script_name="register_prednet.py",
        arguments=[
            "--data_metrics",
            data_metrics,
            ],
        compute_target=cpu_compute_target,
        inputs=[data_metrics, hd_child_cwd],
        source_directory=script_folder,
        allow_reuse=True,
    )
    register_prednet.run_after(train_prednet)

    batch_scoring = PythonScriptStep(
        name="batch_scoring",
        script_name="batch_scoring.py",
        arguments=[
            "--preprocessed_data",
            preprocessed_data,
            "--scored_data",
            scored_data,
            "--dataset",
            dataset,
            # "--prednet_path",
            # prednet_path
            ],
        compute_target=gpu_compute_target,
        inputs=[preprocessed_data],
        outputs=[scored_data],
        source_directory=script_folder,
        runconfig=runconfig,
        allow_reuse=True,
    )
    batch_scoring.run_after(register_prednet)

    train_clf = PythonScriptStep(
        name="train_clf",
        script_name="train_clf.py",
        arguments=[
            "--preprocessed_data",
            preprocessed_data,
            "--scored_data",
            scored_data,
            "--model_path",
            model_path],
        compute_target=cpu_compute_target,
        inputs=[preprocessed_data, scored_data],
        outputs=[model_path],
        source_directory=script_folder,
        runconfig=runconfig,
        allow_reuse=True,
    )
    train_clf.run_after(batch_scoring)

    register_clf = PythonScriptStep(
        name="register_clf",
        script_name="register_clf.py",
        arguments=[
            "--model_path",
            model_path],
        inputs=[model_path],
        compute_target=cpu_compute_target,
        source_directory=script_folder,
        allow_reuse=True,
        runconfig=runconfig,
    )
    register_clf.run_after(train_clf)

    pipeline = Pipeline(
        workspace=ws,
        steps=[
            data_prep,
            train_prednet,
            register_prednet,
            batch_scoring,
            train_clf,
            register_clf,
        ],
    )
    pipeline.validate()

    pipeline_name = "prednet_" + dataset
    published_pipeline = pipeline.publish(name=pipeline_name)

    _ = Schedule.create(
        workspace=ws,
        name=pipeline_name + "_sch",
        pipeline_id=published_pipeline.id,
        experiment_name=pipeline_name,
        datastore=def_blob_store,
        wait_for_provisioning=True,
        description="Datastore scheduler for Pipeline" + pipeline_name,
        path_on_datastore=os.path.join(
            "prednet/data/raw_data", dataset, "Train"
        ),
        polling_interval=60 * 24,
    )

    published_pipeline.submit(ws, pipeline_name)


parser = argparse.ArgumentParser(description="Process input arguments")
parser.add_argument(
    "--cpu_compute_name",
    default="cpu-cluster",
    type=str,
    dest="cpu_compute_name",
    help="name of cpu cluster",
)
parser.add_argument(
    "--gpu_compute_name",
    default="gpu-cluster",
    type=str,
    dest="gpu_compute_name",
    help="name of gpu cluster",
)

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

keyvault = ws.get_default_keyvault()
tenant_id = keyvault.get_secret('tenantId')
service_principal_id = keyvault.get_secret("servicePrincipalId")
service_principal_password = keyvault.get_secret("servicePrincipalPassword")

svc_pr = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=service_principal_id,
    service_principal_password=service_principal_password,
)

ws = Workspace(ws.subscription_id, ws.resource_group, ws.name, auth=svc_pr)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

def_blob_store = ws.get_default_datastore()

print("Blobstore's name: {}".format(def_blob_store.name))

# create a list of datasets stored in blob
print("Checking for new datasets")
blob_service = BlockBlobService(
    def_blob_store.account_name, def_blob_store.account_key
)
generator = blob_service.list_blobs(
    def_blob_store.container_name, prefix="prednet/data/raw_data"
)
datasets = []
for blob in generator:
    dataset = blob.name.split("/")[3]
    if (
        dataset not in datasets
        and dataset.startswith("UCSD")
        and not dataset.endswith("txt")
    ):
        datasets.append(dataset)
        print("Found dataset:", dataset)

# Get all published pipeline objects in the workspace
all_pub_pipelines = PublishedPipeline.list(ws)

# Create a list of datasets for which we have (old) and don't have (new) a
# published pipeline
old_datasets = []
new_datasets = []
for dataset in datasets:
    for pub_pipeline in all_pub_pipelines:
        if pub_pipeline.name.endswith(dataset):
            old_datasets.append(dataset)
    if dataset not in old_datasets:
        new_datasets.append(dataset)

for dataset in new_datasets:
    print("Creating pipeline for dataset", dataset)
    build_prednet_pipeline(dataset, ws)
