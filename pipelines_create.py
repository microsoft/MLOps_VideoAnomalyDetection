import os
import json
import azureml
import shutil
import socket
from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.pipeline.steps import HyperDriveStep
from azureml.pipeline.core import PublishedPipeline
from azureml.train.hyperdrive import choice, loguniform
from azureml.train.estimator import Estimator
from azure.storage.blob import BlockBlobService
from azureml.core.runconfig import DEFAULT_GPU_IMAGE, DEFAULT_CPU_IMAGE
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import ScriptRunConfig
from azureml.core.container_registry import ContainerRegistry
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule

from azureml.core import VERSION
print("azureml.core.VERSION", VERSION)

def build_pipeline(dataset, ws, config):
    print("building pipeline for dataset %s in workspace %s" % (dataset, ws.name))

    base_dir = '.'
        
    def_blob_store = ws.get_default_datastore()

    # folder for scripts that need to be uploaded to Aml compute target
    script_folder = './scripts'
    os.makedirs(script_folder, exist_ok=True)
    
    shutil.copy(os.path.join(base_dir, 'video_decoding.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'pipelines_submit.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'pipelines_create.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'train.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'data_utils.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'prednet.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'keras_utils.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'data_preparation.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'model_registration.py'), script_folder)
    shutil.copy(os.path.join(base_dir, 'config.json'), script_folder)
    
    cpu_compute_name = config['cpu_compute']
    try:
        cpu_compute_target = AmlCompute(ws, cpu_compute_name)
        print("found existing compute target: %s" % cpu_compute_name)
    except:# ComputeTargetException:
        print("creating new compute target")
        
        provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                                    max_nodes=4,
                                                                    idle_seconds_before_scaledown=1800)    
        cpu_compute_target = ComputeTarget.create(ws, cpu_compute_name, provisioning_config)
        cpu_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
    # use get_status() to get a detailed status for the current cluster. 
    print(cpu_compute_target.get_status().serialize())

    # choose a name for your cluster
    gpu_compute_name = config['gpu_compute']

    try:
        gpu_compute_target = AmlCompute(workspace=ws, name=gpu_compute_name)
        print("found existing compute target: %s" % gpu_compute_name)
    except: 
        print('Creating a new compute target...')
        provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                                    max_nodes=10,
                                                                    idle_seconds_before_scaledown=1800)

        # create the cluster
        gpu_compute_target = ComputeTarget.create(ws, gpu_compute_name, provisioning_config)

        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it uses the scale settings for the cluster
        gpu_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    # use get_status() to get a detailed status for the current cluster. 
    try:
        print(gpu_compute_target.get_status().serialize())
    except BaseException as e:
        print("Could not get status of compute target.")
        print(e)

    # conda dependencies for compute targets
    cpu_cd = CondaDependencies.create(conda_packages=["py-opencv=3.4.2"], pip_indexurl='https://azuremlsdktestpypi.azureedge.net/sdk-release/Candidate/604C89A437BA41BD942B4F46D9A3591D', pip_packages=["azure-storage-blob==1.5.0", "hickle==3.4.3", "requests==2.21.0", "sklearn", "pandas==0.24.2", "azureml-sdk", "numpy==1.16.2", "pillow==6.0.0"])
    
    # Runconfigs
    cpu_compute_run_config = RunConfiguration(conda_dependencies=cpu_cd)
    cpu_compute_run_config.environment.docker.enabled = True
    cpu_compute_run_config.environment.docker.gpu_support = False
    cpu_compute_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
    cpu_compute_run_config.environment.spark.precache_packages = False

    print("PipelineData object created")

    # DataReference to where video data is stored.
    video_data = DataReference(
        datastore=def_blob_store,
        data_reference_name="video_data",
        path_on_datastore=os.path.join("prednet", "data", "video", dataset))
    print("DataReference object created")
        
    # Naming the intermediate data as processed_data1 and assigning it to the variable processed_data1.
    raw_data = PipelineData("raw_video_fames", datastore=def_blob_store)
    preprocessed_data = PipelineData("preprocessed_video_frames", datastore=def_blob_store)
    data_metrics = PipelineData("data_metrics", datastore=def_blob_store)
    data_output = PipelineData("output_data", datastore=def_blob_store)

    # prepare dataset for training/testing prednet
    video_decoding = PythonScriptStep(
        name='decode_videos',
        script_name="video_decoding.py", 
        arguments=["--input_data", video_data, "--output_data", raw_data],
        inputs=[video_data],
        outputs=[raw_data],
        compute_target=cpu_compute_target, 
        source_directory=script_folder,
        runconfig=cpu_compute_run_config,
        allow_reuse=True,
        hash_paths=['.']
    )
    print("video_decode step created")

    # prepare dataset for training/testing recurrent neural network
    data_prep = PythonScriptStep(
        name='prepare_data',
        script_name="data_preparation.py", 
        arguments=["--input_data", raw_data, "--output_data", preprocessed_data],
        inputs=[raw_data],
        outputs=[preprocessed_data],
        compute_target=cpu_compute_target, 
        source_directory=script_folder,
        runconfig=cpu_compute_run_config,
        allow_reuse=True,
        hash_paths=['.']
    )
    data_prep.run_after(video_decoding)

    print("data_prep step created")


    # configure access to ACR for pulling our custom docker image
    acr = ContainerRegistry()
    acr.address = config['acr_address']
    acr.username = config['acr_username']
    acr.password = config['acr_password']
    
    est = Estimator(source_directory=script_folder,
                    compute_target=gpu_compute_target,
                    entry_script='train.py', 
                    use_gpu=True,
                    node_count=1
                    )

    ps = RandomParameterSampling(
        {
            '--batch_size': choice(1, 2, 4, 8),
            '--filter_sizes': choice("3, 3, 3", "4, 4, 4", "5, 5, 5"),
            '--stack_sizes': choice("48, 96, 192", "36, 72, 144", "12, 24, 48"), #, "48, 96"),
            '--learning_rate': loguniform(-6, -1),
            '--lr_decay': loguniform(-9, -1),
            '--freeze_layers': choice("0, 1, 2", "1, 2, 3", "0, 1", "1, 2", "2, 3", "0", "3"),
            '--transfer_learning': choice("True", "False")
        }
    )

    policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1, delay_evaluation=10)

    hdc = HyperDriveConfig(estimator=est, 
                            hyperparameter_sampling=ps, 
                            policy=policy, 
                            primary_metric_name='val_loss', 
                            primary_metric_goal=PrimaryMetricGoal.MINIMIZE, 
                            max_total_runs=10,
                            max_concurrent_runs=5, 
                            max_duration_minutes=60*6
                            )

    hd_step = HyperDriveStep("train_w_hyperdrive",
        hdc,
        estimator_entry_script_arguments=[
            '--data-folder', preprocessed_data, 
            '--remote_execution',
            '--dataset', dataset
            ],
        inputs=[preprocessed_data],
        metrics_output = data_metrics,
        allow_reuse=True
    )
    hd_step.run_after(data_prep)

    registration_step = PythonScriptStep(
        name='register_model',
        script_name='model_registration.py',
        arguments=['--input_dir', data_metrics, '--output_dir', data_output],
        compute_target=cpu_compute_target,
        inputs=[data_metrics],
        outputs=[data_output],
        source_directory=script_folder,
        allow_reuse=True,
        hash_paths=['.']
    )
    registration_step.run_after(hd_step)

    pipeline = Pipeline(workspace=ws, steps=[video_decoding, data_prep, hd_step, registration_step])
    print ("Pipeline is built")

    pipeline.validate()
    print("Simple validation complete") 

    pipeline_name = 'prednet_' + dataset
    published_pipeline = pipeline.publish(name=pipeline_name)
    

    schedule = Schedule.create(workspace=ws, name=pipeline_name + "_sch",
                            pipeline_id=published_pipeline.id, 
                            experiment_name=pipeline_name,
                            datastore=def_blob_store,
                            wait_for_provisioning=True,
                            description="Datastore scheduler for Pipeline" + pipeline_name,
                            path_on_datastore=os.path.join('prednet/data/video', dataset, 'Train'),
                            polling_interval=60
                            )

    return pipeline_name



# start of script (main)

config_json = 'config.json'
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

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

def_blob_store  = ws.get_default_datastore()

print("Blobstore's name: {}".format(def_blob_store.name))

# create a list of datasets stored in blob
print("Checking for new datasets")
blob_service = BlockBlobService(def_blob_store.account_name, def_blob_store.account_key)
generator = blob_service.list_blobs(def_blob_store.container_name, prefix="prednet/data/video")
datasets = []
for blob in generator:
    dataset = blob.name.split('/')[3]
    if dataset not in datasets and dataset.startswith("UCSD") and not dataset.endswith("txt"):
        datasets.append(dataset)
        print("Found dataset:", dataset)

# Get all published pipeline objects in the workspace
all_pub_pipelines = PublishedPipeline.list(ws)

# Create a list of datasets for which we have (old) and don't have (new) a published pipeline
old_datasets = []
new_datasets = []
for dataset in datasets:
    for pub_pipeline in all_pub_pipelines:
        if pub_pipeline.name.endswith(dataset):
            old_datasets.append(dataset)
    if not dataset in old_datasets:
        new_datasets.append(dataset)

for dataset in new_datasets:
    print("Creating pipeline for dataset", dataset)
    build_pipeline(dataset, ws, config)
