# you can use this scrip to train remotely
import os
import json
import shutil
from azureml.core import Workspace, Run, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.runconfig import DEFAULT_GPU_IMAGE, DEFAULT_CPU_IMAGE
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.train.dnn import TensorFlow


base_dir = '.'

# folder for scripts that need to be uploaded to Aml compute target
script_folder = './scripts'
os.makedirs(script_folder, exist_ok=True)

shutil.copy(os.path.join(base_dir, 'train.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'data_utils.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'prednet.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'keras_utils.py'), script_folder)
shutil.copy(os.path.join(base_dir, '../config.json'), script_folder)


config_json = os.path.join(base_dir, '../config.json')
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


# choose a name for your cluster
gpu_compute_name = config['gpu_compute']

try:
    gpu_compute_target = AmlCompute(workspace=ws, name=gpu_compute_name)
    print("found existing compute target: %s" % gpu_compute_name)
except ComputeTargetException:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                                max_nodes=5,
                                                                idle_seconds_before_scaledown=1800)

    # create the cluster
    gpu_compute_target = ComputeTarget.create(ws, gpu_compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    gpu_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

est = TensorFlow(source_directory=script_folder,
                compute_target=gpu_compute_target,
                pip_packages=['keras==2.0.8', 'theano', 'tensorflow==1.8.0', 'tensorflow-gpu==1.8.0', 'matplotlib', 'horovod', 'hickle'],
                entry_script='train.py', 
                use_gpu=True,
                node_count=1,
                script_params={"--remote_execution": None, "--data-folder": config["data_folder"]}
                )
                
experiment_name = "prednet_train"

exp = Experiment(ws, experiment_name)

run = exp.submit(est)

run.wait_for_completion(show_output=True)

print("done")