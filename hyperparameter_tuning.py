import os
import json
import azureml
import shutil
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveRunConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, loguniform
from azureml.train.dnn import TensorFlow

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# initialize workspace from config.json
ws = Workspace.from_config()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

data_folder = './data'

# folder for scripts that need to be uploaded to Aml compute target
script_folder = './scripts'
os.makedirs(script_folder, exist_ok=True)

# create AML experiment
exp = Experiment(workspace=ws, name='prednet')

# upload data to default datastore
ds = ws.get_default_datastore()
ds.upload(src_dir='./data', target_path='prednet/preprocessed/UCSDped1', overwrite=False, show_progress=True)

# choose a name for your cluster
cluster_name = "gpucluster"

try:
    compute_target = AmlCompute(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=10)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())


# the training logic is in the keras_mnist.py file.
shutil.copy('./train.py', script_folder)
shutil.copy('./data_utils.py', script_folder)
shutil.copy('./prednet.py', script_folder)
shutil.copy('./keras_utils.py', script_folder)


script_params = {
    '--data-folder': ds.path('prednet').as_mount(),
    '--compute_target': cluster_name
}

est = TensorFlow(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target,
                 pip_packages=['keras==2.0.8', 'theano', 'tensorflow==1.8.0', 'tensorflow-gpu==1.8.0', 'matplotlib', 'horovod', 'hickle'],
                 entry_script='train.py', 
                 use_gpu=True,
                 node_count=1)




# run = exp.submit(est)

# print(run)

# run.wait_for_completion(show_output=True)

ps = RandomParameterSampling(
    {
        '--batch_size': choice(2, 4, 8, 16),
        '--filter_sizes': choice("3 3 3", "4 4 4", "5 5 5"),
        '--stack_sizes': choice("48 96 192", "36 72 144", "12 24 48"),
        '--learning_rate': loguniform(-6, -1),
        '--lr_decay': loguniform(-9, -1)
    }
)

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)#, delay_evaluation=20)

hdc = HyperDriveRunConfig(estimator=est, 
                          hyperparameter_sampling=ps, 
                          policy=policy, 
                          primary_metric_name='val_loss', 
                          primary_metric_goal=PrimaryMetricGoal.MINIMIZE, 
                          max_total_runs=5,
                          max_concurrent_runs=50)

hdr = exp.submit(config=hdc)

hdr.wait_for_completion(show_output=True)

best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(best_run)

# Writing the run id to /aml_config/run_id.json for use by a DevOps pipeline.
run_id = {}
run_id['run_id'] = best_run.id
run_id['experiment_name'] = best_run.experiment.name

# save run info 
os.makedirs('aml_config', exist_ok = True)
with open('aml_config/run_id.json', 'w') as outfile:
    json.dump(run_id, outfile)