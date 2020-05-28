import argparse
import os
import json
from azureml.core import Workspace, Run, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_metrics',
    dest="data_metrics",
    default="data_metrics")
# parser.add_argument(
#     '--prednet_path',
#     dest="prednet_path",
#     default="prednet_path")

args = parser.parse_args()
print("all args: ", args)

run = Run.get_context()
try:
    ws = run.experiment.workspace
except AttributeError:
    ws = Workspace.from_config()

data_metrics = os.path.dirname(args.data_metrics)

with open(os.path.join(data_metrics, 'data_metrics')) as f:
    metrics = json.load(f)

best_loss = 1.0
best_run_id = None

print(metrics)
for run in metrics.keys():
    try:
        loss = metrics[run]['val_loss'][-1]
        if loss < best_loss:
            best_loss = loss
            best_run_id = run
    except Exception:
        print("WARNING: Could get val_los for run_id", run)
        pass

print("best run", best_run_id, best_loss)


# start an Azure ML run
run = Run.get_context()
run_details = run.get_details()

environment_definition = run_details['runDefinition']['environment']
experiment_name = environment_definition['name'].split()[1]

exp = Experiment(ws, name=experiment_name)
best_run = Run(exp, best_run_id)

model_dir = 'outputs/model'  # 'outputs'

# register the model
if best_run_id:
    tags = {}
    tags['run_id'] = best_run_id
    tags['val_loss'] = metrics[best_run_id]['val_loss'][-1]
    model = best_run.register_model(model_name=experiment_name,
                                    model_path=model_dir,
                                    tags=tags)
    # model.download(target_dir=args.prednet_path)
else:
    raise Exception("Couldn't not find a model to register."
                    "Probably because no run completed")
