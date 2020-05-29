import argparse
import os
from azureml.core import Workspace, Run
from azureml.core.model import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    dest="model_path",
    default="./outputs/")
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="UCSDped1",
    help="the dataset that we are using",
    type=str,
    required=False,
)
args = parser.parse_args()
print("all args: ", args)

run = Run.get_context()
try:
    ws = run.experiment.workspace
except AttributeError:
    ws = Workspace.from_config()

model = Model.register(
    ws,
    os.path.join(args.model_path, "model.pkl"),
    model_name="clf_" + args.dataset)
