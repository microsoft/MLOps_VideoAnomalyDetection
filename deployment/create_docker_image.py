import json
from azureml.core.runconfig import CondaDependencies
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import ContainerImage, Image

# initialize workspace from config.json
ws = Workspace.from_config()

prednet_model_name = 'prednet_UCSDped1'
prednet_model_version = 2
logistic_regression_model_name = 'logistic_regression'
logistic_regression_model_version = 1

cd = CondaDependencies.create(
  pip_packages=[
    'keras',
    'tensorflow==1.15',
    'matplotlib',
    'hickle',
    'pandas',
    'azureml-sdk',
    "scikit-learn"])

cd.save_to_file(base_directory='./', conda_file_path='myenv.yml')

prednet_model = Model(
  ws,
  name=prednet_model_name,
  version=prednet_model_version)

logistic_regression_model = Model(
  ws,
  name=logistic_regression_model_name,
  version=logistic_regression_model_version)

img_config = ContainerImage.image_configuration(
  execution_script="score.py", 
  runtime="python", 
  conda_file="myenv.yml",
  dependencies=['prednet.py', 'aml_config/model.json'])

image_name = prednet_model_name.replace("_", "").lower()

print("Image name:", image_name)

image = Image.create(
  name=image_name,
  models=[prednet_model],
  image_config=img_config,
  workspace=ws)

image.wait_for_creation(show_output=True)


if image.creation_state != 'Succeeded':
  raise Exception('Image creation status: {image.creation_state}')

print('{}(v.{} [{}]) stored at {} with build log {}'.format(image.name, image.version, image.creation_state, image.image_location, image.image_build_log_uri))

# Writing the image details to /aml_config/image.json
image_json = {}
image_json['image_name'] = image.name
image_json['image_version'] = image.version
image_json['image_location'] = image.image_location

with open('aml_config/image.json', 'w') as outfile:
  json.dump(image_json, outfile)
