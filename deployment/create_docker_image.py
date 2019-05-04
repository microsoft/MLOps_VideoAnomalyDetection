import sys
import json
from azureml.core.runconfig import CondaDependencies
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import ContainerImage, Image


try:
    with open("aml_config/model.json") as f:
        config = json.load(f)
except:
    print('No new model to register thus no need to create new scoring image')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(0)

# initialize workspace from config.json
ws = Workspace.from_config()

model_name = 'prednet_UCSDped1' #config['model_name']
model_version = 2 # config['model_version']



cd = CondaDependencies.create(pip_packages=['keras==2.0.8', 'theano', 'tensorflow==1.8.0', 'matplotlib', 'hickle', 'pandas', 'azureml-sdk'])

cd.save_to_file(base_directory='./', conda_file_path='myenv.yml')

model = Model(ws, name=model_name, version=model_version)

img_config = ContainerImage.image_configuration(execution_script="score.py", 
                                               runtime="python", 
                                               conda_file="myenv.yml",
                                               dependencies=['prednet.py', 'keras_utils.py', 'aml_config/model.json'])

image_name = model_name.replace("_", "").lower()

print("Image name:", image_name)

image = Image.create(name = image_name,
                     models = [model], 
                     image_config = img_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)


if image.creation_state != 'Succeeded':
  raise Exception('Image creation status: {image.creation_state}')

print('{}(v.{} [{}]) stored at {} with build log {}'.format(image.name, image.version, image.creation_state, image.image_location, image.image_build_log_uri))

# Writing the image details to /aml_config/image.json
image_json = {}
image_json['image_name'] = image.name
image_json['image_version'] = image.version
image_json['image_location'] = image.image_location
with open('aml_config/image.json', 'w') as outfile:
  json.dump(image_json,outfile)

