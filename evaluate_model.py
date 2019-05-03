import os, json
from azureml.core import Workspace, Run, Experiment
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.exceptions._azureml_exception import ModelNotFoundException

from azureml.core.runconfig import CondaDependencies
from azureml.core.image import ContainerImage, Image

from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice

base_dir = '.'

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
    svr_pr = None
    pass
    
ws = Workspace.from_config(path=config_json, auth=svc_pr)

model_list = Model.list(ws)
model_dict = {}
for i, model in enumerate(model_list):
    if model.name in model_dict.keys():
        model_dict[model.name]['newest_version'] = max(model_dict[model.name]['newest_version'], model.version)
    else:
        model_dict[model.name] = {}
        model_dict[model.name]['newest_version'] = model.version

# for each model in the workspace, find the newest model and the production model (second newest) 
production_models = {}
new_models = {}
for model_name in model_dict.keys():
    for i, model in enumerate(model_list):
        if model.name == model_name:
            if model.version < model_dict[model_name]['newest_version']:
                if model_name in production_models.keys():
                    if model.version > production_models[model_name].version:
                        production_models[model_name] = model
                else:
                    production_models[model_name] = model
            else:
                new_models[model_name] = model

promoted_models = {}
for model_name in model_dict.keys():
    if model_name in production_models.keys():
        production_model = production_models[model_name]
        new_model = new_models[model_name]
        if new_model.tags.get('val_loss') < production_model('val_loss'):
            promoted_models[model_name] = new_models[model_name]
    else:
        promoted_models[model_name] = new_models[model_name]

# delete webservice that need to be updated
webservice_list = ws.webservices
for model_name, model in promoted_models.items():
    webservice_name = model_name.replace("_", "").lower()
    if webservice_name in ws.webservices:
        ws.webservices[webservice_name].delete()

## create docker images
cd = CondaDependencies.create(pip_packages=['keras==2.0.8', 'theano', 'tensorflow==1.8.0', 'matplotlib', 'hickle', 'pandas', 'azureml-sdk'])

os.makedirs('aml_config', exist_ok=True)
cd.save_to_file(base_directory='aml_config', conda_file_path='myenv.yml')

img_config = ContainerImage.image_configuration(execution_script="score.py", 
                                               runtime="python", 
                                               conda_file="aml_config/myenv.yml",
                                               dependencies=['prednet.py', 'keras_utils.py']) #, 'aml_config/model.json'])

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

## deploy to ACI

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=1, 
                                               tags={'name':'prednet', 'framework': 'Keras'},
                                               description='Prednet')


aci_service_name = image_name
print(aci_service_name)

service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
service.wait_for_deployment(True)

print('Deployed ACI Webservice: {} \nWebservice Uri: {}'.format(service.name, service.scoring_uri))

print("Done")