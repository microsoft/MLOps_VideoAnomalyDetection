from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import Image
from azureml.core import Workspace
import sys
import json

# Get workspace
ws = Workspace.from_config()

# Get the Image to deploy details
try:
    with open("aml_config/image.json") as f:
        config = json.load(f)
except:
    print('No new model, thus no deployment on ACI')
    sys.exit(0)

image_name = config['image_name']
image_version = config['image_version']

images = Image.list(workspace=ws)
image, = (m for m in images if m.version==image_version and m.name == image_name)
print('From image.json, Image used to deploy webservice on ACI: {}\nImage Version: {}\nImage Location = {}'.format(image.name, image.version, image.image_location))


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

#service=Webservice(name ='aciws0622', workspace =ws)
# Writing the ACI details to /aml_config/aci_webservice.json
aci_webservice = {}
aci_webservice['aci_name'] = service.name
aci_webservice['aci_url'] = service.scoring_uri
with open('aml_config/aci_webservice.json', 'w') as outfile:
  json.dump(aci_webservice,outfile)

