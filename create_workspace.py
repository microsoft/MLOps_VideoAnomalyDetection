from azureml.core import Workspace
import json

with open('config.json', 'r') as f:
    config = json.load(f)

subscription_id = config['subscription_id']
resource_group = config['resource_group']
workspace_name = config['workspace_name']
workspace_region = config['workspace_region']

ws = Workspace.create(workspace_name, subscription_id = subscription_id, resource_group = resource_group, exist_ok=True)

ws.write_config()
