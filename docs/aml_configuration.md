# Azure ML Configuration

This requires the following steps:
1. Configure [AML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace)
1. Create an Azure Service Principal
1. Upload the data to the default datastore of your workspace


## Configure AML workspace

First step is to attach to an AML workspace.

For you convenience, we recommend you start by moving the file `config/config_sample.json` to `config.json` (in root of repo). All you need to fill in is your subscription ID, and specify a pre-existing resource group, but feel free to change any of the details to set up a workspace as desired. You can then execute the file `create_workspace.py` to create your workspace. Do make sure to pay attention to the output when running the script, as it may include further instructions or error messages.

See [documentation](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb) for more info.

## Create Azure Service Principal

This is necessary for non-interactive authentication.  Create a service principal and give it *Contributor* access to your workspace (see [documentation](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb)).

Store your service principal and tenant information in `config.json` in the root of the repository as follows:
```
"service_principal_id": "",
"service_principal_password": "",
"tenant_id": "",
```

## Upload the data to the default datastore of your workspace

We upload the training data so that it can be mounted as remote drives in the aml compute targets. You can use the method `upload_data` in `utils.py`  for that.

For example:
```
conda activate prednet
python3
from utils import upload_data
upload_data()
```
