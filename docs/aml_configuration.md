# Azure ML Configuration

This requires the following steps:
1. Configure AML workspace
1. Create an Azure Service Principal
1. Upload the data to the default datastore of your workspace


## Configure AML workspace

First step is to attach to an AML workspace.

For you convenience, we recommend you start by moving the file `config/config_sample.json` to `config.json` (in root of repo). All you need to fill out is your subscription id. You can then execute the file `create_workspace.py` to create your workspace. Do make sure to pay attention to the output when running the script, as it may include further instructions or error messages.

See [documentation](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb) for more info.

## Create Azure Service Principal

This is necessary for non-interactive authentication.  Create a service principal and give it *Contributor* access to your workspace (see [documentation](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb)).

Store the information in a [config.json](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/how-to-configure-environment.md#create-a-workspace-configuration-file) file in the root directory of this repository.

Once you have this info, you can add it to your `config.json` file by adding these three lines:
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
python
from utils import upload_data
upload_data()
```
