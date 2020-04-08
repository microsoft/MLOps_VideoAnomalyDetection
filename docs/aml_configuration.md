# Azure ML Configuration

This requires the following steps:
1. Configure AML workspace
1. Create an Azure Service Principal
1. Upload the data to the default datastore of your workspace


## Configure AML workspace

First step is to attach to an AML workspace. 

If you don't have one yet, you can create one using this [documentation](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/setup-create-workspace.md#sdk).

## Create Azure Service Principal

This is necessary for non-interactive authentication.  Create a service principal and give it *Contributor* access to your workspace (see [documentation](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb)).

Store the information in a [config.json](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/how-to-configure-environment.md#create-a-workspace-configuration-file) file in the root directory of this repository.

This repository contains a sample configuration file (`config/config_sample.json`) that shows which information needs to be included.

## Upload the data to the default datastore of your workspace

We upload the training data so that it can be mounted as remote drives in the aml compute targets. You can use the method `upload_data` in `utils.py`  for that.
