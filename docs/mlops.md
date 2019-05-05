# MLOps

To get started with MLOps extension for Azure DevOps, go to [https://aka.ms/mlops](https://aka.ms/mlops).

The `config` folder contains two files that will be useful for you to get started. 

`azure-pipelines.yml` - definition of an Azure build pipeline
`config_sample.json` - configuration and credentials for working with aml compute

After you filled in all the empty quotes in the config_sample.json file, save it under e.g. <your_name>_config.json.

Then go into azure-pipelines.yml, and make sure that you change the name of the config.json to whatever you picked here.
