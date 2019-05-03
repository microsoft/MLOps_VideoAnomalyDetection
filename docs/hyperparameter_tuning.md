# Hyperparameter Tuning with HyperDrive

> file: `hyperparameter_tuning.py`
> 
Let's see whether we can improve the performance of our model by tuning the hyperparameters.

This requires the following steps:
1. Configure AML workspace
2. Upload the data to the default datastore of your workspace
3. Define a remote AMLCompute compute target
4. Prepare scripts for uploading to compute target
5. Define Tensorflow estimator
6. Configure HyperDrive run
7. Submit job for execution.


## Configure AML workspace

First step is to attach to an AML workspace. 

If you don't have one yet, you can create one using this [documentation](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/setup-create-workspace.md#sdk).

We recommend storing the information in a [config.json](https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/service/how-to-configure-environment.md#create-a-workspace-configuration-file) file in the root directory of this repository.


## Upload the data to the default datastore of your workspace

We upload the training data so that it can be mounted as remote drives in the aml compute targets.

## Define a remote AMLCompute compute target

We are using a `Standard_NC6` virtual machine. It is inexpensive and includes a GPU powerful enough for this purpose.

## Prepare scripts for uploading to compute target

The training script and dependencies have to be available to the job running on the compute target. 

## Define Tensorflow estimator

HyperDrive works best if we use an Estimator specifically defined for tensorflow models. 

## Configure HyperDrive run

Next, we define which hyperparameters to search over, and which strategy to use for searching.  Here, we are using `RandomParameterSampling` and a `BanditPolicy`.

## Submit job for execution.

Now everything is good to go.