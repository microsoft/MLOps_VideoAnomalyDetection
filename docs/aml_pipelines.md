# Transfer Learning w/ AML Pipelines

You can run the training script `train.py` locally to train your model.  Let's figure out how to scale our solution such that whenever a new dataset is uploaded to blob store, a new anomaly detection is automatically created for this dataset.

We create two AML pipelines to scale our solution. 

First, we create a master AML pipeline (`pipeline_master.py`). This pipeline monitors the Azure Blob Storage container for new data.  Whenever new data is added, it checks whether this is from a new location for which we don't have a model yet. If that is the case, it creates a new AML pipeline for that dataset, by calling (`pipeline_slave.py`).

## AML pipeline for training the model

The AML pipeline for training a model is defined in `pipelines_slave.py`.

It contains the following steps:
1. data_prep - scale and crop images so that their size matches the size of the input layer of our model.
1. train_w_hyperdrive - train the model using hyperdrive
1. register_prednet - register the model in the AML workspace for later deployment
1. batch_scoring - perform batch scoring on the test data.
1. train_classifier - the test data is labeled for whether a video frame contains anomalies.  We use train a classifier that uses the errors of the neural network model to predict whether a video frame contains an anomaly. 
1. register_classifier - register the trained classifier in the AML model registry

## AML pipeline for dynamic generation of new training pipelines

The AML pipeline for this is defined in `pipelines_master.py`.

This pipeline contains the following steps:
1. create_pipelines - to create and publish a new training pipeline
1. submit_pipelines - to submit all newly created pipelines for execution

Submitting the pipelines for execution is of course optional, but it probably makes sense not to waste any time and to train it right away.

We still need to ensure that the master pipeline creates a new training pipeline whenever a new dataset is uploaded to our datastore.

We do this by attaching a Schedule to our pipeline, telling it to look for changes in the default blob store of our datastore.

```
datastore = ws.get_default_datastore()

schedule = Schedule.create(workspace=ws, name=pipeline_name + "_sch",
                           pipeline_id=published_pipeline.id, 
                           experiment_name='Schedule_Run',
                           datastore=datastore,
                           wait_for_provisioning=True,
                           description="Datastore scheduler for Pipeline" + pipeline_name
                           )
```

## Literature
See this blog post: [breaking-the-wall-between-data-scientists-and-app-developers-with-azure-devops](https://azure.microsoft.com/en-us/blog/breaking-the-wall-between-data-scientists-and-app-developers-with-azure-devops/)