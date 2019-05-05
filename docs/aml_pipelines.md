# Transfer Learning w/ AML Pipelines

So far we have looked into training one model, including hyperparameter tuning.  Let's figure out how to scale our solution such that whenever a new dataset is uploaded to blob store, a new anomaly detection is automatically created for this dataset.

To speed things up and to potentially improve the performance of our model, we will use [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning).

We create two AML pipelines to scale our solution. 

First, we set up a training pipeline that:​
- Splits a video into individual frames​.
- Generates a keras graph for the model​.
- Sweeps hyperparameters to find the best model​.

To scale our solution, we then define a second pipeline that:
- Adds a Datastore monitor that triggers the creation of a new training pipeline with the above steps.
- Ensures the new model is better than the currently deployed one​.
- Registers the best model for deployment​ as webservice.

## AML pipeline for training the model

The AML pipeline for training a model is defined in `pipelines_create.py`.

It contains the following steps:
1. video_decoding - extract individual frames of the video and store them in separate files (e.g. tiff)
1. data_prep - scale and crop images so that their size matches the size of the input layer of our model.
1. train_w_hyperdrive - train the model using hyperdrive
1. register_model - register the model in the AML workspace for later deployment

### Transfer learning

Take a look at the definition of the hyperdrive step.

```
ps = RandomParameterSampling(
        {
            [...]
            '--freeze_layers': choice("0, 1, 2", "1, 2, 3", "0, 1", "1, 2", "2, 3", "0", "3"),
            '--transfer_learning': choice("True", "False")
        }
    )
```

What this does is tell hyperdrive to explore whether transfer_learning benefits training. It also explores which layers to freeze during transfer learning. 

If transfer_learning is performed, the `train.py` script looks for an existing model in the model registry, downloads it, and starts retraining it for the current dataset. 

You may be wondering whether training will really be faster, even if we also have training runs without transfer learning.  Those training runs could potentially take very long to converge.  Luckily hyperdrive comes with an early termination policy, so that runs that are taking too long and are performing worse than other runs are immediately canceled. 

```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1, delay_evaluation=20)
```

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