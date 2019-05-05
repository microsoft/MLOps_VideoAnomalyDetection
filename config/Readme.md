# How to create a custom docker image for aml compute

## Login into the Azure Container Registry (ACR)

`az acr login -n <name_of_registry>`

## Create Dockerfile

You can find an example for a Dockerfile in this directory. 

You start by choosing the correct base image in the first line of this file.  In our example, we pick the base image for a gpu compute target. Using cuda 9.0, because we are trying to run a legacy version of tensorflow-gpu (v.1.8.0)

Then we install *all* the conda and pip packages that we will need, pinning the version numbers so that we can be sure about the outcome.

Next we build the image, giving it a meaningful name and a tag (here: 1), so that we can track iterations on getting the docker image right.
`docker build -f Dockerfile -t wopauli_1.8\-gpu:1 .`

This takes a couple of minutes, hopefully less than 5.

## Upload the docker image to ACR

First we tag it such that it also contains the address of the azure container registry.

`docker tag wopauli_1.8-gpu:1 <name_of_registry>.azurecr.io/wopauli_1.8-gpu:1`

Then we can push it to the azure container registry.

`docker push <name_of_registry>.azurecr.io/wopauli_1.8-gpu:1`

## Use the image

Let's look at how we can use our custom docker image for training a deep learning model.  The below code is taken from `pipelines_build.py`.

```
# configure access to ACR for pulling our custom docker image
acr = ContainerRegistry()
acr.address = config['acr_address']
acr.username = config['acr_username']
acr.password = config['acr_password']

# create an estimator 
est = Estimator(source_directory=script_folder,
                compute_target=gpu_compute_target,
                entry_script='train.py', 
                use_gpu=True,
                node_count=1,
                custom_docker_image = "wopauli_1.8-gpu:1",
                image_registry_details=acr,
                user_managed=True
                )
```

When we created the estimator, we make make sure to define the `custom_docker_image`, provide the `image_registry_details`, and make sure we set `user_managed=True`, so no additional packages are install at run-time that could lead to version conflicts.

