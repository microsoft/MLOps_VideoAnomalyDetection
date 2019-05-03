# Data Preparation

> file: `data_prep.py`

## Download the data

Download the data from the UCSD website and unpack it. 

Create a folder `data` in the `video_anomaly` directory, and copy the folders `UCSDped1` and `UCSDped2` into that folder (e.g. `<project_root>/video_anomaly/data/UCSDpred1`). 

## Data prep

The next step for us is to get our data into shape for training the model.

1. Split the data into sets for training, validation, and testing.
2. Load the individual images, then:
    - Resize the image to match the size of our model.
    - Insert them to a numpy array which can be used for training (dimensions: n_images, height, width, depth).
    - Create a second array that contains the folder in which each video frame was stored.
3. [Hickle](https://github.com/telegraphic/hickle) the created arrays to a binary [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file for faster I/O during training.


## Data Split

We use the `sklearn.model_selection import train_test_split` to split the *normal* videos randomly into videos for training the model and videos for validation. We continuously perform model validation during training, to see how well the model does with videos that haven't been used during training.

We also create a dataset for *testing*. Here, we are using the videos which contain anomalies, to see whether our approach allows us to detect anomalies.

## Resize the images

We resize the images so that they match the size of the input layer of our model.  You may ask, why don't we just change the size of the input layer to match the size of our images.  The answer is that it's complicated:

1. There are constraints on possible dimensions of the input layer.  We'll go into more detail on this topic in the step [model_development](./model_development.md)
2. You may need to change this depending on the compute hardware you are using (e.g. A Tesla K80 card will not have as much memory as a Tesla P100).

> This you could add here:
> - Crop images, to remove parts you are not interested in.
> - Blur images, this can sometimes help with convergence.
> - Rotate iamges, which can help with generalization to videos that were recorded in different angles.
> - Converting to gray-scale. If you have videos that were recorded in color (RGB), you could convert them to gray-scale. (We are actually converting gray-scale images to RGB format here. This doesn't make much sense here, but this allows us to keep the model architecture such that it will work with color videos as well.)

## Build numpy arrays to hold video data and file folders

We then insert the preprocessed video frames into numpy arrays, one array for each dataset spilt.  This array has the dimensions n_images * height * width * depth.

We create a second array that will contains for each video frame the folder that it was stored in.  We will use this information to determine which video sequence a video frame belongs to.rchive for fast loading.

## Save the processed video data

We [Hickle](https://github.com/telegraphic/hickle) the created arrays to a binary [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file.

Note, that this binary file as the potential to expose you to version skew. That is, you won't be able to load data into Python 3 if it was stored in Python 2.