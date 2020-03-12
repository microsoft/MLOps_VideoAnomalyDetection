# Model Development

> file: `train.py`

Next, we setup our model and train it.

This involves the following steps:
1. Define the hyperparameters of your model:
   1. Size of the input layer
   2. How to sample the input data
   3. Model parameters (e.g. depth of the model)
   4. Training/Optimization Parameters
   5. Decide how to weight prediction errors for each video frame in a sequence `time_loss_weights` and across layers of the model `layer_loss_weights`.
2. Build the model using the above settings
3. Add callbacks that allow you to schedule changes and to monitor progress.
4. Train the model
   

## Hyperparameters

### Define the size of the input layer. 

This part is pretty straightforward. Look for the definition of `n_channels, im_height, im_width` and adjust it to the size you chose during preprocessing of your video data.

Be careful though, because there is one big constraint here. [Max-pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) with stride 2 is performed in each layer.  That means that the size of layers is divided by 2 each time. If you have a PredNet architecture of depth 4 (which is the default here), you have to make sure that the both dimensions of the image can be divided by 2 without a remainder for as many times as you have layers in your network.

### Sampling the input data

You might be able to leave these settings at their defaults, as they have been tested with various datasets. 
- `batch_size`, how many epochs to include in each batch
- `nb_epoch`, how many epochs to train the model for (also see `EarlyStopping` below)
- `samples_per_epoch`, how many video sequences to include in an epoch
- `nt`, how many video frames to include in each video sequence
- `N_seq_val`, how many video sequences to use for validation at the end of each epoch

### Model hyperparameters

Some of these are so important that they can be set via input arguments to the training script:
- `learning_rate`, yes, it's the learning rate
- `stack_sizes`, the dimensionality of the output space (i.e. the number output of filters in the convolution) for each convolutional layer.
- `*_filt_sizes`, the sizes of the convolution window for each convolutional layer (`A`, `A_hat`, `R`).
- `layer_loss_weights`, how to weight the prediction error in the first and hidden layers `E`.

## Build the model

We can't go into details on how to build a Keras model, but here is a quick [introduction](https://keras.io/#getting-started-30-seconds-to-keras).

Briefly, we have three parts here:
- `inputs`, a tensor holding the inputs to prednet
- `errors`, the result of applying prednet to the inputs.
- `final_errors`, errors weighted by layer `layer_loss_weight` and time `time_loss_weight`. 

We then compile the model and fit it to the video sequences created by a `SequenceGenerator`. 

## Callbacks

One of the nice features of Keras is [callbacks](https://keras.io/callbacks/):

> A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callbacks (as the keyword argument callbacks) to the .fit() method of the Sequential or Model classes. The relevant methods of the callbacks will then be called at each stage of the training. 

We are using three callbacks here:
1. `LearningRateScheduler`, to decrease the learning rate of the model after a certain number of epochs.
2. `ModelCheckpoint`, to save the model and trained weights training.
3. `EarlyStopping`, to stop training if we don't see any improvement on the validation set. This is useful because it allows us to save time, but also because it avoids [overfitting](https://en.wikipedia.org/wiki/Overfitting) the training set.  Here is a Wikipedia article on [EarlyStopping](https://en.wikipedia.org/wiki/Early_stopping).
4. `CSVLogger`, to save the progress of the model to a log file.

## Train the model

Training the model took about 5h with a Tesla K80 NVidia card (vm_size: `Standard_NC6`).  We don't know how long it would take to train the model without a GPU, because it tool too long for us to find out. See this page for your choices for [GPU optimized virtual machine sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu).  