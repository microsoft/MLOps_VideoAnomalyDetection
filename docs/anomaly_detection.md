# Anomaly Detection

We have prepared our data (`data_preparation.py`) and trained our model (`train.py`).  Next, we see whether we can used it for detecting anomalies in video sequences.  

This is done in two steps:
1. Apply the trained model to the test dataset, i.e. to videos that contain anomalies.
2. Post-process the results.
3. Use a random forest classifier to evaluate different measures for defining anomalies. 


The UCSD dataset is a bit unusual in the sense that it actually contains information about which video frames contain anomalies.  We can use this information to validate our approach. That is, we can use a supervised approach to inform our decisions when developing our desired unsupervised solution.  The below will describe this in more detail.

## Apply the trained model to the test dataset

Here, we use the script `batch_scoring.py`. The script begins by loading the data.


### Configuration

There are some minimal settings we have to perform here. Most things don't change from how we configured the model and data for training, so we don't have to go over those again.

Another thing to we need to do is to configure what kind of output we want to model to produce.  Here, we are setting it to output its prediction for the next frame. That is, the activation in the A_hat layer:

```
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
```

Depending on your use-case, you could also log the activation of one of the deeper layers.

### Run

Next, we load the model architecture (`model.json`) and weights (`weights.hdf5`) to apply it to the test dataset (`X_test.hkl` and `sources_test.hkl`).

For each video frame, we log the prediction of the model for each frame pixel by pixel. Building up an array that is exactly of the same format as preprocessed video data (n_images, height, width, depth).  This way we can easily perform stats on the difference between the actual video frames (ground truth) and the model's predictions.

### Post-processing

We can do some immediate post-processing, to create some metrics that could be useful for detecting anomalies.  For each frame, mean squared error, and also various percentiles and the standard deviation of the model's prediction error.

The last step is to save these metrics to a pickled pandas dataframe. 

## Annotate results with labels from test dataset

Next, we load the labels (`y_test`) for the test dataset to annotate our results.

Our results dataframe (from running `batch_scoring.py`) contains one row per dataframe and various metrics for the frame, giving us information about how well the model predicted each video frame (e.g. mean squared error).  We want to add a column that tells us for each frame whether this frame contains an anomaly.

## Explore relationship between model metrics and anomalies

Now we have a reduced our problem to a very simple situation: We can used the metrics of how well our model was able to predict the next video frame to predict whether the video frame contained an anomaly. 

In other words, we have a binary classification problem, where the performance metrics are features, and the anomalies are the target label.

What we can do now is to fit e.g. a random forest classifier to these data, and then investigate which of the features are important for classification.  This is done by the script `fit_anoms.py`.


