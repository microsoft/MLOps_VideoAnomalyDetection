import json
import numpy as np
import os
from keras.models import model_from_json
from prednet import PredNet
from keras.layers import Input
from azureml.core.model import Model
from keras.models import Model as Model_keras
import joblib


def init():

    global prednet_model, logistic_regression_model

    nt = 10

    # logistic regression model for detecing anomalies based on model output
    try:
        model_path = Model.get_model_path("logistic_regression")
        logistic_regression_model = joblib.load(model_path)
    except Exception as e:
        print(e)
        logistic_regression_model = None

    prednet_path = Model.get_model_path('prednet_UCSDped1')
    # model_root = model_root.strip('model.json')
    print(prednet_path)
    # load json and create model
    with open(os.path.join(prednet_path, 'model.json'), 'r') as json_file:
        model_json = json_file.read()

    trained_model = model_from_json(
        model_json,
        custom_objects={"PredNet": PredNet})

    # load weights into new model
    trained_model.load_weights(os.path.join(prednet_path, "weights.hdf5"))   

    # Create testing model (to output predictions)
    layer_config = trained_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'

    test_prednet = PredNet(
        weights=trained_model.layers[1].get_weights(),
        **layer_config)
    input_shape = list(trained_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    prednet_model = Model_keras(inputs=inputs, outputs=predictions)


def run(raw_data):
    # convert json data to numpy array
    data = np.array(json.loads(raw_data)['data'])
    data = data[np.newaxis, :]

    # make predictions
    X_hat = prednet_model.predict(data)

    # calculate error
    model_err = data - X_hat

    # first frame doesn't count
    model_err[:, 0, :, :, :] = 0

    # look at all timesteps except the first
    model_std = np.std((model_err)**2, axis=(2, 3, 4))

    model_std = np.reshape(model_std, (np.prod(model_std.shape), 1))

    if logistic_regression_model:
        is_anom = logistic_regression_model.predict(model_std).tolist()
    else:
        is_anom = False

    return (model_std, is_anom)
