import json
import numpy as np
import pandas as pd
import os
from keras.models import model_from_json
from keras.layers import Input
from azureml.core.model import Model
from keras.models import Model as Model_keras
import joblib
from models.prednet.prednet import PredNet


def load_prednet_model(name):
    nt = 10
    prednet_path = Model.get_model_path(name)

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

    return prednet_model


def init():

    global prednet_models, clf_models

    with open('deployment_assets/models.json', 'r') as f:
        models = json.load(f)

    prednet_models = {}
    for name in models['prednet_model_names']:
        prednet_models[name] = load_prednet_model(name)

    clf_models = {}
    for name in models['clf_model_names']:
        model_path = Model.get_model_path(name)
        clf_models[name] = joblib.load(model_path)


def run(raw_data):
    # convert json data to numpy array
    json_data = json.loads(raw_data)
    camera_id = json_data['id']
    data = np.array(json_data['data'])
    data = data[np.newaxis, :]

    # make predictions
    X_hat = prednet_models["prednet_" + camera_id].predict(data)

    # calculate error
    model_err = data - X_hat

    # first frame doesn't count
    model_err[:, 0, :, :, :] = 0

    # look at all timesteps except the first
    model_std = np.std((model_err) ** 2, axis=(2, 3, 4))
    model_mse = np.mean((model_err) ** 2, axis=(2, 3, 4))
    model_p_50 = np.percentile((model_err) ** 2, 50, axis=(2, 3, 4))
    model_p_75 = np.percentile((model_err) ** 2, 75, axis=(2, 3, 4))
    model_p_90 = np.percentile((model_err) ** 2, 90, axis=(2, 3, 4))
    model_p_95 = np.percentile((model_err) ** 2, 95, axis=(2, 3, 4))
    model_p_99 = np.percentile((model_err) ** 2, 99, axis=(2, 3, 4))

    model_mse = np.reshape(model_mse, np.prod(model_mse.shape))
    model_p_50 = np.reshape(model_p_50, np.prod(model_mse.shape))
    model_p_75 = np.reshape(model_p_75, np.prod(model_mse.shape))
    model_p_90 = np.reshape(model_p_90, np.prod(model_mse.shape))
    model_p_95 = np.reshape(model_p_95, np.prod(model_mse.shape))
    model_p_99 = np.reshape(model_p_99, np.prod(model_mse.shape))
    model_std = np.reshape(model_std, np.prod(model_mse.shape))

    df = pd.DataFrame(
        {
            "model_mse": model_mse,
            "model_p_50": model_p_50,
            "model_p_75": model_p_75,
            "model_p_90": model_p_90,
            "model_p_95": model_p_95,
            "model_p_99": model_p_99,
            "model_std": model_std,
        }
    )

    is_anom = clf_models["clf_" + camera_id].predict(df).tolist()

    return (is_anom)
