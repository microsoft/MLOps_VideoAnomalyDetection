import json
import numpy as np
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

    print("cwd:", os.getcwd())
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
    model_std = np.std((model_err)**2, axis=(2, 3, 4))

    model_std = np.reshape(model_std, (np.prod(model_std.shape), 1)).tolist()

    is_anom = clf_models["clf" + camera_id].predict(model_std).tolist()

    return (model_std, is_anom)
