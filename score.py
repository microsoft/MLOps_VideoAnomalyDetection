import json
import numpy as np
import os
from keras.models import model_from_json
from prednet import PredNet 
from keras.layers import Input
from azureml.core.model import Model
from keras.models import Model as Model_keras


def init():

    global model

    nt = 10

    model_root = Model.get_model_path('prednet_UCSDped1') #, _workspace=ws)
    # model_root = model_root.strip('model.json')
    print(model_root)
    # load json and create model
    json_file = open(os.path.join(model_root, 'model.json'), 'r') # todo, this is going to the real one
    # json_file = open(os.path.join(model_root, 'model', 'model.json'), 'r')
    model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(model_json, custom_objects={"PredNet": PredNet})
    # load weights into new model
    trained_model.load_weights(os.path.join(model_root, "weights.hdf5"))   
    
    # Create testing model (to output predictions)
    layer_config = trained_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    # data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=trained_model.layers[1].get_weights(), **layer_config)
    input_shape = list(trained_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    model = Model_keras(inputs=inputs, outputs=predictions)
    
def run(raw_data):
    # convert json data to numpy array
    data = np.array(json.loads(raw_data)['data'])
    data = data[np.newaxis, :]
    
    # make predictions
    X_hat = model.predict(data)
    
    # calculate error
    model_err = data - X_hat
    model_err[:, 0, :, :, :] = 0 # first frame doesn't count
    model_mse = np.mean( (model_err)**2, axis=(2,3,4) )  # look at all timesteps except the first
    
    return model_mse.tolist()
