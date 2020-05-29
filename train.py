# Build and train PredNet on UCSD sequences.

# This script can be run in isolation, but is really meant to be part of
# an aml pipeline.

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    Callback,
)
from keras.optimizers import Adam

from models.prednet.prednet import PredNet
from models.prednet.data_utils import SequenceGenerator

import urllib.request

from azureml.core import Run
from azureml.core import VERSION

print("defining str2bool")


def str2bool(v):
    """
    convert string representation of a boolean into a boolean representation
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


print("defining args")
parser = argparse.ArgumentParser(description="Process input arguments")
parser.add_argument(
    "--preprocessed_data",
    default="./data/preprocessed/",
    type=str,
    dest="preprocessed_data",
    help="data folder mounting point",
)
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    help="learning rate",
    type=float,
    required=False,
)
parser.add_argument(
    "--lr_decay",
    default=1e-9,
    help="learning rate decay",
    type=float,
    required=False,
)
parser.add_argument(
    "--stack_sizes",
    dest="stack_sizes_arg",
    default="48, 96, 192",
    help="Stack sizes of hidden layers",
    type=str,
    required=False,
)
parser.add_argument(
    "--filter_sizes",
    dest="filter_sizes",
    default="3, 3, 3",
    help="Filter sizes for the three convolutional layers (A, A_hat, R)",
    type=str,
    required=False,
)
parser.add_argument(
    "--layer_loss_weights",
    dest="layer_loss_weights_arg",
    default="1.0, 0.",
    help="Layer loss weights",
    type=str,
    required=False,
)
parser.add_argument(
    "--remote_execution",
    dest="remote_execution",
    action="store_true",
    help="remote execution (AML compute)",
    required=False,
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    default=4,
    help="Batch size",
    type=int,
    required=False,
)
parser.add_argument(
    "--freeze_layers",
    dest="freeze_layers",
    default="0, 1, 2, 3",
    help="space separated list of layers to freeze",
    type=str,
    required=False,
)
parser.add_argument(
    "--fine_tuning",
    dest="fine_tuning",
    default="False",
    help="use the benchmark model and perform fine tuning",
    type=str,
    required=False,
)
parser.add_argument(
    "--dataset",
    dest="dataset",
    default="UCSDped1",
    help="the dataset that we are using",
    type=str,
    required=False,
)
parser.add_argument(
    "--output_mode",
    dest="output_mode",
    default="error",
    help="which output_mode to use (prediction, error)",
    type=str,
    required=False,
)
parser.add_argument(
    "--samples_per_epoch",
    dest="samples_per_epoch",
    default=10,
    help="n samples per epoch",
    type=int,
    required=False,
)
parser.add_argument(
    "--nb_epoch",
    dest="nb_epoch",
    default=150,
    help="max number of epochs",
    type=int,
    required=False,
)

print("parsing args")
args = parser.parse_args()
nb_epoch = args.nb_epoch
samples_per_epoch = args.samples_per_epoch
learning_rate = args.learning_rate
lr_decay = args.lr_decay
stack_sizes_arg = tuple(map(int, args.stack_sizes_arg.split(",")))
layer_loss_weights_arg = tuple(
    map(float, args.layer_loss_weights_arg.split(","))
)
filter_sizes = tuple(map(int, args.filter_sizes.split(",")))
remote_execution = args.remote_execution
preprocessed_data = os.path.join(
    args.preprocessed_data,
    args.dataset)
batch_size = args.batch_size
fine_tuning = str2bool(args.fine_tuning)
if len(args.freeze_layers) > 0 and fine_tuning:
    freeze_layers = tuple(map(int, args.freeze_layers.split(",")))
else:
    freeze_layers = []
dataset = args.dataset
output_mode = args.output_mode

print("training dataset is stored here:", preprocessed_data)

# normally would expect data to be passed with a PipelineData object from
# the previous pipeline step. This allows us to instead download the data
if "coursematerial" in preprocessed_data:
    preprocessed_data = os.path.join(os.getcwd(), "data")
    os.makedirs("data", exist_ok=True)

    urllib.request.urlretrieve(
        os.path.join(args.preprocessed_data, "X_train.hkl"),
        filename=os.path.join(preprocessed_data, "X_train.hkl"),
    )
    urllib.request.urlretrieve(
        os.path.join(args.preprocessed_data, "X_val.hkl"),
        filename=os.path.join(preprocessed_data, "X_val.hkl"),
    )
    urllib.request.urlretrieve(
        os.path.join(args.preprocessed_data, "sources_train.hkl"),
        filename=os.path.join(preprocessed_data, "sources_train.hkl"),
    )
    urllib.request.urlretrieve(
        os.path.join(args.preprocessed_data, "sources_val.hkl"),
        filename=os.path.join(preprocessed_data, "sources_val.hkl"),
    )

# create a ./outputs folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into
# run history
model_dir = 'outputs/model'
os.makedirs(model_dir, exist_ok=True)

# initiate logging if we are running remotely
if remote_execution:
    print("Running on remote compute target:", remote_execution)

    print("azureml.core.VERSION", VERSION)

    # start an Azure ML run
    run = Run.get_context()

    run.log("learning_rate", learning_rate)
    run.log("lr_decay", lr_decay)
    run.log("stack_sizes", args.stack_sizes_arg)
    run.log("layer_loss_weights_arg", args.layer_loss_weights_arg)
    run.log("filter_sizes", args.filter_sizes)
    run.log("preprocessed_data", args.preprocessed_data)
    run.log("dataset", args.dataset)
    run.log("batch_size", batch_size)
    run.log("freeze_layers", args.freeze_layers)
    run.log("fine_tuning", args.fine_tuning)

# model parameters
A_filt_size, Ahat_filt_size, R_filt_size = filter_sizes

# format of video frames (size of input layer)
n_channels, im_height, im_width = (3, 152, 232)

# settings for sampling the video data
nt = 10  # number of timesteps used for sequences in training
N_seq_val = 15  # number of sequences to use for validation

# settings for training and optimization
optimizer_type = "adam"
min_delta = 1e-4
patience = 10
save_model = True  # if weights will be saved
return_sequences = True

# define location for saving model and weights
weights_file = os.path.join(model_dir, "weights.hdf5")
json_file = os.path.join(model_dir, "model.json")

# Load data and source files
X_train_file = os.path.join(preprocessed_data, "X_train.hkl")
train_sources = os.path.join(preprocessed_data, "sources_train.hkl")
X_val_file = os.path.join(preprocessed_data, "X_val.hkl")
y_val_file = os.path.join(preprocessed_data, "y_val.hkl")
val_sources = os.path.join(preprocessed_data, "sources_val.hkl")
X_test_file = os.path.join(preprocessed_data, "X_test.hkl")
y_test_file = os.path.join(preprocessed_data, "y_test.hkl")
test_sources = os.path.join(preprocessed_data, "sources_test.hkl")

if fine_tuning:
    print("Performing transfer learning.")
    from azureml.core import Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.core.model import Model
    from azureml.exceptions._azureml_exception import ModelNotFoundException

    run = Run.get_context()
    ws = run.experiment.workspace

    keyvault = ws.get_default_keyvault()
    tenant_id = keyvault.get_secret('tenantId')
    scv_pr_id = keyvault.get_secret("servicePrincipalId")
    svc_pr_passwd = keyvault.get_secret("servicePrincipalPassword")

    svc_pr = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=scv_pr_id,
        service_principal_password=svc_pr_passwd,
    )

    ws = Workspace(ws.subscription_id, ws.resource_group, ws.name, auth=svc_pr)

    try:
        model_root = Model.get_model_path("prednet_" + dataset, _workspace=ws)
        print("model_root:", model_root)
    except ModelNotFoundException:
        print(
            "Didn't find model for this data set (%s). \
            Looking for model for UCSDped1 (where it all started), \
                because transfer learning was requested."
        )
        try:
            model_root = Model.get_model_path(
                "prednet_UCSDped1", _workspace=ws
            )
            print("model_root:", model_root)
        except ModelNotFoundException as e:
            print(
                "Warning! No model found in registry, cannot perform transfer"
                "learning!"
            )
            print(e)
            fine_tuning = False

# if fine_tuning:
#     # load model from json file
#     # todo, this is going to the real one
#     json_file = open(os.path.join(model_root, "model.json"), "r")
#     model_json = json_file.read()
#     json_file.close()

#     # initialize model
#     trained_model = model_from_json(
#         model_json, custom_objects={"PredNet": PredNet}
#     )
#     trained_model.output_mode = output_mode
#     trained_model.return_sequences = return_sequences

#     # load weights into new model
#     trained_model.load_weights(os.path.join(model_root, "weights.hdf5"))

#     # retrieve configuration of prednet.
#     # all prednet layers are weirdly stored in the second layer of the
#     # overall trained_model
#     layer_config = trained_model.layers[1].get_config()

#     # set output_mode to error, just to be safe
#     layer_config["output_mode"] = output_mode

#     # create instance of prednet model with the above weights and our
#     # layer_config
#     prednet = PredNet(
#         weights=trained_model.layers[1].get_weights(), **layer_config
#     )
#     # determine shape of input and define tensor for input
#     input_shape = list(trained_model.layers[0].batch_input_shape)[1:]
#     inputs = Input(shape=tuple(input_shape))

#     # get number of layers and expected length of video sequences
#     nb_layers = len(trained_model.layers) - 1
#     nt = input_shape[0]
# else:
# Set model characteristics according to our settings
# 4 layer architecture, with 3 input channels (rgb), and 48, 96, 192 units
# in the deep layers
stack_sizes = (n_channels,) + stack_sizes_arg
nb_layers = len(stack_sizes)  # number of layers
input_shape = (
    (n_channels, im_height, im_width)
    if keras.backend.image_data_format() == "channels_first"
    else (im_height, im_width, n_channels)
)
# number of channels in the representation modules
R_stack_sizes = stack_sizes
# length is 1 - len(stack_sizes), here targets for layers 2-4 are
# computered by 3x3 convolutions of errors from layer below
A_filt_sizes = (A_filt_size,) * (nb_layers - 1)
# convolutions of the representation layer for computing the predictions
# in each layer
Ahat_filt_sizes = (Ahat_filt_size,) * nb_layers
# filter sizes for the representation modules
R_filt_sizes = (R_filt_size,) * nb_layers
# build the model (see prednet.py)
prednet = PredNet(
    stack_sizes,
    R_stack_sizes,
    A_filt_sizes,
    Ahat_filt_sizes,
    R_filt_sizes,
    output_mode=output_mode,
    return_sequences=return_sequences,
)

# define tf tensor for inputs
inputs = Input(shape=(nt,) + input_shape)

# Set up how much the error in each video frame in a sequence is taken
# into account
# equally weight all timesteps except the first (done in next line)
time_loss_weights = 1.0 / (nt - 1) * np.ones((nt, 1))
# we obviously don't blame the model for not being able to predict the
# first video frame in a sequence
time_loss_weights[0] = 0


# Set up how much the error in each layer is taken into account during
# training
# weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0],
# "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.array(
    [layer_loss_weights_arg[0]]
    + [layer_loss_weights_arg[1]] * (nb_layers - 1)
)
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)

# errors will be (batch_size, nt, nb_layers)
errors = prednet(inputs)

errors_by_time = TimeDistributed(
    Dense(1, trainable=False),
    weights=[layer_loss_weights, np.zeros(1)],
    trainable=False,
)(
    errors
)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)

final_errors = Dense(
    1, weights=[time_loss_weights, np.zeros(1)], trainable=False
)(
    errors_by_time
)  # weight errors by time
loss_type = "mean_absolute_error"

# define optimizer
optimizer = Adam(lr=learning_rate, decay=lr_decay)

# put it all together
model = keras.models.Model(inputs=inputs, outputs=final_errors)
model.compile(loss=loss_type, optimizer=optimizer)
if fine_tuning:
    model.load_weights(
        os.path.join(model_root, model_dir, "weights.hdf5"),
        by_name=True,
        skip_mismatch=True)

# add early stopping
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        verbose=0,
        mode="min",
    )
]

if save_model:
    callbacks.append(
        ModelCheckpoint(
            filepath=weights_file, monitor="val_loss", save_best_only=True
        )
    )

# log training results to a file
callbacks.append(
    CSVLogger(
        filename=os.path.join("train.log"),
        separator=",",
        append=False,
    )
)

# log progress to AML workspace
if remote_execution:

    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log("val_loss", log["val_loss"])
            run.log("loss", log["loss"])

    callbacks.append(LogRunMetrics())

# if we desire to freeze any layers, this is where we do this
if len(freeze_layers) > 0:
    print("freezing layers:"),
    print(freeze_layers)

    for freeze_layer in freeze_layers:
        for c in [
            "i",
            "f",
            "c",
            "o",
            "a",
            "ahat",
        ]:  # iterate over layers at each depth
            try:
                model.layers[1].conv_layers[c][freeze_layer].trainable = False
            except IndexError:
                if c == "a":  # deepest layer doesn't have an A layer
                    pass
                else:
                    print(
                        "You seemed to be interested in freezing a layer that "
                        "is deeper than the deepest layer of the model.  "
                        "What's that supposed to do for you?"
                    )
                    raise

# object for generating video sequences for training and validation
train_generator = SequenceGenerator(
    X_train_file, train_sources, nt, batch_size=batch_size, shuffle=True
)
val_generator = SequenceGenerator(
    X_val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val
)


# train the model
history = model.fit(
    x=train_generator,
    steps_per_epoch=samples_per_epoch / batch_size,
    epochs=nb_epoch,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=N_seq_val / batch_size,
)

if remote_execution:
    # after training is complete, we log the final loss on the validation set
    run.log("final_val_loss", history.history["val_loss"][-1])

# create a figure of how loss changed of there course of training for
# validation and training set
plt.figure(figsize=(6, 3))
plt.title("({} epochs)".format(nb_epoch), fontsize=14)
plt.plot(
    history.history["val_loss"], "b-", label="Validation Loss", lw=4, alpha=0.5
)
plt.plot(history.history["loss"], "g-", label="Train Loss", lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image("Validation Loss", plot=plt)
else:
    plt.savefig("val_log.png")
plt.close()

# serialize NN architecture to JSON
model_json = model.to_json()

# save model JSON
with open(os.path.join(model_dir, "model.json"), "w") as f:
    f.write(model_json)
