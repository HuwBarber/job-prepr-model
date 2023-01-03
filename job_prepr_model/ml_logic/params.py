import os
import numpy as np

LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
MLFLOW_TRACKING_URI=os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT=os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME=os.environ.get("MLFLOW_MODEL_NAME")

model_params = {
    'maxpooling2d' : 4,
    'activation_for_hidden' : "tanh",
    'kernel_size': (2,2),
    'kernel_size_detail' : (1,1),
    'last_dense_layer_neurons1' : 100,
    'last_dense_layer_neurons2' : 30,
}

train_params = {
    'patience' : 5 ,
    'validation_split' : 0.2,
    'epochs' : 90,
    'batch_size' : 32
}

y_label_dict = {
    'angry' : 0,
    'disgust' : 1,
    'fear' : 2,
    'happy' : 3,
    'neutral' : 4,
    'sad' : 5,
    'surprise' : 6
}
