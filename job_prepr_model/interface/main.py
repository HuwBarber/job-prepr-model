import numpy as np
import pandas as pd
from colorama import Fore, Style

from job_prepr_model.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from job_prepr_model.ml_logic.data import load_train_data, load_test_data
from job_prepr_model.ml_logic.encoders import label_encode
from job_prepr_model.ml_logic.registry import get_model_version
from job_prepr_model.ml_logic.registry import load_model, save_model

#Import model and training params
from job_prepr_model.ml_logic.params import model_params, train_params

import time

maxpooling2d=model_params['maxpooling2d']
kernel_size=model_params['kernel_size']
kernel_size_detail=model_params['kernel_size_detail']
last_dense_layer_neurons1=model_params['last_dense_layer_neurons1']
last_dense_layer_neurons2=model_params['last_dense_layer_neurons2']
activation_for_hidden=model_params['activation_for_hidden']
batch_size=train_params['batch_size']
patience=train_params['patience']
validation_split=train_params['']
epochs=train_params['epochs']

def train():

    y_cat = None
    y=None

    X, y = load_train_data()
    y_cat_len = label_encode(y)[0].shape[0]
    Xshape = (48, 48, 1)
    y_cat = label_encode(y)

    model = initialize_model(X,y_cat_len,Xshape,
                     maxpooling2d=maxpooling2d,
                     activation_for_hidden=activation_for_hidden,
                     kernel_size=kernel_size,
                     kernel_size_detail=kernel_size_detail,
                     last_dense_layer_neurons_1=last_dense_layer_neurons1,
                     last_dense_layer_neurons_2=last_dense_layer_neurons2,
                     )

    model = compile_model(model, learning_rate=None)

    model, history = train_model(model, X, y_cat,
                                batch_size=batch_size,
                                patience=patience,
                                validation_split=validation_split,
                                epochs=epochs,
                                )

    params = dict(
        batch_size=batch_size,
        patience=patience,
        context="train",
        validation_split = validation_split,
        model_version=get_model_version()
    )

    val_accuracy = np.min(history.history['val_accuracy'])
    save_model(model=model, params=params, metrics=dict(mae=val_accuracy))

    return history


def validate():

    # load new data
    X, y = load_test_data()

    y_cat = label_encode(y)

    model = load_model()

    metrics_dict = evaluate_model(model=model, X=X, y=y_cat)

    accuracy = metrics_dict[1]
    # save evaluation
    params = dict(
        batch_size=batch_size,
        patience=patience,
        context="train",
        validation_split = validation_split,
        model_version=get_model_version()
    )

    save_model(params=params, metrics=dict(accuracy=accuracy))

    return accuracy

if __name__ == '__main__':
    train()
    validate()
