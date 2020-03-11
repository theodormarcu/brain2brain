#!/usr/bin/env python3
####################################################################################################
'''
This module contains Hyperparameter optimization code and relies on hyperopt.

Using work described in:
https://github.com/Paperspace/hyperopt-keras-sample
https://hyperopt.github.io/hyperopt/getting-started/search_spaces/


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
####################################################################################################
# Imports
import sys
import time
import string
import json
from pathlib import Path
# General
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
# brain2brain
from brain2brain import utils
from brain2brain import generators
# TCN
from brain2brain.tcn import TCN
from brain2brain.tcn import compiled_tcn
# TF
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, GRU, TimeDistributed, Lambda, Activation
# import wandb
sys.path.append('../')
# Hyperas
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials


def get_opt(opt_str, lr):
    """
    Get optimizer for opt string and learning rate.
    """
    if opt_str == "adam":
        return optimizers.Adam(lr=lr, clipnorm=1.0)
    elif opt_str == "rmsprop":
        return optimizers.RMSprop(lr=lr, clipnorm=1.0)
    else:
        raise Exception("Only Adam and RMSProp are available here.")

# Hyperparameter Optimization Space
def create_space(params):
    """
    Create a space object for hyperopt using
    parameters passed through main.
    """
    space = {
        "batch_size" : hp.quniform('batch_size', 256, 2048, 256),
        "nb_filters" : hp.quniform("nb_filters", 8, 256, 8),
        "kernel_size" : hp.quniform("kernel_size", 3, 9, 1),
        "dilation_stop" : hp.quniform("dilation_stop", 4, 128, 4),
        "nb_stacks" : hp.quniform("nb_stacks", 2, 6, 1),
        "dropout_rate" : hp.uniform("dropout_rate", 0.0, 0.5),
        "kernel_initializer" : params["kernel_initializer"],
        "use_batch_norm" : hp.choice("use_batch_norm", [True, False]),
        "use_layer_norm" : hp.choice("use_layer_norm", [True, False]),
        "optimizer" : hp.choice("optimizer", ["rmsprop", "adam"]),
        "lr" : hp.loguniform('lr', -0.01, 0.01)
    }
    return space.update(params)

# Objective function that hyperopt will optimize.
def tcn_m2m_objective(params):
    """
    Create an objective function based on the supplied params.
    """
    # Read saved paths for training.
    saved_paths = utils.get_file_paths(params["path"])
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(params["electrode_selecton"])

    # Training Generator
    train_generator = generators.FGenerator(file_paths=train,
                                            lookback=params["lookback_window"],
                                            length=params["length_pred"],
                                            delay=params["delay_pred"],
                                            batch_size=params["batch_size"],
                                            sample_period=params["sample_period"],
                                            electrodes=params["electrode_selection"],
                                            shuffle=params["shuffle"],
                                            debug=params["debug_mode"],
                                            ratio=params["data_ratio"])
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val,
                                          lookback=params["lookback_window"],
                                          length=params["length_pred"],
                                          delay=params["delay_pred"],
                                          batch_size=params["batch_size"],
                                          sample_period=params["sample_period"],
                                          electrodes=params["electrode_selection"],
                                          shuffle=params["shuffle"],
                                          debug=params["debug_mode"],
                                          ratio=params["data_ratio"])

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    # Callbacks for early stopping.
    # I don't think you can reliably add other types
    # of callbacks (e.g. checkpoints), since you
    # might need to save too much info. You
    # can save models after hyperparameter optimization.
    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]

    # TCN Model
    input_layer = Input(shape=(params["lookback_window"], params["electrode_count"]))
    x = TCN(nb_filters=params["nb_filters"],
            kernel_size=params["kernel_size"],
            dilations=np.arange(start=1, stop=params["dilation_stop"]),
            nb_stacks=params["nb_stacks"],
            padding=params["padding"],
            use_skip_connections=params["use_skip_connections"],
            return_sequences=params["return_sequences"],
            activation=params["activation"],
            dropout_rate=params["dropout_rate"],
            kernel_initializer=params["kernel_initializer"],
            use_batch_norm=params["use_batch_norm"],
            use_layer_norm=params["use_layer_norm"])(input_layer)
    
    # Final regression layer.
    x = Dense(params["electrode_count"])(x)

    output_layer = x
    model = Model(input_layer, output_layer)
    model.compile(optimizer=get_opt(params["optimizer"], params["lr"]),
                  loss=params["loss"])

    result = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=params["epochs"],
                                  validation_data=val_generator,
                                  validation_steps=val_steps)

    min_loss = np.min(result.history['val_loss'])
    return {'loss': min_loss, 'status': STATUS_OK, 'model': model, 'history':result}

def tcn_optimization_noseq(params):
    # Create space with parameters for experiment.
    space = create_space(params)

    # Create trials object.
    trials = Trials()
    # Minimize function.
    best = fmin(tcn_m2m_objective, space, algo=tpe.suggest, trials=trials, max_evals=100)
    
    print (best)
    print (trials.best_trial)
    print("End of run!")