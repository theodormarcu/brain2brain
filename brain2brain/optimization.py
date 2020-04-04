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
import os
import time
import string
import json
from pathlib import Path
# General
import numpy as np
import pickle
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, GRU, TimeDistributed, Lambda, Activation
# import wandb
sys.path.append('../')
# Hyperas
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
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
    assert params is not None
    space = {
        "batch_size" : 512,
        "nb_filters" : hp.quniform("nb_filters", 8, 128, 8),
        "kernel_size" : hp.quniform("kernel_size", 3, 6, 1),
        "dilation_stop" : hp.choice("dilation_stop", [4, 5, 6, 7, 8]),
        "nb_stacks" : hp.quniform("nb_stacks", 2, 3, 1),
        "dropout_rate" : hp.uniform("dropout_rate", 0.0, 0.5),
        "kernel_initializer" : params["kernel_initializer"],
        "use_batch_norm" : hp.choice("use_batch_norm", [True, False]),
        "use_layer_norm" : hp.choice("use_layer_norm", [True, False]),
        "optimizer" : hp.choice("optimizer", ["rmsprop", "adam"]),
        "lr" : hp.loguniform('lr', 0, 0.01)
    }
    params.update(space)
    return params

# Objective function that hyperopt will optimize.
def tcn_m2m_objective_noseq(params):
    """
    Create an objective function based on the supplied params.
    """
    # Read saved paths for training.
    saved_paths = utils.get_file_paths(params["path"])
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(params["electrode_selection"])

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
    print(f"train_generator shape: {len(train_generator)}")
    print(f"val_generator shape: {len(val_generator)}")
    print(f"train_generator batch shape: {train_generator[0][0].shape, train_generator[0][1].shape}")
    print(f"val_generator batch shape: {val_generator[0][0].shape, val_generator[0][1].shape}")
    train_steps = len(train_generator)
    val_steps = len(val_generator)

    # TCN Model
    input_layer = Input(shape=(params["lookback_window"], electrode_count))
    dilations = tuple(2 ** np.arange(params["dilation_stop"]))
    new_dilations = []
    # Convert from np.int64 to int
    for d in dilations:
        new_dilations.append(d.item())
    x = TCN(nb_filters=int(params["nb_filters"]),
            kernel_size=(int(params["kernel_size"])),
            dilations=new_dilations,
            nb_stacks=int(params["nb_stacks"]),
            padding=params["padding"],
            use_skip_connections=params["use_skip_connections"],
            return_sequences=params["return_sequences"],
            activation=params["activation"],
            dropout_rate=params["dropout_rate"],
            kernel_initializer=params["kernel_initializer"],
            use_batch_norm=params["use_batch_norm"],
            use_layer_norm=params["use_layer_norm"])(input_layer)
    
    # Final regression layer.
    x = Dense(electrode_count)(x)
    output_layer = x
    model = Model(input_layer, output_layer)
    model.compile(optimizer=get_opt(params["optimizer"], params["lr"]),
                  loss=params["loss"])
    model.summary()
    # Callbacks for early stopping.
    # I don't think you can reliably add other types
    # of callbacks (e.g. checkpoints), since you
    # might need to save too much info. You
    # can save models after hyperparameter optimization.
    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min"
        ),
        TerminateOnNaN()
    ]

    result = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=params["epochs"],
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
                                  validation_steps=val_steps)
    try:
        min_loss = np.min(result.history['val_loss'])
        return {'loss': min_loss, 'status': STATUS_OK, 'model': model, 'history':result}
    except:
        print("Exception Occurred. Attempting to Recover Gracefully.")
    return {'status': STATUS_FAIL, 'model': model, 'history':result}
        

def tcn_optimization_noseq(params):
    # Ensure target directory exists.
    try:
        Path(params["target_folder"]).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {params['target_folder']}")

    # Create space with parameters for experiment.
    space = create_space(params)

    # Create trials object.
    trial_dump_file = os.path.join(params['target_folder'], "trial_dump.p")
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(trial_dump_file, "rb"))
        print("Found saved Trials! Loading...")
    except:  # create a new trials object and start searching
        trials = Trials()
    print(type(trials))
    
    # Minimize function.
    best = fmin(tcn_m2m_objective_noseq, space, algo=tpe.suggest, trials=trials, max_evals=params["max_evals"])
    # save the trials object
    with open(trial_dump_file, "wb") as f:
        pickle.dump(trials, f)
    
    print (best)
    print (trials.best_trial)
    print("End of run!")