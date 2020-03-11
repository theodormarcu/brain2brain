#!/usr/bin/env python3
####################################################################################################
'''
This module contains Hyperparameter optimization code.

Using work described in:
https://github.com/Paperspace/hyperopt-keras-sample
https://hyperopt.github.io/hyperopt/getting-started/search_spaces/


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
####################################################################################################
import numpy as np
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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, GRU, TimeDistributed, Lambda
# import wandb
sys.path.append('../')
# Hyperas
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

# Hyperparameter Optimization
# from hyperopt import hp, tpe

# m2m_noseq_space = {
#     "batch_size" : hp.quniform('batch_size', 256, 2048, 256),
#     "nb_filters" : hp.quniform("nb_filters", 8, 256, 8),
#     "kernel_size" : hp.quniform("kernel_size", 3, 9, 1),
#     "dilations" : np.arange(start=1, stop=hp.quniform("dilations", 4, 128, 4)),
#     "nb_stacks" : hp.quniform("nb_stacks", 2, 6, 1),
#     "dropout_rate" : hp.uniform("dropout_rate", 0.0, 0.5),
#     "kernel_initializer" : hp.choice("kernel_initializer"),
#     "activation" : "linear",
#     "opt" : "RMSprop",
#     "lr" : hp.loguniform('lr', -0.01, 0.01)
# }

lookback_window = None
electrode_count = None
padding = None
activation = None
kernel_initializer = None
use_skip_connections = None
return_sequences = None

# Objective function that hyperopt will optimize
# def tcn_m2m_objective

def tcn_m2m_model(train_generator, validation_generator):
    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]

    assert lookback_window is not None
    assert electrode_count is not None
    assert padding is not None
    assert activation is not None
    assert kernel_initializer is not None
    assert use_skip_connections is not None
    assert return_sequences is not None

    
    # TCN
    input_layer = Input(shape=(lookback_window, electrode_count))

    x = TCN(nb_filters={{quniform( 8, 256, 8)}},
            kernel_size={{quniform(3, 9, 1)}},
            dilations=np.arange(start=1, stop={{quniform(4, 128, 4)}}),
            nb_stacks={{quniform(2, 6, 1)}},
            padding=padding,
            use_skip_connections=use_skip_connections,
            return_sequences=return_sequences,
            activation=activation,
            dropout_rate={{uniform(0.0, 0.5)}},
            kernel_initializer=kernel_initializer,
            use_batch_norm={{choice([True, False])}},
            use_layer_norm={{choice([True, False])}})(input_layer)

    # x = TimeDistributed(Dense(electrode_count))(x)
    x = Dense(5)(x)
    # x = Lambda(lambda x: x[:, :length_pred , :])(x)
    # x = Activation('linear')(x)
    # model.add(Dense(electrode_count))
    # model.add(TimeDistributed(Dense(electrode_count)))
    # model.add(Activation("linear"))
    # model.add(Lambda(lambda x: x[:, :length_pred , :]))
    output_layer = x
    model = Model(input_layer, output_layer)
    model.compile({{choice(['rmsprop', 'adam', 'sgd'])}}, loss='mean_absolute_error')

    result = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=20,
                                  validation_data=validation_generator,
                                  validation_steps=len(validation_generator))

    min_loss = np.min(result.history['val_loss'])
    return {'loss': min_loss, 'status': STATUS_OK, 'model': model, 'history':result}


def data(path, batch_size, epochs, lookback_window, length_pred, delay_pred, samples_per_second, electrode_selection, debug_mode):

    # Read saved paths for training.
    saved_paths = utils.get_file_paths(path)
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(electrode_selection)
    # Sampling of electrodes.
    timesteps_per_sample = int(lookback_window // samples_per_second)

    # Training Generator
    train_generator = generators.FGenerator(file_paths=train,
                                            lookback=lookback_window, length=length_pred, delay=delay_pred,
                                            batch_size=batch_size, sample_period=samples_per_second,
                                            electrodes=electrode_selection, shuffle=True, debug=debug_mode)
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val,
                                          lookback=lookback_window, length=length_pred, delay=delay_pred,
                                          batch_size=batch_size, sample_period=samples_per_second,
                                          electrodes=electrode_selection, shuffle=False, debug=debug_mode)
    
    return train_generator, val_generator

def tcn_optimization(experiment_dict):
    global lookback_window, electrode_count, use_skip_connections,\
           return_sequences

    path = experiment_dict['path']
    batch_size = experiment_dict['batch_size']
    epochs = experiment_dict['epochs']
    lookback_window = experiment_dict['lookback_window']
    length_pred = experiment_dict['length_pred']
    delay_pred = experiment_dict['delay_pred']
    samples_per_second = experiment_dict['samples_per_second']
    electrode_selection=experiment_dict['electrode_selection']
    padding=experiment_dict["padding"]
    activation=experiment_dict['activation']
    kernel_initializer=experiment_dict["kernel_initializer"]
    use_skip_connections=experiment_dict["use_skip_connections"]
    return_sequences=experiment_dict["return_sequences"]
    debug_mode = experiment_dict['debug_mode']

    electrode_count = len(electrode_selection)

    data_args = (path, batch_size, epochs, lookback_window, length_pred,
                 delay_pred, samples_per_second, 
                 electrode_selection, debug_mode)
    
    train_generator, validation_generator = data(data_args)

    best_run, best_model = optim.minimize(model=tcn_m2m_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=5,
                                                 trials=Trials(),
                                                 data_args=data_args)

    print("Evaluation of best performing model:")
    print(best_model.evaluate(validation_generator))

