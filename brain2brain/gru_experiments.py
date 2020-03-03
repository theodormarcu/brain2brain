#!/usr/bin/env python3
####################################################################################################
'''
This module contains brain2brain experiments.


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
####################################################################################################
# Imports
import sys
import time
import string
import json
# General
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
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
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, GRU
# import wandb
sys.path.append('../')

####################################################################################################
def gru_experiment(experiment_dict: dict):
    """
    GRU test.
    """
    file_prefix = experiment_dict['file_prefix']
    path = experiment_dict['path']
    batch_size = experiment_dict['batch_size']
    epochs = experiment_dict['epochs']
    lookback_window = experiment_dict['lookback_window']
    length_pred = experiment_dict['length_pred']
    delay_pred = experiment_dict['delay_pred']
    samples_per_second = experiment_dict['samples_per_second']
    electrode_selection=experiment_dict['electrode_selection']
    debug_mode = experiment_dict['debug_mode']
    dropout_rate = experiment_dict['dropout_rate']
    recurrent_dropout = experiment_dict['recurrent_dropout']

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

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    # GRU
    model = Sequential()
    model.add(GRU(32,
                  dropout=dropout_rate,
                  recurrent_dropout=recurrent_dropout,
                  input_shape=(timesteps_per_sample, electrode_count)))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')

    # Save Summary
    model.summary()
    model_summary_path = file_prefix + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model_architecture = model.to_json()
    model_architecture_file_path = file_prefix + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)


    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=20,
                                  validation_data = val_generator,
                                  validation_steps=val_steps)
    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')
    
    # Save History to File (For Later)
    history_path = file_prefix + "history.json"
    with open(history_path, 'w') as outfile:
        json.dump(history.history, outfile)

    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(file_prefix + "train_val_loss_plot.png")
    plt.clf()

    p = model.predict_generator(val_generator, steps=val_steps,
                                callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=True, verbose=1)
    plt.figure()
    plt.plot(p)
    targets = []
    for i in range(len(val_generator)):
        x, y = val_generator[i]
        for target in y:
            targets.append(target[0][0])
    plt.plot(targets)
    plt.title('Actual vs predicted.')
    plt.legend(['predicted', 'actual'])
    plt.savefig(file_prefix + "plot.png")
    plt.clf()


####################################################################################################