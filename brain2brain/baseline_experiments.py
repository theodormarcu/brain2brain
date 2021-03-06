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
from pathlib import Path
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
def baseline_experiment(experiment_dict: dict):
    """
    Baseline test.
    """
    target_folder = experiment_dict['target_folder']
    path = experiment_dict['path']
    batch_size = experiment_dict['batch_size']
    epochs = experiment_dict['epochs']
    lookback_window = experiment_dict['lookback_window']
    length_pred = experiment_dict['length_pred']
    delay_pred = experiment_dict['delay_pred']
    samples_per_second = experiment_dict['samples_per_second']
    electrode_selection=experiment_dict['electrode_selection']
    debug_mode = experiment_dict['debug_mode']
    hidden_units = experiment_dict['hidden_units']
    activation = experiment_dict['activation']
    opt = experiment_dict['opt']
    loss = experiment_dict['loss']

    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")

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

    # LSTM
    model = Sequential()
    model.add(Flatten(input_shape=(timesteps_per_sample, electrode_count)))
    model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(1)) 
    if opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=loss)
    else:
        raise Exception(f"Could not find optimizer {opt} Aborting.")

    # Save Summary
    summary = model.summary()
    model_summary_path = target_folder + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data = val_generator)
    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    # Save History to File (For Later)
    history_path = target_folder + "history.json"
    with open(history_path, 'w') as outfile:
        json.dump(history.history, outfile)
    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_plt = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs_plt, loss, 'bo', label="Training Loss")
    plt.plot(epochs_plt, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(target_folder + "train_val_loss_plot.png")
    plt.clf()

    predictions = model.predict_generator(val_generator, steps=val_steps,
                                callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=True, verbose=1)
    predictions_path = target_folder + "predictions.json"
    np.save(predictions_path, predictions)
    plt.figure()
    plt.plot(predictions)
    targets = []
    for i in range(len(val_generator)):
        x, y = val_generator[i]
        for target in y:
            targets.append(target[0][0])
    plt.plot(targets)
    plt.title('Actual vs predicted')
    plt.legend(['predicted', 'actual'])
    plt.savefig(target_folder + "plot.png")
    plt.clf()


####################################################################################################