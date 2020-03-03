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

####################################################################################################
def baseline_experiment(experiment_dict: dict):
    """
    Baseline test.
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
    hidden_units = experiment_dict['hidden_units']
    activation = experiment_dict['activation']
    opt = experiment_dict['opt']
    loss = experiment_dict['loss']

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
    model_summary_path = file_prefix + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model_architecture = model.to_json()
    model_architecture_file_path = file_prefix + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data = val_generator)
    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')
    # Save History to File (For Later)
    history_path = file_prefix + "history.json"
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
    plt.title('Actual vs predicted')
    plt.legend(['predicted', 'actual'])
    plt.savefig(file_prefix + "plot.png")
    plt.clf()


####################################################################################################

####################################################################################################

def tcn_experiment(experiment_dict: dict):
    """
    Testing TCN general suite.
    experiment_description: str,
    file_prefix: str,
    path: str,
    batch_size: int,
    epochs: int, 
    lookback_window: int,
    length_pred: int,
    delay_pred: int,
    samples_per_second: int,
    electrode_selection: list,
    debug_mode: bool = False,
    num_feat: int,
    num_classes: int,
    kernel_size: int,
    dilations: list,
    nb_stacks: int,
    output_len: int,
    padding: str,
    use_skip_connections: bool,
    return_sequences: bool,
    regression: bool,
    dropout_rate: float,
    name: str,
    kernel_initializer: str,
    activation: str,
    opt: str,
    lr: float
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
    num_feat = experiment_dict['num_feat']
    num_classes = experiment_dict['num_classes']
    nb_filters = experiment_dict['nb_filters']
    kernel_size = experiment_dict['kernel_size']
    dilations=experiment_dict['dilations']
    nb_stacks = experiment_dict['nb_stacks']
    output_len = experiment_dict['output_len']
    padding = experiment_dict['padding']
    use_skip_connections = experiment_dict['use_skip_connections']
    return_sequences = experiment_dict['return_sequences']
    regression = experiment_dict['regression']
    dropout_rate = experiment_dict['dropout_rate']
    name = experiment_dict['name']
    kernel_initializer = experiment_dict['kernel_initializer']
    activation = experiment_dict['activation']
    opt = experiment_dict['opt']
    lr = experiment_dict['lr']

    # wandb.init(project=file_prefix+"wandb")
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

    # # TCN
    # i = Input(shape=(timesteps_per_sample, electrode_count),
    #           batch_size=batch_size)
    # m = TCN()(i)
    # # No activation.
    # m = Dense(1)(m)
    # model = Model(inputs=[i], outputs=[m])
    model = compiled_tcn(num_feat=num_feat,
                         batch_size=batch_size,
                         num_classes=num_classes,
                         nb_filters=nb_filters,
                         kernel_size=kernel_size,
                         dilations=dilations,
                         nb_stacks=nb_stacks,
                         max_len=timesteps_per_sample,
                         output_len=output_len,
                         padding=padding,
                         use_skip_connections=use_skip_connections,
                         return_sequences=return_sequences,
                         regression=regression,
                         dropout_rate=dropout_rate,
                         name=name,
                         kernel_initializer=kernel_initializer,
                         activation=activation,
                         opt=opt,
                         lr=lr)
    # Save Summary
    summary = model.summary()
    model_summary_path = file_prefix + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print("calling compile")
    # model.compile('adam', 'mae')
    # Save Model Config and Architecture
    model_config = model.get_config()
    model_config_file_path = file_prefix + "model_config.json"
    with open(model_config_file_path, 'w') as outfile:
        json.dump(model_config, outfile)
    model_architecture = model.to_json()
    model_architecture_file_path = file_prefix + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(generator=train_generator,
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        validation_data=val_generator,
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
    # wandb.save(file_prefix + "wandb.h5")
####################################################################################################

# def main():
    # tcn_experiment2()

# if __name__ == '__main__':
    # main()
