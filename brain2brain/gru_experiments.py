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
import os
# General
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from pathlib import Path
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
from tensorflow.keras.layers import Dense, Flatten, GRU, Lambda, TimeDistributed
# Wandb
import wandb
from wandb.keras import WandbCallback
sys.path.append('../')

####################################################################################################
def gru_experiment(experiment_dict: dict):
    """
    GRU test.
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
    dropout_rate = experiment_dict['dropout_rate']
    recurrent_dropout = experiment_dict['recurrent_dropout']
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

    # GRU
    model = Sequential()
    model.add(GRU(hidden_units,
                  dropout=dropout_rate,
                  recurrent_dropout=recurrent_dropout,
                  input_shape=(timesteps_per_sample, electrode_count)))
    model.add(Dense(1))
    if opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=loss)
    else:
        raise Exception(f"Could not find optimizer {opt} Aborting.")
    # Save Summary
    model.summary()
    model_summary_path = target_folder + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)


    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=20,
                                  validation_data = val_generator,
                                  validation_steps=val_steps)
    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    
    # Save History to File (For Later)
    history_path = target_folder + "history.json"
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
    plt.title('Actual vs predicted.')
    plt.legend(['predicted', 'actual'])
    plt.savefig(target_folder + "plot.png")
    plt.clf()


####################################################################################################

####################################################################################################
def gru_m2m(experiment_dict: dict):
    """
    GRU test. Many to many.
    https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
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
    dropout_rate = experiment_dict['dropout_rate']
    recurrent_dropout = experiment_dict['recurrent_dropout']
    hidden_units = experiment_dict['hidden_units']
    dense_hidden_units = experiment_dict['dense_hidden_units']
    activation = experiment_dict['activation']
    opt = experiment_dict['opt']
    loss = experiment_dict['loss']
    return_sequences = experiment_dict['return_sequences']

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
    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")
    print(f"Electrode Count={electrode_count}")
    # GRU

    # GRU
    model = Sequential()
    model.add(GRU(units=hidden_units,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                input_shape=(timesteps_per_sample, electrode_count),
                return_sequences = return_sequences))
    model.add(TimeDistributed(Dense(electrode_count)))
    # model.add(Activation("linear"))
    model.add(Lambda(lambda x: x[:, :length_pred , :]))
    if opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=loss)
    else:
        raise Exception(f"Could not find optimizer {opt} Aborting.")
    # Save Summary
    model.summary()
    model_summary_path = target_folder + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)


    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=20,
                                  validation_data = val_generator,
                                  validation_steps=val_steps)
    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    
    # Save History to File (For Later)
    history_path = target_folder + "history.json"
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
    plt.title('Actual vs predicted.')
    plt.legend(['predicted', 'actual'])
    plt.savefig(target_folder + "plot.png")
    plt.clf()


####################################################################################################

####################################################################################################

def gru_m2o_noseq(experiment_dict: dict):
    """
    GRU test. Many to one.
    https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    ####################################################################################################
    experiment_name = experiment_dict['experiment_name']
    experiment_description = experiment_dict['experiment_description']
    target_folder = experiment_dict['target_folder']
    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # Save dictionary in json as well
    utils.save_json_file(experiment_dict, target_folder + "experiment_params.json")
    # Configure wandb
    # Toggle Offline Mode
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(name=experiment_name,
               notes=experiment_description,
               config=experiment_dict,
               dir=target_folder,
               entity="theodormarcu",
               project="brain2brain")
    # Save Hyperparams
    config = wandb.config
    # Read saved paths for training.
    saved_paths = utils.get_file_paths(config.path)
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(config.electrode_selection)
    # Sampling of electrodes.
    timesteps_per_sample = int(config.lookback_window // config.samples_per_second)
    # Training Generator
    train_generator = generators.FGenerator(file_paths=train,
                                            lookback=config.lookback_window,
                                            length=config.length_pred,
                                            delay=config.delay_pred,
                                            batch_size=config.batch_size,
                                            sample_period=config.samples_per_second,
                                            electrodes=config.electrode_selection,
                                            electrode_output_ix=config.electrode_out,
                                            shuffle=True,
                                            debug=config.debug_mode)
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val,
                                          lookback=config.lookback_window,
                                          length=config.length_pred,
                                          delay=config.delay_pred,
                                          batch_size=config.batch_size,
                                          sample_period=config.samples_per_second,
                                          electrodes=config.electrode_selection,
                                          electrode_output_ix=config.electrode_out,
                                          shuffle=False,
                                          debug=config.debug_mode)

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")

    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=target_folder+"model_checkpoint.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=True,
        )
    ]
    # GRU Many-to-one Model Architecture
    model = Sequential()
    model.add(GRU(units=config.hidden_units,
            dropout=config.dropout_rate,
            recurrent_dropout=config.recurrent_dropout,
            input_shape=(timesteps_per_sample, electrode_count),
            return_sequences = experiment_dict['return_sequences']))
    model.add(Dense(1))
    if config.opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=config.loss)
    else:
        raise Exception(f"Could not find optimizer {config.opt} Aborting.")
    # Save Summary
    model.summary()
    if config.opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=config.loss)
    else:
        raise Exception(f"Could not find optimizer {config.opt} Aborting.")
    print("calling compile")
    # Save Model Config and Architecture
    utils.save_json_file(model.get_config(), target_folder + "model_config.json")
    utils.save_json_file(model.to_json(), target_folder + "model_architecture.json")

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=config.epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
                                  validation_steps=val_steps)

    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    # Save History to File (For Later)
    utils.save_json_file(history.history, target_folder + "history.json")
    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
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
    plt.title('Actual vs predicted.' + experiment_name)
    plt.legend(['predicted', 'actual'])
    plt.savefig(target_folder + "predict_vs_targets.png")
    plt.clf()
    wandb.save("wandb.h5")
    model.save(os.path.join(wandb.run.dir, "model_wandb.h5"))
####################################################################################################

####################################################################################################

def gru_m2o_seq(experiment_dict: dict):
    """
    GRU test. One to one.
    https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    ####################################################################################################
    experiment_name = experiment_dict['experiment_name']
    experiment_description = experiment_dict['experiment_description']
    target_folder = experiment_dict['target_folder']
    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # Save dictionary in json as well
    utils.save_json_file(experiment_dict, target_folder + "experiment_params.json")
    # Configure wandb
    # Toggle Offline Mode
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(name=experiment_name,
               notes=experiment_description,
               config=experiment_dict,
               dir=target_folder,
               entity="theodormarcu",
               project="brain2brain")
    # Save Hyperparams
    config = wandb.config
    # Read saved paths for training.
    saved_paths = utils.get_file_paths(config.path)
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(config.electrode_selection)
    # Sampling of electrodes.
    timesteps_per_sample = int(config.lookback_window // config.samples_per_second)
    # Training Generator
    train_generator = generators.FGenerator(file_paths=train,
                                            lookback=config.lookback_window,
                                            length=config.length_pred,
                                            delay=config.delay_pred,
                                            batch_size=config.batch_size,
                                            sample_period=config.samples_per_second,
                                            electrodes=config.electrode_selection,
                                            electrode_output_ix=config.electrode_out,
                                            shuffle=True,
                                            debug=config.debug_mode)
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val,
                                          lookback=config.lookback_window,
                                          length=config.length_pred,
                                          delay=config.delay_pred,
                                          batch_size=config.batch_size,
                                          sample_period=config.samples_per_second,
                                          electrodes=config.electrode_selection,
                                          electrode_output_ix=config.electrode_out,
                                          shuffle=False,
                                          debug=config.debug_mode)

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")

    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=1,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=target_folder+"model_checkpoint.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=True,
        )
    ]
     # GRU One-to-one Architecture
    model = Sequential()
    model.add(GRU(config.hidden_units,
                  activation=config.activation,
                  return_sequences=config.return_sequences,
                  input_shape=(config.lookback_window, 1),
                  dropout=config.dropout_rate,
                  recurrent_dropout=config.recurrent_dropout))
    print(model.output_shape)
    model.add(GRU(config.hidden_units, activation=config.activation))
    model.add(Dense(config.length_pred))
    model.compile(optimizer=config.opt, loss=config.loss)
    # Save Summary
    model.summary()
    if config.opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=config.loss)
    else:
        raise Exception(f"Could not find optimizer {config.opt} Aborting.")
    print("calling compile")
    # Save Model Config and Architecture
    utils.save_json_file(model.get_config(), target_folder + "model_config.json")
    utils.save_json_file(model.to_json(), target_folder + "model_architecture.json")

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=config.epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
                                  validation_steps=val_steps)

    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    # Save History to File (For Later)
    utils.save_json_file(history.history, target_folder + "history.json")
    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(target_folder + "train_val_loss_plot.png")
    plt.clf()

    predictions = model.predict_generator(val_generator, steps=val_steps,
                                          callbacks=None, max_queue_size=10, workers=1,
                                          use_multiprocessing=True, verbose=1)

#     predictions_path = target_folder + "predictions.json"
#     np.save(predictions_path, predictions)

#     plt.figure()
#     plt.plot(predictions)
#     targets = []
#     for i in range(len(val_generator)):
#         x, y = val_generator[i]
#         for target in y:
#             targets.append(target[0][0])
#     plt.plot(targets)
#     plt.title('Actual vs predicted.' + experiment_name)
#     plt.legend(['predicted', 'actual'])
#     plt.savefig(target_folder + "predict_vs_targets.png")
#     plt.clf()
    wandb.save("wandb.h5")
    model.save(os.path.join(wandb.run.dir, "model_wandb.h5"))
# ####################################################################################################

####################################################################################################

def gru_o2o_stack(experiment_dict: dict):
    """
    GRU test. One to one.
    https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
    https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    """
    ####################################################################################################
    experiment_name = experiment_dict['experiment_name']
    experiment_description = experiment_dict['experiment_description']
    target_folder = experiment_dict['target_folder']
    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # Save dictionary in json as well
    utils.save_json_file(experiment_dict, target_folder + "experiment_params.json")
    # Configure wandb
    # Toggle Offline Mode
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(name=experiment_name,
               notes=experiment_description,
               config=experiment_dict,
               dir=target_folder,
               entity="theodormarcu",
               project="brain2brain")
    # Save Hyperparams
    config = wandb.config
    # Read saved paths for training.
    # saved_paths = utils.get_file_paths(config.path)
    train_paths = utils.get_file_paths(config.train_path)
    val_paths = utils.get_file_paths(config.val_path)
    # Split the train files into a training and validation set.
    # train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train_paths[0])[1]
    # Electrodes
    electrode_count = 1
    # Sampling of electrodes.
    timesteps_per_sample = int(config.lookback_window // config.samples_per_second)
    # Training Generator
    train_generator = generators.FGenerator(file_paths=train_paths,
                                            lookback=config.lookback_window,
                                            length=config.length_pred,
                                            delay=config.delay_pred,
                                            batch_size=config.batch_size,
                                            sample_period=config.samples_per_second,
                                            electrodes=[config.electrode],
                                            electrode_output_ix=config.electrode,
                                            shuffle=True,
                                            debug=config.debug_mode)
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val_paths,
                                            lookback=config.lookback_window,
                                            length=config.length_pred,
                                            delay=config.delay_pred,
                                            batch_size=config.batch_size,
                                            sample_period=config.samples_per_second,
                                            electrodes=[config.electrode],
                                            electrode_output_ix=config.electrode,
                                            shuffle=False,
                                            debug=config.debug_mode)

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")

    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=1,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=target_folder+"model_checkpoint.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=True,
        )
    ]
     # GRU One-to-one Architecture
    model = Sequential()
    model.add(GRU(config.hidden_units,
                  activation=config.activation,
                  return_sequences=config.return_sequences,
                  input_shape=(config.lookback_window, 1),
                  dropout=config.dropout_rate,
                  recurrent_dropout=config.recurrent_dropout))
    print(model.output_shape)
    model.add(GRU(config.hidden_units,
                  activation=config.activation))
    model.add(Dense(config.length_pred))
    model.compile(optimizer=config.opt, loss=config.loss)
    # Save Summary
    model.summary()
    # Save Model Config and Architecture
    utils.save_json_file(model.get_config(), target_folder + "model_config.json")
    utils.save_json_file(model.to_json(), target_folder + "model_architecture.json")

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=config.epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
                                  validation_steps=val_steps)

    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    # Save History to File (For Later)
    utils.save_json_file(history.history, target_folder + "history.json")
    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(target_folder + "train_val_loss_plot.png")
    plt.clf()

    # predictions = model.predict_generator(val_generator, steps=val_steps,
                                        #   callbacks=None, max_queue_size=10, workers=1,
                                        #   use_multiprocessing=True, verbose=1)
    wandb.save("wandb.h5")
    model.save(os.path.join(wandb.run.dir, "model_wandb.h5"))
# ####################################################################################################

####################################################################################################

def gru_single_sweep(experiment_dict: dict):
    """
    GRU for multiple one-to-one electrodes to identify the best one to predict.
    """
    ####################################################################################################
    experiment_name = experiment_dict['experiment_name']
    experiment_description = experiment_dict['experiment_description']
    target_folder = experiment_dict['target_folder']

    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # Save dictionary in json as well
    utils.save_json_file(experiment_dict, target_folder + "experiment_params.json")
    # Configure wandb
    # Toggle Offline Mode
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(name=experiment_name,
               notes=experiment_description,
               config=experiment_dict,
               dir=target_folder,
               entity="theodormarcu",
               project="brain2brain")
    # Save Hyperparams
    config = wandb.config
    # Read saved paths for training.
    saved_paths = utils.get_file_paths(config.path)
    # Split the train files into a training and validation set.
    train, val = utils.split_file_paths(saved_paths, 0.8)
    total_electrode_count = utils.get_file_shape(train[0])[1]
    # Electrodes
    electrode_count = len(config.electrode_selection)
    # Sampling of electrodes.
    timesteps_per_sample = int(config.lookback_window // config.samples_per_second)
    # Training Generator
    train_generator = generators.FGenerator(file_paths=train,
                                            lookback=config.lookback_window,
                                            length=config.length_pred,
                                            delay=config.delay_pred,
                                            batch_size=config.batch_size,
                                            sample_period=config.samples_per_second,
                                            electrodes=config.electrode_selection,
                                            electrode_output_ix=config.electrode_out,
                                            shuffle=True,
                                            debug=config.debug_mode)
    # Validation Generator
    val_generator = generators.FGenerator(file_paths=val,
                                          lookback=config.lookback_window,
                                          length=config.length_pred,
                                          delay=config.delay_pred,
                                          batch_size=config.batch_size,
                                          sample_period=config.samples_per_second,
                                          electrodes=config.electrode_selection,
                                          electrode_output_ix=config.electrode_out,
                                          shuffle=False,
                                          debug=config.debug_mode)

    train_steps = len(train_generator)
    val_steps = len(val_generator)

    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")

    callbacks_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=target_folder+"model_checkpoint.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        WandbCallback(
            monitor="val_loss",
            mode="min",
            save_model=True,
        )
    ]
    # GRU One-to-one Architecture
    model = Sequential()
    model.add(GRU(units=config.hidden_units,
                  dropout=config.dropout_rate,
                  recurrent_dropout=config.recurrent_dropout,
                  input_shape=(timesteps_per_sample, electrode_count),
                  return_sequences = experiment_dict['return_sequences']))
    # Length of prediction
    model.add(Dense(config.length_pred))
    # Workaround because config does not work with copy.deepcopy
    # https://github.com/wandb/client/issues/833 fix does not work
    length_pred = config.length_pred
    model.add(Lambda(lambda x: x[:, -length_pred:, :]))
    if config.opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=config.loss)
    else:
        raise Exception(f"Could not find optimizer {config.opt} Aborting.")
    # Save Summary
    model.summary()
    if config.opt == "RMSprop":
        model.compile(optimizer=RMSprop(), loss=config.loss)
    else:
        raise Exception(f"Could not find optimizer {config.opt} Aborting.")
    print("calling compile")
    # Save Model Config and Architecture
    utils.save_json_file(model.get_config(), target_folder + "model_config.json")
    utils.save_json_file(model.to_json(), target_folder + "model_architecture.json")

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=config.epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
                                  validation_steps=val_steps)

    model.save(target_folder + "model.h5")
    model.save_weights(target_folder + 'model_weights.h5')
    # Save History to File (For Later)
    utils.save_json_file(history.history, target_folder + "history.json")
    # Plot Loss Curves for Validation and Training
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(target_folder + "train_val_loss_plot.png")
    plt.clf()

    # predictions = model.predict_generator(val_generator, steps=val_steps,
                                        #   callbacks=None, max_queue_size=10, workers=1,
                                        #   use_multiprocessing=True, verbose=1)

#     predictions_path = target_folder + "predictions.json"
#     np.save(predictions_path, predictions)

#     plt.figure()
#     plt.plot(predictions)
#     targets = []
#     for i in range(len(val_generator)):
#         x, y = val_generator[i]
#         for target in y:
#             targets.append(target[0][0])
#     plt.plot(targets)
#     plt.title('Actual vs predicted.' + experiment_name)
#     plt.legend(['predicted', 'actual'])
#     plt.savefig(target_folder + "predict_vs_targets.png")
#     plt.clf()
    wandb.save("wandb.h5")
    model.save(os.path.join(wandb.run.dir, "model_wandb.h5"))
# ####################################################################################################
