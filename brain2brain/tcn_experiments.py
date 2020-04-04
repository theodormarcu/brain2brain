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
import os
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
# Wandb
import wandb
from wandb.keras import WandbCallback

sys.path.append('../')

####################################################################################################

def tcn_experiment(experiment_dict: dict):
    """
    Testing TCN general suite.
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


    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")

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

    print(f"Train Generator Batch Shape:\n"
          f"Sample={train_generator[0][0].shape} Pred={train_generator[0][1].shape}")
    print(f"Validation Generator Batch Shape:\n"
          f"Sample={val_generator[0][0].shape} Pred={val_generator[0][1].shape}")

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
    model_summary_path = target_folder + "model_summary.txt"
    with open(model_summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print("calling compile")
    # model.compile('adam', 'mae')
    # Save Model Config and Architecture
    model_config = model.get_config()
    model_config_file_path = target_folder + "model_config.json"
    with open(model_config_file_path, 'w') as outfile:
        json.dump(model_config, outfile)
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  validation_data=val_generator,
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
    # wandb.save(target_folder + "wandb.h5")
####################################################################################################

def tcn_m2m(experiment_dict: dict):
    """
    TCN for many-to-many ouputs (many electrodes to many electrodes).
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
    loss = experiment_dict['loss']

    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # wandb.init(project=target_folder+"wandb")
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
        )
    ]
    input_layer = Input(shape=(lookback_window, electrode_count))
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            nb_stacks=nb_stacks,
            padding=padding,
            use_skip_connections=use_skip_connections,
            return_sequences=return_sequences,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer)(input_layer)
    x = TimeDistributed(Dense(electrode_count))(x)
    # x = Lambda(lambda x: x[:, :length_pred , :])(x)
    output_layer = x
    model = Model(input_layer, output_layer)
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
    print("calling compile")
    # model.compile('adam', 'mae')
    # Save Model Config and Architecture
    model_config = model.get_config()
    model_config_file_path = target_folder + "model_config.json"
    with open(model_config_file_path, 'w') as outfile:
        json.dump(model_config, outfile)
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
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

    # plt.figure()
    # plt.plot(p)
    # targets = []
    # for i in range(len(val_generator)):
    #     x, y = val_generator[i]
    #     for target in y:
    #         targets.append(target[0][0])
    # plt.plot(targets)
    # plt.title('Actual vs predicted.')
    # plt.legend(['predicted', 'actual'])
    # plt.savefig(target_folder + "plot.png")
    # plt.clf()
    # wandb.save(target_folder + "wandb.h5")
####################################################################################################

####################################################################################################

def tcn_m2m_noseq(experiment_dict: dict):
    """
    TCN for many-to-many ouputs (many electrodes to many electrodes).
    This experiment returns no sequences, but uses the fix described in:
    https://github.com/philipperemy/keras-tcn#supported-task-types
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
    loss = experiment_dict['loss']

    # Ensure target directory exists.
    try:
        Path(target_folder).mkdir(parents=True, exist_ok=True)
    except IOError:
        print(f"Directory creation failed for path {target_folder}")
    # wandb.init(project=target_folder+"wandb")
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
        )
    ]
    input_layer = Input(shape=(lookback_window, electrode_count))
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            nb_stacks=nb_stacks,
            padding=padding,
            use_skip_connections=use_skip_connections,
            return_sequences=return_sequences,
            activation=activation,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer)(input_layer)
    # x = TimeDistributed(Dense(electrode_count))(x)
    # x = Lambda(lambda x: x[:, :length_pred , :])(x)
    x = Dense(electrode_count)(x)
    output_layer = x
    model = Model(input_layer, output_layer)
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
    print("calling compile")
    # model.compile('adam', 'mae')
    # Save Model Config and Architecture
    model_config = model.get_config()
    model_config_file_path = target_folder + "model_config.json"
    with open(model_config_file_path, 'w') as outfile:
        json.dump(model_config, outfile)
    model_architecture = model.to_json()
    model_architecture_file_path = target_folder + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  callbacks=callbacks_list,
                                  validation_data=val_generator,
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

    # plt.figure()
    # plt.plot(p)
    # targets = []
    # for i in range(len(val_generator)):
    #     x, y = val_generator[i]
    #     for target in y:
    #         targets.append(target[0][0])
    # plt.plot(targets)
    # plt.title('Actual vs predicted.')
    # plt.legend(['predicted', 'actual'])
    # plt.savefig(target_folder + "plot.png")
    # plt.clf()
    # wandb.save(target_folder + "wandb.h5")
####################################################################################################

####################################################################################################

def tcn_m2o_noseq(experiment_dict: dict):
    """
    TCN for many-to-one ouputs (many electrodes to one electrode).
    """
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
    # Model Architecture
    input_layer = Input(shape=(config.lookback_window, electrode_count))
    x = TCN(nb_filters=config.nb_filters,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            nb_stacks=config.nb_stacks,
            padding=config.padding,
            use_skip_connections=config.use_skip_connections,
            return_sequences=config.return_sequences,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            kernel_initializer=config.kernel_initializer)(input_layer)
    # x = TimeDistributed(Dense(electrode_count))(x)
    # x = Lambda(lambda x: x[:, :length_pred , :])(x)
    x = Dense(1)(x)
    output_layer = x
    model = Model(input_layer, output_layer)
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