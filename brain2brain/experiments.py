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
from tensorflow.keras.layers import Dense
# import wandb
sys.path.append('../')

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
    experiment_description = experiment_dict.experiment_description
    file_prefix = experiment_dict.file_prefix,
    path = experiment_dict.path,
    batch_size = experiment_dict.batch_size,
    epochs = experiment_dict.epochs, 
    lookback_window = experiment_dict.lookback_window,
    length_pred = experiment_dict.length_pred,
    delay_pred = experiment_dict.delay_pred,
    samples_per_second = experiment_dict.samples_per_second,
    electrode_selection=electrode_list,
    debug_mode = experiment_dict.debug_mode,
    num_feat = experiment_dict.num_feat,
    num_classes = experiment_dict.num_classes,
    nb_filters = experiment_dict.nb_filters,
    kernel_size = experiment_dict.kernel_size,
    dilations=dilations,
    nb_stacks = experiment_dict.nb_stacks,
    output_len = experiment_dict.output_len,
    padding = experiment_dict.padding,
    use_skip_connections = experiment_dict.use_skip_connections,
    return_sequences = experiment_dict.return_sequences,
    regression = experiment_dict.regression,
    dropout_rate = experiment_dict.dropout_rate,
    name = experiment_dict.name,
    kernel_initializer = experiment_dict.kernel_initializer,
    activation = experiment_dict.activation,
    opt = experiment_dict.opt,
    lr = experiment_dict.lr

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
    # print(f"Experiment Settings:\n"
    #       f"Experiment Description: {experiment_description}\n"
    #       f"saved_paths: {path}\n"
    #       f"batch_size: {batch_size}\n"
    #       f"epochs: {epochs}\n"
    #       f"lookback_window: {lookback_window//512}s\n"
    #       f"length_prediction: {length_pred} timesteps; {length_pred//512}s\n"
    #       f"delay_pred: {delay_pred} timesteps; {delay_pred//512}s\n"
    #       f"samples_per_second: {512//samples_per_second}\n"
    #       f"timesteps_per_sample: {timesteps_per_sample}\n"
    #       f"electrode_selection: {electrode_selection}\n"
    #       f"electrode_count: {electrode_count}\n")
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
    dilations = [1, 2, 4, 8, 16, 32]
    print(dilations)
    print(len(dilations))
    model = compiled_tcn(num_feat=1,
                         num_classes=1,
                         nb_filters=16,
                         kernel_size=3,
                         dilations=dilations,
                         nb_stacks=1,
                         max_len=timesteps_per_sample,
                         output_len=1,
                         padding='causal',
                         use_skip_connections=True,
                         return_sequences=False,
                         regression=True,
                         dropout_rate=0.05,
                         name="tcn",
                         kernel_initializer="he_normal",
                         activation="linear",
                         opt="adam",
                         lr=0.0005)
    # Save Summary
    summary = model.summary()
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

    model_history = model.fit_generator(generator=train_generator,
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        validation_data=val_generator,
                                        validation_steps=val_steps)

    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')
    # Save History to File (For Later)
    model_history_path = file_prefix + "model_history.json"
    with open(model_history_path, 'w') as outfile:
        json.dump(model_history.history, outfile)
    # Plot Loss Curves for Validation and Training
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
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

####################################################################################################


def tcn_experiment2():
    """
    Testing TCN on 1 electrode for patient 676.
    No activation.
    Data is normalized.
    02/27/2020
    """

    # Description (For printing.)
    experiment_description = "TCN on 1 electrode for patient 676. No activation."
    # Debug Mode (for the generator to use less data.) => Reduces training time.
    debug_mode = False

    file_prefix = "experiment_test_676_bs128_tcn_2_"
    # wandb.init(project=file_prefix+"wandb")
    # Read saved paths for training.
    path = "./brain2brain/train_676_norm_files_projects.txt"
    saved_paths_676 = utils.get_file_paths(path)
    batch_size = 128
    epochs = 5
    # Split the train files into a training and validation set.
    train_676, val_676 = utils.split_file_paths(saved_paths_676, 0.8)
    total_electrode_count = 114
    # The time we look back.
    lookback_window = 512 * 5  # 5 seconds
    # Length of sequence predicted.
    length_pred = 1  # 1 timestep
    # Delay between lookback and length.
    delay_pred = 0  # No delay.
    # Sampling of electrodes.
    samples_per_second = 1  # Samples Per Second
    timesteps_per_sample = int(lookback_window // samples_per_second)
    # Electrodes
    electrode_selection = [0]
    electrode_count = len(electrode_selection)

    print(f"Running Experiment"
          f"Experiment Description: {experiment_description}"
          f"saved_paths: {path}"
          f"batch_size: {batch_size}"
          f"epochs: {epochs}"
          f"lookback_window: {lookback_window//512}s"
          f"length_prediction: {length_pred} timesteps; {length_pred//512}s"
          f"delay_pred: {delay_pred} timesteps; {delay_pred//512}s"
          f"samples_per_second: {512//samples_per_second}"
          f"timesteps_per_sample: {timesteps_per_sample}"
          f"electrode_count: {electrode_count}")

    # Training Generator
    train_676_generator = generators.FGenerator(file_paths=train_676,
                                                lookback=lookback_window, length=length_pred, delay=delay_pred,
                                                batch_size=batch_size, sample_period=samples_per_second,
                                                electrodes=electrode_selection, shuffle=True, debug=debug_mode)
    # Validation Generator
    val_676_generator = generators.FGenerator(file_paths=val_676,
                                              lookback=lookback_window, length=length_pred, delay=delay_pred,
                                              batch_size=batch_size, sample_period=samples_per_second,
                                              electrodes=electrode_selection, shuffle=False, debug=debug_mode)

    train_steps = len(train_676_generator)
    val_steps = len(val_676_generator)

    # # TCN
    # i = Input(shape=(timesteps_per_sample, electrode_count),
    #           batch_size=batch_size)
    # m = TCN()(i)
    # # No activation.
    # m = Dense(1)(m)
    # model = Model(inputs=[i], outputs=[m])

    # Save Summary
    summary = model.summary()
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

    model_history = model.fit_generator(generator=train_676_generator,
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        validation_data=val_676_generator,
                                        validation_steps=val_steps)

    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')
    # Save History to File (For Later)
    model_history_path = file_prefix + "model_history.json"
    with open(model_history_path, 'w') as outfile:
        json.dump(model_history.history, outfile)
    # Plot Loss Curves for Validation and Training
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(file_prefix + "train_val_loss_plot.png")
    plt.clf()

    p = model.predict_generator(val_676_generator, steps=val_steps,
                                callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=True, verbose=1)
    plt.figure()
    plt.plot(p)
    targets = []
    for i in range(len(val_676_generator)):
        x, y = val_676_generator[i]
        for target in y:
            targets.append(target[0][0])
    plt.plot(targets)
    plt.title('TCN pred on 1 electrode for patient 676.')
    plt.legend(['predicted', 'actual'])
    plt.savefig(file_prefix + "plot.png")
    plt.clf()
    # wandb.save(file_prefix + "wandb.h5")
####################################################################################################


####################################################################################################
def tcn_experiment1():
    """
    Testing TCN on 1 electrode for patient 676.
    Data is normalized.
    02/23/2020
    """

    file_prefix = "experiment_test_676_bs128_2_debug_"
    # wandb.init(project=file_prefix+"wandb")
    # Read saved paths for training.
    saved_paths_676 = utils.get_file_paths(
        "../brain2brain/train_676_norm_files.txt")

    # Split the train files into a training and validation set.
    train_676, val_676 = utils.split_file_paths(saved_paths_676, 0.8)
    total_electrode_count = 114
    # The time we look back.
    lookback_window = 512 * 5  # 5 seconds
    # Length of sequence predicted.
    length_pred = 1  # 1 timestep
    # Delay between lookback and length.
    delay_pred = 0  # No delay.
    # Sampling of electrodes.
    samples_per_second = 4  # Samples Per Second
    timesteps_per_sample = int(lookback_window // samples_per_second)
    # Electrodes
    electrode_selection = [0]
    electrode_count = len(electrode_selection)

    # Training Generator
    train_676_generator = generators.FGenerator(file_paths=train_676,
                                                lookback=lookback_window, length=length_pred, delay=delay_pred,
                                                batch_size=128, sample_period=samples_per_second,
                                                electrodes=electrode_selection, shuffle=True)
    # Validation Generator
    val_676_generator = generators.FGenerator(file_paths=val_676,
                                              lookback=lookback_window, length=length_pred, delay=delay_pred,
                                              batch_size=128, sample_period=samples_per_second,
                                              electrodes=electrode_selection, shuffle=False, debug=True)

    train_steps = len(train_676_generator)
    val_steps = len(val_676_generator)

    # TCN
    i = Input(shape=(timesteps_per_sample, electrode_count))
    m = TCN()(i)
    m = Dense(1, activation='linear')(m)

    model = Model(inputs=[i], outputs=[m])
    # Save Summary
    summary = model.summary()
    model.compile('adam', 'mae')
    # Save Model Config and Architecture
    model_config = model.get_config()
    model_config_file_path = file_prefix + "model_config.json"
    with open(model_config_file_path, 'w') as outfile:
        json.dump(model_config, outfile)

    model_architecture = model.to_json()
    model_architecture_file_path = file_prefix + "model_architecture.json"
    with open(model_architecture_file_path, 'w') as outfile:
        json.dump(model_architecture, outfile)

    model.fit_generator(generator=train_676_generator,
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=val_676_generator,
                        validation_steps=val_steps)
    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')

    p = model.predict_generator(val_676_generator, steps=val_steps,
                                callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=True, verbose=1)
    plt.plot(p)
    targets = []
    for i in range(len(val_676_generator)):
        x, y = val_676_generator[i]
        for target in y:
            targets.append(target[0][0])
    plt.plot(targets)
    plt.title('TCN pred on 1 electrode for patient 676.')
    plt.legend(['predicted', 'actual'])
    plt.savefig(file_prefix + "plot.png")
    # wandb.save(file_prefix + "wandb.h5")
####################################################################################################

# def main():
    # tcn_experiment2()

# if __name__ == '__main__':
    # main()
