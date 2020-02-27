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
# TF
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
# import wandb
sys.path.append('../')

####################################################################################################
def tcn_experiment2():
    """
    Testing TCN on 1 electrode for patient 676.
    No activation.
    Data is normalized.
    02/27/2020
    """

    # Description (For printing.)
    experiment_description="GRU on 1 electrode for patient 676. No activation."
    # Debug Mode (for the generator to use less data.) => Reduces training time.
    debug_mode = False
    
    file_prefix = "experiment_test_676_bs128_gru_"
    # wandb.init(project=file_prefix+"wandb")
    # Read saved paths for training.
    path = "./brain2brain/train_676_norm_files_projects.txt"
    saved_paths_676 = utils.get_file_paths(path)
    batch_size = 128
    epochs = 20
    # Split the train files into a training and validation set.
    train_676, val_676 = utils.split_file_paths(saved_paths_676, 0.8)
    total_electrode_count = 114
    # The time we look back.
    lookback_window = 512 * 5 # 5 seconds
    # Length of sequence predicted.
    length_pred = 1 # 1 timestep
    # Delay between lookback and length.
    delay_pred = 0 # No delay.
    # Sampling of electrodes.
    samples_per_second = 1 # Samples Per Second
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
    train_676_generator = generators.FGenerator(file_paths = train_676,
                                                lookback=lookback_window, length = length_pred, delay = delay_pred,
                                                batch_size = batch_size, sample_period = samples_per_second,
                                                electrodes= electrode_selection, shuffle = True, debug=debug_mode)
    # Validation Generator
    val_676_generator = generators.FGenerator(file_paths = val_676,
                                            lookback=lookback_window, length = length_pred, delay = delay_pred,
                                            batch_size = batch_size, sample_period = samples_per_second,
                                            electrodes= electrode_selection, shuffle = False, debug=debug_mode)

    train_steps = len(train_676_generator)
    val_steps = len(val_676_generator)

    # TCN
    i = Input(shape=(timesteps_per_sample, electrode_count), batch_size=batch_size)
    m = TCN()(i)
    # No activation.
    m = Dense(1)(m)
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

    model_history = model.fit_generator(generator=train_676_generator,
                                        steps_per_epoch=train_steps,
                                        epochs=epochs,
                                        validation_data=val_676_generator,
                                        validation_steps=val_steps)

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

    model.save(file_prefix + "model.h5")
    model.save_weights(file_prefix + 'model_weights.h5')
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
    saved_paths_676 = utils.get_file_paths("../brain2brain/train_676_norm_files.txt")

    # Split the train files into a training and validation set.
    train_676, val_676 = utils.split_file_paths(saved_paths_676, 0.8)
    total_electrode_count = 114
    # The time we look back.
    lookback_window = 512 * 5 # 5 seconds
    # Length of sequence predicted.
    length_pred = 1 # 1 timestep
    # Delay between lookback and length.
    delay_pred = 0 # No delay.
    # Sampling of electrodes.
    samples_per_second = 4 # Samples Per Second
    timesteps_per_sample = int(lookback_window // samples_per_second)
    # Electrodes
    electrode_selection = [0]
    electrode_count = len(electrode_selection)

    # Training Generator
    train_676_generator = generators.FGenerator(file_paths = train_676,
                                                lookback=lookback_window, length = length_pred, delay = delay_pred,
                                                batch_size = 128, sample_period = samples_per_second,
                                                electrodes= electrode_selection, shuffle = True)
    # Validation Generator
    val_676_generator = generators.FGenerator(file_paths = val_676,
                                            lookback=lookback_window, length = length_pred, delay = delay_pred,
                                            batch_size = 128, sample_period = samples_per_second,
                                            electrodes= electrode_selection, shuffle = False, debug=True)

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


def main():
    tcn_experiment2()

if __name__ == '__main__':
    main()