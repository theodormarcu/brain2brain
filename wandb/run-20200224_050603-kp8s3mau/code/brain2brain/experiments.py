'''
This module contains brain2brain experiments.


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
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

import wandb

sys.path.append('../')

def tcn_experiment1():
    """
    Testing TCN on 1 electrode for patient 676.
    Data is normalized.
    02/23/2020
    """
    
    file_prefix = "experiment_test_676_bs128_"
    wandb.init(project=file_prefix+"wandb")
    # Read saved paths for training.
    saved_paths_676 = utils.get_file_paths("./brain2brain/train_676_norm_files_projects.txt")

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
    samples_per_second = 512 / 128 # Samples Per Second
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
                                            electrodes= electrode_selection, shuffle = False)

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
                        epochs=20,
                        validation_data=val_676_generator,
                        validation_steps=val_steps)
    model.save(file_prefix + "model.h5")
    p = model.predict_generator(val_676_generator, steps=val_steps,
                                callbacks=None, max_queue_size=10, workers=1,
                                use_multiprocessing=False, verbose=1)
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
    wandb.save(file_prefix + "wandb.h5")

tcn_experiment1()