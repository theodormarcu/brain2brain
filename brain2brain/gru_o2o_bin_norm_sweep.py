#!/usr/bin/env python3
####################################################################################################
'''
This module contains the code for a sweep on all electrodes.

Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
####################################################################################################
# Imports
import sys
sys.path.append('../')
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


experiment_dict = {
    "experiment_name": "gru_o2o_sweep_676_bin_norm",
    "experiment_description" : "Attempting to predict signals for all electrodes in order to better understand which ones work. All hyperparameters below get passed to hyperparameters_defaults.",
    "target_folder" : "gru_o2o_sweep_676_bin_norm/",
    "train_path" : "/home/tmarcu/brain2brain/brain2brain/train_676_bin_norm_2.txt",
    "val_path" : "/home/tmarcu/brain2brain/brain2brain/val_676_bin_norm_2.txt",
    "batch_size" : 512,
    "epochs" : 128,
    "early_stopping_patience": 2,
    "lookback_window" : 15,
    "length_pred" : 3,
    "delay_pred" : 0,
    "electrode" : 0,
    "activation" : "relu",
    "samples_per_second" : 1,
    "return_sequences" : True,
    "debug_mode" : False,
    "hidden_units": 100,
    "dropout_rate": 0.05,
    "recurrent_dropout": 0.05,
    "activation" : "linear",
    "opt" : "adam",
    "loss" : "mse"
}

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
plt.title(f"Training and Validation Loss. Electrode {config.electrode}")
plt.savefig(target_folder + "train_val_loss_plot.png")
plt.clf()