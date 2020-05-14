#!/usr/bin/env python3
####################################################################################################
'''
This module contains a script for hyperparameter optimization for
the m2o model.

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
sys.path.append('../')
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



# Load experiment defaults

with open("/home/tmarcu/brain2brain/experiments/experiment_params/tcn_experiment_m2o_7_noseq.json") as f:
    experiment_dict = json.load(f)

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