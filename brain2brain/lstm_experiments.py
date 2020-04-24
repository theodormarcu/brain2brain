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