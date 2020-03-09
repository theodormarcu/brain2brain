#!/usr/bin/env python3
####################################################################################################
'''
This module contains Hyperparameter optimization code.

Using work described in:
https://github.com/Paperspace/hyperopt-keras-sample
https://hyperopt.github.io/hyperopt/getting-started/search_spaces/


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''
####################################################################################################
import numpy as np

# Hyperparameter Optimization
from hyperopt import hp, tpe

m2m_noseq_space = {
    "batch_size" : hp.quniform('batch_size', 256, 2048, 256),
    "nb_filters" : hp.quniform("nb_filters", 8, 256, 8),
    "kernel_size" : hp.quniform("kernel_size", 3, 9, 1),
    "dilations" : np.arange(start=1, stop=hp.quniform("dilations", 4, 128, 4)),
    "nb_stacks" : hp.quniform("nb_stacks", 2, 6, 1),
    "dropout_rate" : hp.uniform("dropout_rate", 0.0, 0.5),
    "kernel_initializer" : hp.choice("kernel_initializer"),
    "activation" : "linear",
    "opt" : "RMSprop",
    "lr" : hp.loguniform('lr', -0.01, 0.01)
}
