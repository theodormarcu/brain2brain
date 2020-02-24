'''
This module contains brain2brain experiments.


Created by Theodor Marcu 2019-2020
tmarcu@princeton.edu
'''

from brain2brain.utils import utils

class Experiments:
    '''
    TODO: Add class description.
    '''

    @staticmethod
    def baseline():
        ''' Basic TCN experiment.
        '''
        # Preparing the training, validation, and test generators
        lookback = 5120 # Observations will go back 10s.
        step = 128 # Observations will be sampled at 4 data points per second.
        delay = 256 # Targets are 0.5s in the future.
        batch_size = 64 # Number of samples per batch.

        
