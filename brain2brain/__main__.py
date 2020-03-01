#!/usr/bin/env python3
'''
Main class for CLI functionality.
'''

import argparse
from brain2brain import utils
from brain2brain import generators
from brain2brain import experiments
import numpy as np

def main():
    '''
    Main function for __main__.py. This is the entry point in the CLI.
    '''
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("experiment_name", help="Name of experiment dictionary.", type=str)
    parser.add_argument("experiment_json", help="Path to json file containing experiment params.", type=str)
    # parser.add_argument("use_layer_norm", help="Layer Normalization.", type=bool)
    args = parser.parse_args()


    experiment_1 = {'experiment_description' : "Experiment 1 TCN",
                    'file_prefix' : "experiment1_",
                    'path' : './brain2brain/train_676_norm_files_projects.txt',
                    'batch_size' : 128,
                    'epochs' : 20, 
                    'lookback_window' : 512*5,
                    'length_pred' : 1,
                    'delay_pred' : 0,
                    'samples_per_second' : 1,
                    'electrode_selection' : np.arange(0, 114),
                    'debug_mode' : False,
                    'num_feat' : 1,
                    'num_classes' : 1,
                    'nb_filters' : 32,
                    'kernel_size' : 3,
                    'dilations' : [1, 2, 4, 8, 16, 32],
                    'nb_stacks' : 2,
                    'output_len' : 1,
                    'padding' : 'causal',
                    'use_skip_connections' : True,
                    'return_sequences' : False,
                    'regression' : True,
                    'dropout_rate' : 0.05,
                    'name' : 'TCN Model',
                    'kernel_initializer' : 'he_normal',
                    'activation' : 'linear',
                    'opt' : 'adam',
                    'lr' : '0.01'}
    
    switch(args.experiment_name) {
        case "experiment_1":
            print("Running experiment 1!")
            return
            experiments.tcn_experiment(experiment_dict)
        default: 
            print("Error! Could not find experiment name.")
    }
    return
 
    # Run experiment.)
    # experiments.tcn_experiment(experiment_description= experiment_dict.experiment_description,
    #                            file_prefix= experiment_dict.file_prefix,
    #                            path= experiment_dict.path,
    #                            batch_size= experiment_dict.batch_size,
    #                            epochs= experiment_dict.epochs, 
    #                            lookback_window= experiment_dict.lookback_window,
    #                            length_pred= experiment_dict.length_pred,
    #                            delay_pred= experiment_dict.delay_pred,
    #                            samples_per_second= experiment_dict.samples_per_second,
    #                            electrode_selection=electrode_list,
    #                            debug_mode= experiment_dict.debug_mode,
    #                            num_feat= experiment_dict.num_feat,
    #                            num_classes= experiment_dict.num_classes,
    #                            nb_filters= experiment_dict.nb_filters,
    #                            kernel_size= experiment_dict.kernel_size,
    #                            dilations=dilations,
    #                            nb_stacks= experiment_dict.nb_stacks,
    #                            output_len= experiment_dict.output_len,
    #                            padding= experiment_dict.padding,
    #                            use_skip_connections= experiment_dict.use_skip_connections,
    #                            return_sequences= experiment_dict.return_sequences,
    #                            regression= experiment_dict.regression,
    #                            dropout_rate= experiment_dict.dropout_rate,
    #                            name= experiment_dict.name,
    #                            kernel_initializer= experiment_dict.kernel_initializer,
    #                            activation= experiment_dict.activation,
    #                            opt= experiment_dict.opt,
    #                            lr= experiment_dict.lr)

if __name__ == '__main__':
    main()
