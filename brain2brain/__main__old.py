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
    parser.add_argument("experiment_description", help="Description of experiment.")
    parser.add_argument("file_prefix", help="Prefix of output files.")
    parser.add_argument("path", help="Path to list of saved files.")
    parser.add_argument("batch_size", help="Size of the batch.", type=int)
    parser.add_argument("epochs", help="Epoch count.", type=int)
    parser.add_argument("lookback_window", help="Number of steps to look back \
                        on. Each second of data has 512 steps.", type=int)
    parser.add_argument("length_pred", help="The length of the prediction.", type=int)
    parser.add_argument("delay_pred", help="The delay between the prediction \
                        and the lookback window.", type=int)
    parser.add_argument("samples_per_second", help="Number of samples per second.", type=int)
    parser.add_argument("start_electrode_selection", help="The first \
                        electrode in the selection.", type=int)
    parser.add_argument("end_electrode_selection", help="The last electrode \
                        in the selection, non-inclusive.", type=int)
    parser.add_argument("debug_mode", help="Debug mode.", type=bool)
    parser.add_argument("num_feat", help="The number of features.", type=int)
    parser.add_argument("num_classes", help="The number of classes.", type=int)
    parser.add_argument("nb_filters", help="The number of filters.", type=int)
    parser.add_argument("kernel_size", help="The size of the convolution kernel.", type=int)
    parser.add_argument("dilations", help="The list of dilations.", type=int, nargs="+")
    parser.add_argument("nb_stacks", help="The number of stacks for residual blocks to use.", type=int)
    parser.add_argument("output_len", help="The dimension of the output.", type=int)
    parser.add_argument("padding", help="What kind of padding to use.")
    parser.add_argument("use_skip_connections", help="Should the model use skip connections.", type=bool)
    parser.add_argument("return_sequences", help="Whether to return sequences.", type=bool)
    parser.add_argument("regression", help="Whether this is a regression or classification task", type=bool)
    parser.add_argument("dropout_rate", help="The dropout rate.", type=float)
    parser.add_argument("name", help="The name of the model.", type=str)
    parser.add_argument("kernel_initializer", help="Kernel Initializer.", type=str)
    parser.add_argument("activation", help="Activation Function.", type=str)
    parser.add_argument("opt", help="Optimizer.", type=str)
    parser.add_argument("lr", help="Learning Rate.", type=float)
    # parser.add_argument("use_batch_norm", help="Batch Normalization.", type=bool)
    # parser.add_argument("use_layer_norm", help="Layer Normalization.", type=bool)
    args = parser.parse_args()

    electrode_list = np.arange(args.start_electrode_selection, args.end_electrode_selection)
    dilations = args.dilations
    print(dilations)
    return

    # Run experiment.
    experiments.tcn_experiment(experiment_description=args.experiment_description,
                               file_prefix=args.file_prefix,
                               path=args.path,
                               batch_size=args.batch_size,
                               epochs=args.epochs, 
                               lookback_window=args.lookback_window,
                               length_pred=args.length_pred,
                               delay_pred=args.delay_pred,
                               samples_per_second=args.samples_per_second,
                               electrode_selection=electrode_list,
                               debug_mode=args.debug_mode,
                               num_feat=args.num_feat,
                               num_classes=args.num_classes,
                               nb_filters=args.nb_filters,
                               kernel_size=args.kernel_size,
                               dilations=dilations,
                               nb_stacks=args.nb_stacks,
                               output_len=args.output_len,
                               padding=args.padding,
                               use_skip_connections=args.use_skip_connections,
                               return_sequences=args.return_sequences,
                               regression=args.regression,
                               dropout_rate=args.dropout_rate,
                               name=args.name,
                               kernel_initializer=args.kernel_initializer,
                               activation=args.activation,
                               opt=args.opt,
                               lr=args.lr)

if __name__ == '__main__':
    main()
