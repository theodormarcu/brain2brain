#!/usr/bin/env python3
# Theodor Marcu
# tmarcu@princeton.edu
# Created January 2020
# Computer Science Senior Thesis
'''
Main class for CLI functionality.
'''

import os
import importlib
import importlib.util
import argparse
from brain2brain import utils
from brain2brain import generators
from brain2brain import tcn_experiments, baseline_experiments, gru_experiments, optimization
import numpy as np
import pprint
import yaml


def main():
    '''
    Main function for __main__.py. This is the entry point in the CLI.
    '''
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("experiment_folder", 
                        help="Path to experiment folder that contains a YAML file and model.py.", type=str)
    args = parser.parse_args()

    params_file_name = "params.yaml"
    params_model_file_name = "model.py"
    
    # Read YAML Experiment File
    with open(os.path.join(args.experiment_folder, params_file_name)) as f:
        experiment_params = yaml.load(f, Loader=yaml.SafeLoader)

    print(experiment_params)

    model_full_path = os.path.join(args.experiment_folder, params_model_file_name)

    spec = importlib.util.spec_from_file_location("model", model_full_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    model.train(experiment_params)

if __name__ == '__main__':
    main()
