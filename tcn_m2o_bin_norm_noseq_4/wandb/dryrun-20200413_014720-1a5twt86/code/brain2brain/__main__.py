#!/usr/bin/env python3
# Theodor Marcu
# tmarcu@princeton.edu
# Created January 2020
# Computer Science Senior Thesis
'''
Main class for CLI functionality.
'''

import argparse
from brain2brain import utils
from brain2brain import generators
from brain2brain import tcn_experiments, baseline_experiments, gru_experiments, optimization
import numpy as np
import pprint
import json

def main():
    '''
    Main function for __main__.py. This is the entry point in the CLI.
    '''
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("experiment_name", help="Name of experiment.", type=str)
    parser.add_argument("experiment_json", help="Path to json file containing experiment params.", type=str)
    # parser.add_argument("use_layer_norm", help="Layer Normalization.", type=bool)
    args = parser.parse_args()

    with open(args.experiment_json) as f:
        experiment_params = json.load(f)
    
    pprint.sorted = lambda x, key=None: x
    pprint.pprint(experiment_params)

    if "dry_run" in experiment_params and experiment_params["dry_run"] == True:
        return

    experiment_name = args.experiment_name
    

    if experiment_name == "tcn_experiment":
        tcn_experiments.tcn_experiment(experiment_params)
    elif experiment_name == "tcn_optimization_noseq":
        optimization.tcn_optimization_noseq(experiment_params)
    elif experiment_name == "tcn_m2m":
        tcn_experiments.tcn_m2m(experiment_params)
    elif experiment_name == "tcn_m2o_seq":
        tcn_experiments.tcn_m2o_noseq(experiment_params)
    elif experiment_name == "tcn_m2o_noseq":
        tcn_experiments.tcn_m2o_noseq(experiment_params)
    elif experiment_name == "tcn_m2m_noseq":
        tcn_experiments.tcn_m2m_noseq(experiment_params)
    elif experiment_name == "baseline_experiment":
        baseline_experiments.baseline_experiment(experiment_params)
    elif experiment_name == "gru_experiment":
        if 'many_to_many' in experiment_params:
            if experiment_params['many_to_many'] == True:
                gru_experiments.gru_m2m(experiment_params)
        else:
            gru_experiments.gru_experiment(experiment_params)
    elif experiment_name == "gru_m2o":
        gru_experiments.gru_m2o(experiment_params)
    else:
        raise Exception(f"Experiment {experiment_name} does not exist! Aborting.")
if __name__ == '__main__':
    main()
