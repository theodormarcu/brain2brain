#!/bin/bash

# load the anaconda environment module
module load anaconda3

# Activate environment
conda activate brain2brain_env

# For Jupyter Lab:
jupyter-lab --no-browser --port=8889 --ip=127.0.0.1
