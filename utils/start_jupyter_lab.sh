#!/bin/bash

# load modules or conda environments here
module load anaconda3

# load brain2brain_env
conda activate brain2brain_env

# Run Jupyter
jupyter-lab --no-browser --port=8889 --ip=127.0.0.1

# List Running Notebooks
jupyter-lab list



