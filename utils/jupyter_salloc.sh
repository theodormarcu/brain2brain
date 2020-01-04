# Unset 'XDG_RUNTIME_DIR' to avoid permission issue:
export XDG_RUNTIME_DIR=""

module load anaconda3

# For Jupyter Lab:
jupyter-lab --no-browser --port=8889 --ip=0.0.0.0