#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time 02:05:00
#SBATCH --job-name jupyter_tmarcu
#SBATCH --output jupyter-%J.log
#SBATCH --job-name=jupyter
#SBATCH -o 'jupyter-%A.log'

ipnip=$(hostname -i)
ipnport=$(shuf -i8000-9999 -n1)
XDG_RUNTIME_DIR=""

echo -e "
    This is your ssh port-forwarding command:
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@tigergpu.princeton.edu
    -----------------------------------------------------------------
    This is your url:
    ------------------------------------------------------------------
    http://localhost:$ipnport
    ------------------------------------------------------------------
    "

module load anaconda3
conda activate brain2brain_env
jupyter notebook --no-browser --port=$ipnport --ip=$ipnip
