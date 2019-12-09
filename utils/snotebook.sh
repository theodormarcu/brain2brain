#!/bin/bash
#SBATCH --time 01:10:00
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --job-name=snote
#SBATCH -o 'jupyter-%A.log'
#none --gres=gpu:1

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

module load anaconda
source activate kerasgpu
jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip

