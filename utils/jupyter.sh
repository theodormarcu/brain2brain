#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --time 03:05:00
#SBATCH --job-name jupyter-notebook-tmarcu
#SBATCH --output jupyter-notebook-%J.log
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tmarcu@princeton.edu

# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="tigergpu"
port=8001

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.princeton.edu

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
module load anaconda3

# load brain2brain_env
conda activate brain2brain_env

# Run Jupyter
jupyter-lab --no-browser --port=${port} --ip=${node}

# List Running Notebooks
jupyter notebook list
