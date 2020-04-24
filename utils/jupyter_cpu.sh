#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time 02:00:00
#SBATCH --job-name=jupyter-tmarcu
#SBATCH -o 'jupyter-%A.log'
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
