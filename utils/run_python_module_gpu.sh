#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:1
#SBATCH --time 3:00:00
#SBATCH --job-name neural_net_tmarcu
#SBATCH --output neural-net-%J.log
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tmarcu@princeton.edu

# load modules or conda environments here
module load anaconda3

# load brain2brain_env
conda activate brain2brain_env

echo "Running Scheduled Experiment."
experiment_name=$1
echo $experiment_name

experiment_param_file=$2
echo $experiment_param_file

# target_folder=$3
# echo $target_folder

# Run experiment. -u = unbuffered printing.
python3 -u -m brain2brain $experiment_name $experiment_param_file