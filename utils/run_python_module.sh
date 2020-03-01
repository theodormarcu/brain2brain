#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --time 01:05:00
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

test_file_name=$1
echo $test_file_name

# Run experiment
python3 -m brain2brain @$test_file_name