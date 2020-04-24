#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=50G
#SBATCH --time 1:00:00
#SBATCH --job-name sweep_tmarcu
#SBATCH --output %J.log
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tmarcu@princeton.edu

# load modules or conda environments here
module load anaconda3

# load brain2brain_env
conda activate brain2brain_env

# Sweep
wandb sweep --controller gru_o2o_bin_norm_sweep.yaml
