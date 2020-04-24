#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time 01:00:00
#SBATCH --job-name=bin-tmarcu
#SBATCH -o 'bin-norm-%A.log'
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tmarcu@princeton.edu

echo "HI"
cd brain2brain
python bin_data_script.py