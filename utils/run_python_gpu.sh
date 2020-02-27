#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time 03:05:00
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


# export ScratchDir="/tmp/tmarcu"
# echo "Copying Files to: " $ScratchDir
# mkdir -p $ScratchDir
# cp -r /projects/HASSON/247/data/normalized-conversations $ScratchDir
# echo "Files copied! Running script."

filename=$1
python3 -m $filename

# echo "Removing files from: " $ScratchDir
# rm -r $ScratchDir
# echo "Files removed!"
# 

