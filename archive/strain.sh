#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --mem=80GB
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH -o 'training-%A.log'

# You can use `salloc --gres=gpu:1 --time 01:10:00` to test something

module load anaconda
#module load cudatoolkit/9.2
#module load cudnn/cuda-9.2
conda activate tiger_gpu_env

echo 'Start time:' `date`
echo "$@"
# python bash_style_train_tcn.py "$@"
# python get_data.py "$@"
echo 'End time:' `date`
