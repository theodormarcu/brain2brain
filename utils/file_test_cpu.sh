#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time 0:05:00
#SBATCH --job-name=jupyter-tmarcu
#SBATCH -o 'jupyter-%A.log'
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tmarcu@princeton.edu

# load modules or conda environments here
module load anaconda3

# load brain2brain_env
conda activate brain2brain_env

# fake work
echo "fake work"
sleep 20 &
pid=$!
kill $pid
wait $pid
echo $pid was terminated.

cp "jupyter-"$SLURM_JOB_ID".log" "test.log"


