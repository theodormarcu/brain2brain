#~/bin/bash
#Running: bash train_model.sh --model_type tcn_d --model_name new_model

# Load Anaconda and Environment
# To recreate environment, use:
# `conda env create --file environment.yml`
module load anaconda3
conda activate tiger_gpu_env

sbatch strain.sh \
    --electrode "$electrode" \
    --num_conversations "$num_conversations" \
    --learning_rate "$learning_rate" \
    --lag "$lag" \
    --input_length "$window_size" \
    "$@"-e"$electrode"-w"$window_size"-n"$num_conversations"-r"$learning_rate"-l"$lag"
