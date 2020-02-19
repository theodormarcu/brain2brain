#~/bin/bash
#Running: bash eval_model.sh --electrodes 59 38 13 50 --input_length 2 4 8 16 24  --lag 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 32 64 128 256 512 1024 2048 4096 --model_type tcn_d --model_name new_model

module load anaconda
conda activate tiger_gpu_env




for num_conversations in 6; do
	for learning_rate in 0.001; do
		python eval_tcn_update.py \
			--electrode "$electrode" \
			--num_conversations "$num_conversations" \
			--learning_rate "$learning_rate" \
			--lag "$lag" \
			--input_length "$window_size" \
			"$@"-e"$electrode"-w"$window_size"-n"$num_conversations"-r"$learning_rate"-l"$lag"
	done
done
