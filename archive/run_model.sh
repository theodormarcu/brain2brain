#~/bin/bash
#Running: bash run_model.sh --model_type tcn_d --model_name new_model

module load anaconda
conda activate tiger_gpu_env

for electrode in 13; do
	for num_conversations in 6; do
		for learning_rate in 0.001; do
			for lag in 4; do
				for window_size in 24; do
					sbatch strain.sh \
						--electrode "$electrode" \
						--num_conversations "$num_conversations" \
						--learning_rate "$learning_rate" \
						--lag "$lag" \
						--input_length "$window_size" \
						"$@"-e"$electrode"-w"$window_size"-n"$num_conversations"-r"$learning_rate"-l"$lag"

				done
			done
		done
	done
done
