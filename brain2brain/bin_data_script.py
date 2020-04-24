import sys
sys.path.append("../")
from brain2brain.utils import *

all_paths_676 = get_file_paths_from_root(patient_number=676, shuffle=True)

normalize_dataset(file_paths=all_paths_676, path_list_out="676_bin_norm_2",
                        split_data=True, split_ratio=0.8,
                        output_directory="/projects/HASSON/247/data/binned_conversations/676_bin_norm_2/",
                        file_prefix="bin_norm_2_", binned=True, avg_timestep_count=25)