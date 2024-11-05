import os
import json
import numpy as np

# Define the paths where the text evaluation folders are located
# text_eval_folders = [
#     "results/mbm_paper_texteval", 
#     "results/mbm_paper_texteval_2", 
#     "results/mbm_paper_texteval_3", 
#     "results/mbm_paper_texteval_4", 
#     "results/mbm_paper_texteval_5", 
#     "results/mbm_paper_texteval_6"
# ]

text_eval_folders = [
    "results/mbm_paper_texteval_1b"
]

# Metrics to extract: accuracy (acc,none) for each task
task_keys = [
    "agieval_sat_en/acc,none",
    "arc_easy/acc,none",
    "bigbench_conceptual_combinations_multiple_choice/acc,none",
    "bigbench_cs_algorithms_multiple_choice/acc,none",
    "boolq/acc,none",
    "copa/acc,none",
    "hellaswag/acc,none",
    "mathqa/acc,none",
    "piqa/acc,none",
    "pubmedqa/acc,none"
]


# Gather all values for each task
all_task_metrics = {key: [] for key in task_keys}

# Load the files and extract the metrics
files_data = []

for folder in text_eval_folders:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            file_metrics = {}
            for key in task_keys:
                keys = key.split('/')
                if "results" in data:
                    value = data["results"]
                else:
                    value = data
                for k in keys:
                    value = value.get(k, None)
                    if value is None:
                        break
                if value is not None:
                    file_metrics[key] = value
                    all_task_metrics[key].append(value)
            files_data.append((file_path, file_metrics))

# Compute means and standard deviations for each task
means = {key: np.mean(all_task_metrics[key]) for key in task_keys}
stds = {key: np.std(all_task_metrics[key]) for key in task_keys}

# Compute z-scores for each file and update
for file_path, file_metrics in files_data:
    z_scores = {}
    global_z_score_sum = 0
    valid_z_scores_count = 0
    
    for key in task_keys:
        if key in file_metrics:
            z = (file_metrics[key] - means[key]) / stds[key]
            z_scores[f"{key.split('/')[0]}-z-score"] = z
            global_z_score_sum += z
            valid_z_scores_count += 1

    # Compute the global z-score as the average of valid z-scores
    global_z_score = global_z_score_sum / valid_z_scores_count if valid_z_scores_count > 0 else None

    # Update the file with the z-score data
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data['z-score'] = {
            "value": global_z_score,
            **z_scores
        }
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
