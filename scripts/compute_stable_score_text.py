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
    # "bigbench_cs_algorithms_multiple_choice/acc,none",
    "boolq/acc,none",
    "copa/acc,none",
    "hellaswag/acc,none",
    "mathqa/acc,none",
    "piqa/acc,none",
    "pubmedqa/acc,none"
]

task_random_results = {
    "agieval_sat_en/acc,none": 0.2,   # 5 options
    "arc_easy/acc,none": 0.25,        # 4 options
    "bigbench_conceptual_combinations_multiple_choice/acc,none": 0.25,  # 4 options
    # "bigbench_cs_algorithms_multiple_choice/acc,none": 0.25,            # 4 options
    "boolq/acc,none": 0.5,            # 2 options (Yes/No)
    "copa/acc,none": 0.5,             # 2 options
    "hellaswag/acc,none": 0.25,       # 4 options
    "mathqa/acc,none": 0.2,           # 5 options
    "piqa/acc,none": 0.5,             # 2 options
    "pubmedqa/acc,none": 0.333        # 3 options (Yes/No/Maybe)
}

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

# Compute stable-scores for each file and update
for file_path, file_metrics in files_data:
    stable_scores = {}
    global_stable_score_sum = 0
    valid_stable_scores_count = 0
    
    for key in task_keys:
        if key in file_metrics:
            stable = (file_metrics[key] - task_random_results[key])
            stable_scores[f"{key.split('/')[0]}-stable-score"] = stable
            global_stable_score_sum += stable
            valid_stable_scores_count += 1

    # Compute the global stable-score as the average of valid stable-scores
    global_stable_score = global_stable_score_sum / valid_stable_scores_count if valid_stable_scores_count > 0 else None

    # Update the file with the stable-score data
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data['stable-score'] = {
            "value": global_stable_score,
            **stable_scores
        }
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
