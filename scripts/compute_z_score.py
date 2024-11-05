import os
import json
import numpy as np

# Define the paths where the aggregated folders are located
# folders = [
#     "results/mbm_paper_eval/aggregated",
#     "results/mbm_paper_eval_2/aggregated",
#     "results/mbm_paper_eval_3/aggregated",
#     "results/mbm_paper_eval_4/aggregated",
#     "results/mbm_paper_eval_5/aggregated",
#     "results/mbm_paper_eval_6/aggregated",
# ]

folders = [
    "results/mbm_paper_eval_1b/aggregated",
]


# Metrics to extract
metrics_keys = [
    "pope_pope-full/accuracy__POPE-final-Accuracy",
    "ocid-ref_ocid-ref-full/accuracy__OCIDRef-All",
    "refcoco_refcoco-full/accuracy__RefCOCO",
    "text-vqa_text-vqa-full/accuracy__TextVQA-Pure",
    # "vizwiz_vizwiz-full/accuracy__VizWiz-Overall",
    "gqa_gqa-full/accuracy",
    "vqa-v2_vqa-v2-full/accuracy"
]

# Gather all values for each metric
all_metrics = {key: [] for key in metrics_keys}

# Load the files and extract metrics
files_data = []

for folder in folders:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if not file_name.endswith('.json'):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            file_metrics = {}
            for key in metrics_keys:
                keys = key.split('/')
                value = data
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k, None)
                    else:
                        value = None
                    if value is None:
                        break
                if value is not None:
                    file_metrics[key] = value
                    all_metrics[key].append(value)
            files_data.append((file_path, file_metrics))

# Compute means and standard deviations for each metric
means = {key: np.mean(all_metrics[key]) for key in metrics_keys}
stds = {key: np.std(all_metrics[key]) for key in metrics_keys}

# Compute z-scores for each file and update
for file_path, file_metrics in files_data:
    z_scores = {}
    global_z_score_sum = 0
    valid_z_scores_count = 0
    
    for key in metrics_keys:
        if key in file_metrics:
            z = (file_metrics[key] - means[key]) / stds[key]
            z_scores[f"{key.split('/')[-1]}-z-score"] = z
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
