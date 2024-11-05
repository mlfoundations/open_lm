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
    "gqa_gqa-full/accuracy",
    "vqa-v2_vqa-v2-full/accuracy"
]

random_results = {
    "pope_pope-full/accuracy__POPE-final-Accuracy": 0.5,
    "ocid-ref_ocid-ref-full/accuracy__OCIDRef-All": 0,
    "refcoco_refcoco-full/accuracy__RefCOCO": 0,
    "text-vqa_text-vqa-full/accuracy__TextVQA-Pure": 0,
    # "vizwiz_vizwiz-full/accuracy__VizWiz-Overall": 0,
    "gqa_gqa-full/accuracy": 0.25,
    "vqa-v2_vqa-v2-full/accuracy": 0.25
}

# Gather all values for each metric
all_metrics = {key: [] for key in metrics_keys}

# Load the files and extract metrics
files_data = []
# means = {key: np.mean(all_metrics[key]) for key in metrics_keys}
# stds = {key: np.std(all_metrics[key]) for key in metrics_keys}
# stde = {key: stds[key] / np.sqrt(len(all_metrics[key])) for key in metrics_keys}

for folder in folders:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if not file_name.endswith('.json'):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            file_metrics = {}
            for key in metrics_keys:
                full_key = key
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
                    file_metrics[full_key] = value
                    all_metrics[full_key].append(value)
            files_data.append((file_path, file_metrics))

# Compute stable-scores for each file and update
for file_path, file_metrics in files_data:
    stable_scores = {}
    global_stable_score_sum = 0
    valid_stable_scores_count = 0
    
    for key in metrics_keys:
        if key in file_metrics:
            if key in ["gqa_gqa-full/accuracy", "vqa-v2_vqa-v2-full/accuracy", "vizwiz_vizwiz-full/accuracy__VizWiz-Overall"]:
                stable = file_metrics[key]/100 - random_results[key]
                assert file_metrics[key] <= 100
            else:
                stable = (file_metrics[key] - random_results[key])
                assert file_metrics[key] <= 1
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
