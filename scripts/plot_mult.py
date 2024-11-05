import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import re
import pandas as pd


# Define a consistent symbol map for mult values
symbol_map = {
    '1p0': 'circle',
    '2p0': 'triangle-up',
    '4p0': 'square'
}

def get_value_from_json(file_path, key_list):
    """Extracts the value from a JSON file given a list of nested keys."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        value = data
        for i, key in enumerate(key_list):
            if isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    return None, f"Key '{key}' not found at level {i}. Available keys: {', '.join(value.keys())}"
            else:
                return None, f"Cannot traverse further at key '{key}' (level {i}). Current value is not a dict: {type(value)}"
        
        return value, None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}"
    except IOError as e:
        return None, f"IO Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
    
def extract_mixing_ratios(filename: str):
    datasets = ['llava', 'dclm', 'datacompdr1b_caption']
    ratios = {dataset: 0 for dataset in datasets}
    
    filename_shortened = filename.replace("llava-multimodal+", "")
    
    is_datacompdr1b = filename_shortened.startswith("datacompdr1b_caption")
    filename_shortened = filename_shortened.replace("datacompdr1b_caption", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
    is_llava = filename_shortened.startswith("llava")
    filename_shortened = filename_shortened.replace("llava", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
    is_dclm = filename_shortened.startswith("dclm")
    filename_shortened = filename_shortened.replace("dclm", "")
    while filename_shortened.startswith("_") or filename_shortened.startswith("-"):
        filename_shortened = filename_shortened[1:]
            
    if not filename_shortened.startswith("0"):
        if is_datacompdr1b:
            ratios['datacompdr1b_caption'] = 1.0
        if is_llava:
            ratios['llava'] = 1.0
        if is_dclm:
            ratios['dclm'] = 1.0
        return ratios
    
    # find next number in format 0p\d+
    next_number = re.search(r'0p\d+', filename_shortened)
    filename_shortened = filename_shortened[next_number.end():]
    if is_datacompdr1b:
        ratios['datacompdr1b_caption'] = float(next_number.group().replace('p', '.'))
    elif is_llava:
        is_llava = False
        ratios['llava'] = float(next_number.group().replace('p', '.'))
        
    next_number = re.search(r'0p\d+', filename_shortened)
    filename_shortened = filename_shortened[next_number.end():]
    if is_llava:
        ratios['llava'] = float(next_number.group().replace('p', '.')) 
    elif is_dclm:
        ratios['dclm'] = float(next_number.group().replace('p', '.')) 
        return ratios
    
    if is_dclm:
        next_number = re.search(r'0p\d+', filename_shortened)
        ratios['dclm'] = float(next_number.group().replace('p', '.'))
     
    return ratios


def extract_epoch_from_name(dirname):
    """Extracts the epoch number from the directory name (assumed format 'epochs=<number>')."""
    match = re.search(r'epochs=(\d+)', dirname)
    if match:
        epoch = int(match.group(1))
        return epoch
    return None


def extract_mult_from_name(dirname):
    """Extracts the mult value from the directory name (assumed format 'mult=<value>')."""
    match = re.search(r'mult=(\d+p\d+)', dirname)
    if match:
        mult = int(float(match.group(1).replace('p', '.')))
        return mult
    return None

def extract_bucket_from_name(dirname):
    """Extracts the bucket value from the directory name (assumed format '-<value>')."""
    if "bucket6" in dirname:
        return 6
    elif "lr=0p0009788" in dirname:
        return 5
    elif "lr=0p0034466" in dirname:
        return 4
    elif "lr=0p006499" in dirname:
        return 3
    elif "lr=0090834" in dirname:
        return 2
    elif "lr=0p003" in dirname:
        return 1
    return None

# Update the extract_values function to use the new return value of get_value_from_json
def extract_values(base_dir_1, base_dir_2, key_list_1, key_list_2):
    data = []
    errors = []

    try:
        files_in_base_dir_1 = os.listdir(base_dir_1)
    except Exception as e:
        errors.append(f"Error reading directory {base_dir_1}: {str(e)}")
        return pd.DataFrame(), errors

    for base_name in files_in_base_dir_1:
        try:
            file_path = os.path.join(base_dir_1, base_name)
            base_name = base_name.split(".")[0]
            
            x1_value, x_error = get_value_from_json(file_path, key_list_1)
            if x_error:
                errors.append(f"Error extracting x1_value from {file_path}: {x_error}")
                continue
            
            aggregated_file = os.path.join(base_dir_2, f'{base_name}.json')
            if not os.path.exists(aggregated_file):
                errors.append(f"Aggregated file not found: {aggregated_file}")
                continue
            
            x2_value, y_error = get_value_from_json(aggregated_file, key_list_2)
            if y_error:
                errors.append(f"Error extracting x2_value from {aggregated_file}: {y_error}")
                continue
            
            epoch = extract_epoch_from_name(base_name)
            mult = extract_mult_from_name(base_name)
            bucket = extract_bucket_from_name(base_name)
            mix = extract_mixing_ratios(base_name)
            
            if epoch is None or mult is None:
                errors.append(f"Unable to extract epoch or mult from {base_name}")
                continue
            
            data.append({
                'x1': x1_value,
                'x2': x2_value,
                "mix": mix,
                'epoch': epoch,
                'mult': mult,
                'file_name': base_name,
                "bucket": bucket
            })
        except Exception as e:
            errors.append(f"Error processing {base_name}: {str(e)}")

    return pd.DataFrame(data), errors

def make_plot(base_dir_1, base_dir_2, key_list_vision, key_list_text, name_suffix=""):
    """
    Generates a plot of pretraining image ratio vs. vision z-score and text z-score.

    Args:
        base_dir_1 (str): Directory containing JSON files for vision z-scores.
        base_dir_2 (str): Directory containing JSON files for text z-scores.
        key_list_vision_z_score (list): List of keys to extract vision z-score from JSON.
        key_list_text_z_score (list): List of keys to extract text z-score from JSON.
    """
    
    text_evals ={
        "results/agieval_sat_en/acc,none": "agieval_sat_en",
        "results/arc_easy/acc,none": "arc_easy",
        "results/bigbench_conceptual_combinations_multiple_choice/acc,none": "bigbench_conceptual_combinations",
        "results/bigbench_cs_algorithms_multiple_choice/acc,none": "bigbench_cs_algorithms",
        "results/boolq/acc,none": "boolq",
        "results/copa/acc,none": "copa",
        "results/hellaswag/acc,none": "hellaswag",
        "results/mathqa/acc,none": "mathqa",
        "results/piqa/acc,none": "piqa",
        "results/pubmedqa/acc,none": "pubmedqa",
        "z-score/value": "z-score",
        "stable-score/value": "stable-score"
    }
    
    vision_evals = {
        "pope_pope-full/accuracy__POPE-final-Accuracy": "pope",
        "ocid-ref_ocid-ref-full/accuracy__OCIDRef-All": "OCIDRe",
        "refcoco_refcoco-full/accuracy__RefCOCO": "RefCOCO",
        "text-vqa_text-vqa-full/accuracy__TextVQA-Pure": "TextVQA",
        "vizwiz_vizwiz-full/accuracy__VizWiz-Overall": "VizWiz",
        "gqa_gqa-full/accuracy": "gqa",
        "vqa-v2_vqa-v2-full/accuracy": "vqa-v2",
        "z-score/value": "z-score",
        "stable-score/value": "stable-score"
    }
    
    text_eval = text_evals[key_list_text]
    vision_eval = vision_evals[key_list_vision]
    
    key_list_text = key_list_text.split("/")
    key_list_vision = key_list_vision.split("/")
    
    # Extract data
    data_df, errors = extract_values(base_dir_1, base_dir_2, key_list_vision, key_list_text)

    if data_df.empty:
        print("No data found.")
        for error in errors:
            print(error)
        return

    # Extract mixing ratios
    ratios_list = []
    for index, row in data_df.iterrows():
        file_name = row['file_name']
        ratios = extract_mixing_ratios(file_name)
        ratios_list.append(ratios)

    # Add ratios to DataFrame
    ratios_df = pd.DataFrame(ratios_list)
    data_df = pd.concat([data_df.reset_index(drop=True), ratios_df.reset_index(drop=True)], axis=1)

    # Compute pretraining image ratio
    data_df['pretraining_image_ratio'] = 1 - data_df['dclm']

    # Handle missing values
    data_df.fillna({'datacompdr1b_caption': 0, 'llava': 0, 'dclm': 0}, inplace=True)

    # Extract checkpoint_portion from 'mult'
    data_df['checkpoint_portion'] = data_df['mult']

    # Filter data based on specified criteria
    ratio = 0.1
    epoch = 4
    if "1b" in base_dir_1:
        raise ValueError("This script is not intended for 1b data.")
    else:
        filtered_df = data_df[
            (data_df['datacompdr1b_caption']/data_df['llava'] == 4) &
            (abs(data_df['pretraining_image_ratio']-ratio)<0.00001) &
            (~data_df["file_name"].str.contains("shutterstock")) &
            (~data_df["file_name"].str.contains("cc12m")) &
            (~data_df["file_name"].str.contains("datacomp1b_caption")) &
            (data_df['epoch'] == 4)
        ]

    if filtered_df.empty:
        print("No data matches the specified criteria.")
        return

    # Plotting
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    sorted_df = filtered_df.sort_values(by='mult')
    ax1.plot(sorted_df['mult'], sorted_df['x1'], 'go-', label=f'Vision {vision_eval}')
    ax2.plot(sorted_df['mult'], sorted_df['x2'], 'bo-', label=f'Text {text_eval}')

    ax1.set_xlabel('Pretraining cc')
    ax1.set_ylabel(f'Vision {vision_eval}', color='g')
    ax2.set_ylabel(f"Text {text_eval}", color='b')

    # Combine legends
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1+h2, l1+l2, loc='best')
    
    for filename in filtered_df['file_name']:
        print(filename.replace("llava-multimodal+", ""))

    plt.title('Pretraining CC vs Vision and Text Scores')
    plt.savefig(f'pretrainingcc_vs_vision_{vision_eval}_text_{text_eval}_ratio{ratio}_epoch{epoch}{name_suffix}.png')
    plt.show()

if __name__ == "__main__":
    bucket = "_5"
    base_dir_1 = f"results/mbm_paper_eval{bucket}/aggregated"
    base_dir_2 = f"results/mbm_paper_texteval{bucket}"
    # key_list_vision_z_score = 'vqa-v2_vqa-v2-full/accuracy'
    # key_list_text_z_score = "results/arc_easy/acc,none"
    # key_list_vision_z_score = 'z-score/value'
    # key_list_text_z_score = 'z-score/value'
    key_list_vision_z_score = 'stable-score/value'
    key_list_text_z_score = 'stable-score/value'

    make_plot(base_dir_1, base_dir_2, key_list_vision_z_score, key_list_text_z_score, name_suffix=bucket)