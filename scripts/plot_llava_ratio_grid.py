import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import argparse
import matplotlib.ticker as ticker


# Define a consistent symbol map for mult values
symbol_map = {
    '1p0': 'o',
    '2p0': '^',
    '4p0': 's'
}

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

def get_value_from_json(file_path, key_list):
    """Extracts the value from a JSON file given a list of nested keys."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        value = data
        norm = 1
        for i, key in enumerate(key_list):
            if key in ["gqa_gqa-full", "vqa-v2_vqa-v2-full", "vizwiz_vizwiz-full"]:
                norm = 100            
            if isinstance(value, dict):
                if key in value:
                    value = value[key]
                else:
                    return None, f"Key '{key}' not found at level {i}. Available keys: {', '.join(value.keys())}"
            else:
                return None, f"Cannot traverse further at key '{key}' (level {i}). Current value is not a dict: {type(value)}"
        
        return value / norm, None
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
        return match.group(1)
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



def make_plot(base_dir_1, base_dir_2, key_list_vision, key_list_text, ax, epoch_num=2, bucket=5, name_suffix="", title=True):
    """
    Generates a plot of pretraining image ratio vs. vision score and text score,
    with multiple lines for each mult value.
    """
    
    vision_eval = vision_evals[key_list_vision]
    text_eval = text_evals[key_list_text]
    
    key_list_vision = key_list_vision.split("/")
    key_list_text = key_list_text.split("/")
    
    # Extract data
    data_df, errors = extract_values(base_dir_1, base_dir_2, key_list_vision, key_list_text)

    if data_df.empty:
        print("No data found.")
        for error in errors:
            print(error)
        return

    # Extract mixing ratios and add to DataFrame
    ratios_list = [extract_mixing_ratios(file_name) for file_name in data_df['file_name']]
    ratios_df = pd.DataFrame(ratios_list)
    data_df = pd.concat([data_df.reset_index(drop=True), ratios_df.reset_index(drop=True)], axis=1)

    # Compute pretraining image ratio
    data_df['pretraining_image_ratio'] = 1 - data_df['dclm']

    # Handle missing values
    data_df.fillna({'datacompdr1b_caption': 0, 'llava': 0, 'dclm': 0}, inplace=True)

    # Filter data based on specified criteria
    filter_conditions = (
        (data_df['llava'] == 0) &            
        (data_df['epoch'] == epoch_num) &
        (data_df['file_name'].str.contains("fused-siglip")) &
        (~data_df["file_name"].str.contains("shutterstock")) &
        (~data_df["file_name"].str.contains("cc12m")) &
        (~data_df["file_name"].str.contains("datacomp1b_caption"))
    )
    
    if "1b" not in base_dir_1:
        filter_conditions &= (data_df['bucket'] == bucket)

    filtered_df = data_df[filter_conditions]

    if filtered_df.empty:
        print("No data matches the specified criteria.")
        return

    # Plotting
    ax2 = ax.twinx()
    
    # Define color maps for vision and text lines
    vision_colors = plt.cm.Greens(np.linspace(0.3, 0.8, len(filtered_df['mult'].unique())))
    text_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(filtered_df['mult'].unique())))
    
    for i, mult in enumerate(sorted(filtered_df['mult'].unique())):
        mult_df = filtered_df[filtered_df['mult'] == mult].sort_values(by='pretraining_image_ratio')
        
        # Vision score
        ax.plot(mult_df['pretraining_image_ratio'].clip(1e-3), mult_df['x1'], 
                color=vision_colors[i], marker='o', linestyle='-', 
                label=f'Vision {vision_eval} (mult={mult})', linewidth=2, markersize=6)
        
        # Text score
        ax2.plot(mult_df['pretraining_image_ratio'].clip(1e-3), mult_df['x2'], 
                 color=text_colors[i], marker='s', linestyle='--', 
                 label=f'Text {text_eval} (mult={mult})', linewidth=2, markersize=6)
    
    ax.set_xscale('log')
    ax2.set_xscale('log')

    # Custom function to format x-axis labels
    def custom_x_fmt(x, pos):
        if x == 1e-3:
            return '0%'
        return f'{x:.0%}'

    # Set custom x-axis formatter
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_x_fmt))
    
    # Make sure the x-axis has 5% in it
    ax.set_xticks(list(filtered_df['pretraining_image_ratio'].clip(1e-3).unique()))

    ax.set_xlabel('Pretraining Image Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Vision {vision_eval}', color='green', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f"Text {text_eval}", color='blue', fontsize=14, fontweight='bold')

    if title:
        ax.set_title(f'Vision: {vision_eval} vs Text: {text_eval}', fontsize=14, fontweight='bold')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    # Set y-axis colors
    ax.spines['left'].set_color('green')
    ax.tick_params(axis='y', colors='green')
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
   
def create_grid_plot(base_dir_1, base_dir_2, key_text, epoch_num=2, bucket=5, ratio=0.1, name_suffix="", output_dir=""):
    
    key_list_text = key_text
    
    num_plots = len(vision_evals)
    rows = (num_plots + 1) // 2  # Round up to nearest integer
    cols = 2
    
    fig = plt.figure(figsize=(12, 3 * rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Set style
    plt.style.use('seaborn-v0_8-pastel')
    
    for i, (key_list_vision, vision_name) in enumerate(vision_evals.items()):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        make_plot(base_dir_1, base_dir_2, key_list_vision, key_list_text, ax, epoch_num, bucket, ratio, name_suffix)
    
    plt.tight_layout()
    if output_dir != "" and output_dir[-1] != "/":
        output_dir += "/"
    plt.savefig(f'{output_dir}llava_image_ratio_vs_vision_text{text_evals[key_list_text]}_grid_epoch{epoch_num}_bucket{bucket}{name_suffix}_ratio{ratio}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_single_plot(base_dir_1, base_dir_2, key_text, key_vision, epoch_num=2, bucket=5, name_suffix="", output_dir=""):
    key_text = key_text
    key_vision = key_vision
    
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
    
    # Set style
    plt.style.use('seaborn-v0_8-pastel')
    
    make_plot(base_dir_1, base_dir_2, key_vision, key_text, ax, epoch_num, bucket, name_suffix, title=False)
    
    plt.tight_layout()
    if output_dir[-1] != "/":
        output_dir += "/"
    plt.savefig(f'{output_dir}pretraining_image_ratio_datacompdronly_vs_vision{vision_evals[key_vision]}_text{text_evals[key_text]}_epoch{epoch_num}_bucket{bucket}{name_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for vision and text scores.")
    parser.add_argument("--dir", type=str, default="1b", help="Source directory to plot.")
    parser.add_argument("--bucket", type=int, default=5, help="Bucket to plot.")
    parser.add_argument("--epoch_num", type=int, default=2, help="Epoch number to plot.")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to add to the plot filename.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for the plot.")
    parser.add_argument("--vision_key", type=str, default="all", help="Key for vision score.")
    parser.add_argument("--text_key", type=str, default="stable-score/value", help="Key for text score.")
    parser.add_argument("--ratio", type=str, default="0p1", help="Bucket to plot.")
    args = parser.parse_args()

    if not args.dir == "": 
        args.dir = f"_{args.dir}"   
    base_dir_1 = f"results/mbm_paper_eval{args.dir}/aggregated"
    base_dir_2 = f"results/mbm_paper_texteval{args.dir}"
    
    args.ratio = float(args.ratio.replace("p", "."))
    if args.vision_key == "all":
        create_grid_plot(base_dir_1, base_dir_2, args.text_key, args.epoch_num, args.bucket, args.ratio, args.name_suffix, args.output_dir)
    else:
        create_single_plot(base_dir_1, base_dir_2, args.text_key, args.vision_key, args.epoch_num, args.bucket, args.ratio, args.name_suffix, args.output_dir)
