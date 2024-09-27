import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def get_value_from_json(file_path, key_list):
    """Extracts the value from a JSON file given a list of nested keys."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        value = data
        for key in key_list:
            if isinstance(value, dict):
                value = value.get(key, value)
                if value is None:
                    return None
            else:
                print(f"s3://tri-ml-datasets/openlm/mbm_paper_eval_6/results/aggregated/{file_path.split('aggregated/')[1]}")
                return None
    if isinstance(value, dict):
        print(f"Dict has not been fully traversed: {value.keys()}")
        return None
    return value

def extract_epoch_from_name(dirname):
    """Extracts the epoch number from the directory name (assumed format 'epochs=<number>')."""
    match = re.search(r'epochs=(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None


def extract_mult_from_name(dirname):
    """Extracts the mult value from the directory name (assumed format 'mult=<value>')."""
    match = re.search(r'mult=(\d+p\d+)', dirname)
    if match:
        return match.group(1)
    return None

def extract_ratio_from_dirname(dirname):
    """Extracts the ratio value from the directory name values are in the form 0p05
        possible formats:
            - "-<value1>_<value2>"
            - "-<value1>-<value2>-<value3>"
    """
    matches = re.findall(r'(0+p\d+)', dirname)
    
    if matches:
        # Return the last match
        return matches[-1]
    
    return None

def extract_values(base_dir_1, base_dir_2, key_list_1, key_list_2, name_exclude=None, name_include=None):
    x_vals = []
    y_vals = []
    epoch_vals = []
    mult_vals = []

    files_in_base_dir_1 = os.listdir(base_dir_1)

    # Iterate over directories that contain results
    for base_name  in files_in_base_dir_1:
        file_path = os.path.join(base_dir_1, base_name)
        base_name = base_name.split(".")[0]

        # Apply file name filters
        if name_exclude:
            if any(filter_str in file_path for filter_str in name_exclude):
                continue  # Skip if any of the filters match the file_path name
        if name_include:
            if not all(filter_str in file_path for filter_str in name_include):
                continue
        
        # Get the value from the first file using the provided key_list_1
        x_value = get_value_from_json(file_path, key_list_1)
        if x_value is None:
            continue

        
        # Path to the corresponding json file in base_dir_2
        aggregated_file = os.path.join(base_dir_2, f'{base_name}.json')
        if not os.path.exists(aggregated_file):
            continue
        
        # Get the value from the second file using the provided key_list_2
        y_value = get_value_from_json(aggregated_file, key_list_2)
        if y_value is None:
            continue
        
        # Extract the epoch number from the directory name
        epoch = extract_epoch_from_name(base_name)
        if epoch is None:
            continue
        
        # Extract the mult value from the directory name
        mult = extract_mult_from_name(base_name)
        if mult is None:
            continue
        
        # Append the values to the lists
        x_vals.append(x_value)
        y_vals.append(y_value)
        epoch_vals.append(epoch)
        mult_vals.append(mult)

    return x_vals, y_vals, epoch_vals, mult_vals


def plot_values(x_vals, y_vals, epoch_vals, mult_vals, x_label, y_label, plot_name):
    plt.figure(figsize=(8, 6))

    # Normalize the epoch values for color mapping
    norm = plt.Normalize(min(epoch_vals), max(epoch_vals))
    colors = cm.viridis(norm(epoch_vals))

    # Define different shapes for different mult values
    marker_shapes = {
        "1p0": "o",  # Circle
        "2p0": "s",  # Square
        "4p0": "^"   # Triangle
    }

    # Scatter plot with different shapes for different mult values
    for mult_value in set(mult_vals):
        # Get indices of points with the current mult value
        indices = [i for i, mult in enumerate(mult_vals) if mult == mult_value]
        
        # Extract corresponding values
        x = [x_vals[i] for i in indices]
        y = [y_vals[i] for i in indices]
        c = [colors[i] for i in indices]
        
        # Plot points with the corresponding shape
        plt.scatter(x, y, c=c, marker=marker_shapes.get(mult_value, "o"), s=100, edgecolor='k', label=f"mult={mult_value}")

    # Create a color bar and set its ticks to the actual epoch values
    cbar = plt.colorbar()
    cbar.set_label('Epoch')

    # Set the ticks of the colorbar to the epoch values (or evenly spaced)
    cbar.set_ticks(np.linspace(norm(epoch_vals).min(), norm(epoch_vals).max(), len(set(epoch_vals))))
    cbar.set_ticklabels(sorted(set(epoch_vals)))

    # Generate the title based on key_list_1's second entry
    title = f"{key_list_1[1]} acc_norm vs VQA accuracy (Colored by Epoch)"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(loc='lower left')  # Add legend for mult shapes at the bottom left
    plt.savefig(plot_name)
    plt.show()

def main(base_directory_1, base_directory_2, key_list_1, key_list_2, name_exclude=None, name_include=None):
    # Get the directories that contain results files

    # Extract the values for plotting
    x_values, y_values, epoch_values, mult_values = extract_values(base_directory_1, base_directory_2, key_list_1, key_list_2, name_exclude, name_include)

    name_inc = "_inc_[" + "_".join(name_include) + "]" if name_include else ""
    name_exc = "_exc_[" + "_".join(name_exclude) + "]"  if name_exclude else ""
    bucket = base_directory_1.split("/")[-2].split("_")[-1]
    os.makedirs(f"results/mbm_plot_{bucket}/", exist_ok=True)
    plot_name = f'results/mbm_plot_{bucket}/{key_list_1[1]}_vs_{key_list_2[0]}{name_exc}{name_inc}.png'

    x_label = f"{key_list_1[1]} acc_norm"
    y_label = f"{key_list_2[0]} accuracy"

    # Plot the results with color mapping for epoch numbers
    plot_values(x_values, y_values, epoch_values, mult_values, x_label=x_label, y_label=y_label, plot_name=plot_name)

# Example usage:
if __name__ == "__main__":
    bucket = "1b"
    base_directory_1 = f'results/mbm_paper_texteval_{bucket}/'
    base_directory_2 = f'results/mbm_paper_eval_{bucket}/aggregated/'
    key_list_1 = ["results", "hellaswag", "acc_norm,none"]
    key_list_2 = ["vqa-v2_vqa-v2-full", "accuracy"]

    main(base_directory_1, base_directory_2, key_list_1, key_list_2, name_exclude=[], name_include=[])
