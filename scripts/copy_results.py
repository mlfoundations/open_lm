import os
import shutil
import argparse

def check_directories(base_path, output_path):
    dirs_with_results = []
    dirs_without_results = []

    # List directories in the base path
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Function to check for files starting with "results_" recursively
    def contains_results_file(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("results_"):
                    return os.path.join(root, file)
        return None

    # Iterate over each directory
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        result_file = contains_results_file(dir_path)
        if result_file:
            dirs_with_results.append(directory)
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Copy the result file to the destination
            dest_path = os.path.join(output_path, f"{directory}.json")
            shutil.copyfile(result_file, dest_path)

        else:
            dirs_without_results.append(directory)

    return dirs_with_results, dirs_without_results

def main(bucket="6"):
    # Define the base directory and output directory
    base_directory = f'lm_eval_logs/mbm_paper_finetune{bucket}/'
    output_directory = f'results/mbm_paper_texteval{bucket}/'

    # Get the directories that contain or don't contain "results_"
    with_results, without_results = check_directories(base_directory, output_directory)

    # Output the results
    # print("Directories with 'results_':", with_results)
    print(f"Directories without 'results_' {len(without_results)}/{len(with_results)+len(without_results)}:")
    for dir in without_results:
        print(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy results files from directories with 'results_' to a new directory.")
    parser.add_argument("--bucket", type=str, default="6", help="Bucket number for the logs.")
    args = parser.parse_args()
    if args.bucket != "":
        args.bucket = f"_{args.bucket}"
    main(args.bucket)


