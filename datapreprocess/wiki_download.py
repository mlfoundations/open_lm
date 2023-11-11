""" 
Quick wikipedia download script from huggingface for quickstart purposes.
Just downloads the 20220301 english wikipedia from huggingface and 
does no extra preprocessing.

"""

import argparse
from datasets import load_dataset  # huggingface
import os


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = load_dataset("wikipedia", "20220301.en")

    for split, dataset in data.items():
        print("Processing split: %s" % data)
        output_file = os.path.join(output_dir, "wiki_en_20220301_%s.jsonl" % (split))
        dataset.to_json(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store the wikipedia .jsonl file",
    )

    args = parser.parse_args()
    main(args.output_dir)
