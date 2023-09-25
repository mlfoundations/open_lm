import argparse
import re
import os
import simdjson
import sys
import subprocess
import tempfile
import multiprocessing as mp
import shutil
from pathlib import Path
from cloudpathlib import CloudPath
from tqdm import tqdm


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=path_or_cloudpath,
        required=True,
        help="Directory containing a dataset in webdataset format.",
    )
    parser.add_argument(
        "--manifest-filename",
        type=str,
        default="manifest.jsonl",
        help="Filename for the manifest that will be stored in the webdataset directory.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers.")
    args = parser.parse_args(args)
    return args

'''
def count_samples(shard_path):
    count = int(subprocess.check_output(f"tar tf {shard_path} | wc -l", shell=True))
    return count
'''

def count_samples(shard_path):
    # Check if the shard_path is a CloudPath
    if isinstance(shard_path, CloudPath):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            # Download the shard from S3 to the temporary file
            shard_path.download_to(temp_file_path)
            # Run the tar command on the local temporary file
            count = int(subprocess.check_output(f"tar tf {temp_file_path} | wc -l", shell=True))
            # Remove the temporary file
            temp_file_path.unlink()
    else:
        # If shard_path is not a CloudPath, run the tar command directly on it
        count = int(subprocess.check_output(f"tar tf {shard_path} | wc -l", shell=True))
    return count

def worker_fn(input_data):
    basename, data_dir = input_data
    shard_path = data_dir / basename
    return (basename, {
        "shard": basename.split("-")[1].split(".")[0],
        "num_chunks": count_samples(shard_path),
    })


def main(args):
    args = parse_args(args)

    shards = sorted([x for x in args.data_dir.iterdir() if x.name.endswith(".tar")])
    input_data = [(shard.name, args.data_dir) for shard in shards]

    print(f"Shards to process: {len(shards)}")
    print("Creating pool.")
    with mp.Pool(args.num_workers) as pool:
        data = []
        for worker_data in tqdm(pool.imap_unordered(worker_fn, input_data)):
            data.append(worker_data)
            

    data = sorted(data)
    data = [item[1] for item in data]
    manifest_path = args.data_dir / args.manifest_filename
    with manifest_path.open("w") as fp:
        for item in data:
            simdjson.dump(item, fp)
            fp.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
