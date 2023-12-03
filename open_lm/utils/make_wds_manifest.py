import argparse
import re
import simdjson
import sys
import subprocess
import multiprocessing as mp
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
    parser.add_argument("--tmp-dir", type=str, default=None, help="Temporary directory.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers.")
    args = parser.parse_args(args)
    return args


def count_samples(shard_path, tmp_dir):
    if isinstance(shard_path, CloudPath):
        temp_shard_path = Path(tmp_dir) / shard_path.name
        shard_path.download_to(temp_shard_path)
    else:
        temp_shard_path = shard_path

    count = int(subprocess.check_output(f"tar tf {temp_shard_path} | wc -l", shell=True))

    if isinstance(shard_path, CloudPath):
        temp_shard_path.unlink()

    return count


def worker_fn(input_data):
    basename, data_dir, tmp_dir = input_data
    shard_path = data_dir / basename
    return (
        basename,
        {
            "shard": basename.split(".")[0],
            "num_sequences": count_samples(shard_path, tmp_dir),
        },
    )


def main(args):
    args = parse_args(args)

    shards = sorted([x for x in args.data_dir.iterdir() if x.name.endswith(".tar")])
    input_data = [(shard.name, args.data_dir, args.tmp_dir) for shard in shards]

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
