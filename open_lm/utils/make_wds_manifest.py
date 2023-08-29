import argparse
import re
import simdjson
import sys
import multiprocessing as mp
from pathlib import Path
from cloudpathlib import CloudPath
import webdataset as wds
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
        "--tmp-dir",
        type=str,
        default="./",
        help="Temporary directory for interfacing with s3.",
    )
    parser.add_argument(
        "--manifest-filename",
        type=str,
        default="manifest.json",
        help="Filename for the manifest that will be stored in the webdataset directory.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers.")
    args = parser.parse_args(args)
    return args


def count_samples(shard_path):
    shard_ds = wds.WebDataset(str(shard_path))
    count = 0
    for _ in iter(shard_ds):
        count += 1
    return count


def worker_fn(input_data):
    basename, data_dir = input_data
    shard_path = data_dir / basename
    return {basename: count_samples(shard_path)}


def main(args):
    args = parse_args(args)

    shards = args.data_dir.iterdir()
    input_data = [(shard.name, args.data_dir) for shard in shards]

    with mp.Pool(args.num_workers) as pool:
        data = {}
        for worker_data in tqdm(pool.imap_unordered(worker_fn, input_data)):
            data.update(worker_data)

    manifest_path = args.data_dir / args.manifest_filename
    with manifest_path.open("w") as fp:
        simdjson.dump(data, fp)


if __name__ == "__main__":
    main(sys.argv[1:])
