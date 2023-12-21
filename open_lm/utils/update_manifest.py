"""Convert manifests to the new format.

This file converts existing manifest files to the new format (changing the "num_chunks" field to "num_sequences").
"""

import argparse
import re
import shutil
import simdjson
import sys
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
        "--manifest-path",
        type=path_or_cloudpath,
        required=True,
        help="Manifest file to update.",
    )
    parser.add_argument("--tmp-dir", type=str, default="/tmp", help="Temporary directory.")
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)

    tmp_dir = Path(args.tmp_dir)

    temp_manifest_filename = tmp_dir / args.manifest_path.name

    with args.manifest_path.open("rb") as f:
        data = f.read()

    jsons = [simdjson.loads(o) for o in data.decode("utf-8").split("\n")[:-1]]

    with temp_manifest_filename.open("w") as f:
        for data in tqdm(jsons):
            new_data = {}
            new_data["shard"] = data["shard"]
            new_data["num_sequences"] = data["num_chunks"]
            f.write(simdjson.dumps(new_data))
            f.write("\n")

    if isinstance(args.manifest_path, CloudPath):
        args.manifest_path.upload_from(temp_manifest_filename)
    else:
        shutil.copy(temp_manifest_filename, args.manifest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
