import argparse
import collections
import hashlib
import io
import json
import os
import resource
import tarfile
import time
import traceback
from enum import Enum
from io import BytesIO
from typing import List
from loguru import logger

import boto3
import fsspec
import numpy as np
import psutil
import simdjson
import webdataset as wds
from braceexpand import braceexpand
from transformers import GPTNeoXTokenizerFast

import ray
from ray._private.internal_api import memory_summary
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask


class SpecialTokens(Enum):
    END_OF_TEXT = -1
    PAD = -2
    END_OF_DOCUMENT = -3


class PadType(Enum):
    CIRCULAR = 0
    PAD_TOKEN = 1


def jsonl_to_tokens(jsonl, tokenizer, keep_text=False, discard_meta=False):
    all_rows = {"tokens": []}
    jsonl_text = jsonl["text"]
    tokens = tokenizer(jsonl_text) + [SpecialTokens.END_OF_TEXT.value]
    out_dict = {}
    if not discard_meta:
        out_dict["metadata"] = jsonl.copy()
        del out_dict["metadata"]["text"]
    if keep_text:
        out_dict["text"] = jsonl["text"]
    out_dict["tokens"] = tokens
    return out_dict


def repartition_and_pad(input_list, seq_len, pad_type, meta_lists={}):
    """Repartition the input list into sublists of size seq_len.
    If the last sublist isn't seq_len long, pad it with values from the beginning."""

    # Flatten the list
    flat_list = [item for sublist in input_list for item in sublist]
    # Calculate the number of elements needed to pad the last sublist
    pad_len = (
        seq_len - (len(flat_list) % seq_len) if len(flat_list) % seq_len != 0 else 0
    )

    if pad_type == PadType.CIRCULAR:
        # Pad the flattened list with values from its beginning
        flat_list.extend(flat_list[:pad_len])
    elif pad_type == PadType.PAD_TOKEN:
        flat_list.extend([SpecialTokens.PAD.value] * pad_len)
    # Repartition the flattened list into sublists of size seq_len
    repartioned_lists = [
        flat_list[i : i + seq_len] for i in range(0, len(flat_list), seq_len)
    ]
    repartioned_meta_lists = collections.defaultdict(list)
    for k, _meta_list in meta_lists.items():
        idx = 0
        for sublist in repartioned_lists:
            current_meta_list = []
            sublist_len = len(sublist)
            while sublist_len > 0 and idx <= len(_meta_list):
                current_meta_list.append(_meta_list[idx])
                sublist_len -= len(input_list[idx])
                if pad_type == PadType.CIRCULAR:
                    idx = (idx + 1) % len(input_list)
                else:
                    idx += 1
            repartioned_meta_lists[k].append(current_meta_list)

    return repartioned_lists, repartioned_meta_lists


def cut_to_context(jsonl_batch, seqlen=1024, pad_type=PadType.CIRCULAR):
    tokens_list = jsonl_batch.pop("tokens")
    repartioned_lists, repartioned_meta_lists = repartition_and_pad(
        tokens_list, seq_len=seqlen, pad_type=pad_type, meta_lists=jsonl_batch
    )
    out = {"tokens": repartioned_lists}
    for k, v in repartioned_meta_lists.items():
        out[k] = v
    return out


def map_write_wds(batch, batch_size, folder, total_count=None):
    # Calculate tar index based on the first id
    first_id = batch["id"][0]
    start_idx = first_id // batch_size

    # Determine the number of leading zeros dynamically based on total_count
    if total_count:
        digits = len(str(total_count // batch_size))
    else:
        digits = 5  # default to 5 if total_count is not provided

    # Format tar index with the determined number of leading zeros
    tar_index = f"{start_idx:0{digits}}"

    # Create tar file name
    tar_name = f"{tar_index}.tar"

    # Write the batch to a tarball using webdataset's TarWriter
    bio = io.BytesIO()
    with wds.TarWriter(bio) as sink:
        for i in range(len(batch["id"])):
            tokens = [int(x) for x in batch["tokens"][i]]
            uid = hashlib.md5(simdjson.dumps(tokens).encode()).hexdigest()
            sample = {"__key__": uid, "json": tokens}
            sink.write(sample)

    bio.seek(0)
    write_to_location(folder, tar_name, bio)
    return batch


def write_to_location(folder, tar_name, bio):
    path = f"{folder}/{tar_name}"

    # Check if the path is an S3 path
    if path.startswith("s3://"):
        s3 = boto3.client("s3")

        # Properly extract bucket and key from the S3 path
        s3_path_parts = path[5:].split("/")
        bucket = s3_path_parts[0]
        key = "/".join(s3_path_parts[1:])

        try:
            s3.put_object(Bucket=bucket, Key=key, Body=bio.getvalue())
        except Exception as e:
            assert False, f"bucket is {bucket} key is {key} and {e}"

    else:
        # Create directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            with open(path, "wb") as f:
                f.write(bio.getvalue())
        except Exception as e:
            assert False, f"error is {path} and {e}"


def load_tokenizer(tokenizer):
    if tokenizer == "EleutherAI/gpt-neox-20b":
        enc = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        return lambda x: enc(x).input_ids
    else:
        raise ValueError(f"Unknown Tokenizer: {tokenizer}")
    
def glob_files(path, suffix=".jsonl"):
    """
    Glob files based on a given path and suffix. 
    Supports both local and S3 paths.

    :param path: path to glob. Can be local or S3 (e.g., s3://bucket-name/path/)
    :param suffix: suffix of files to match. Defaults to ".jsonl"
    :return: list of file paths matching the pattern
    """
    if path.startswith("s3://"):
        # Use boto3 for S3 paths
        s3 = boto3.client('s3')
        bucket_name, prefix = path[5:].split('/', 1)

        # Ensure the prefix ends with a '/'
        if not prefix.endswith('/'):
            prefix += '/'

        # List the objects in the bucket with the given prefix
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        all_files = [f"s3://{bucket_name}/{obj['Key']}" for obj in objects.get('Contents', [])]
        
        # Filter out the files based on the suffix
        matching_files = [f for f in all_files if f.endswith(suffix)]

    else:
        # Use glob for local paths
        search_pattern = f"{path.rstrip('/')}/*{suffix}"
        matching_files = glob.glob(search_pattern)

    return matching_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="input path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output",
        help="output path",
        type=str,
        required=True
        # e.g s3://dcnlp-data/rpj_tokenized_upsampled_eleutherai_deduplicated/
    )
    parser.add_argument("--keep_text", action="store_true")
    parser.add_argument("--discard_metadata", action="store_true")
    parser.add_argument(
        "--no_shuffle", help="do not dedup + random shuffle", action="store_true"
    )
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--pad_type", type=str, default="circular")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--initial_batch_size", type=int, default=128)
    parser.add_argument("--wds_chunk_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--ray_address", type=str, default=None)
    parser.add_argument("--ray_spill_location", type=str, default="s3://dcnlp-hub/ray_spill")

    args = parser.parse_args()
    # configure remote spilling

    if args.ray_address is None:
        ray.init()
    else:
        ray.init(args.ray_address)
    # TODO  support multiple inputs
    input_paths = glob_files(args.input, suffix=".jsonl")
    if args.subset is not None:
        input_paths = input_paths[:args.subset]
        
    output_path = args.output
    seqlen = args.seqlen
    batch_size = args.initial_batch_size
    wds_chunk_size = args.wds_chunk_size
    if args.pad_type == "circular":
        pad_type = PadType.CIRCULAR
    elif args.pad_type == "pad_token":
        pad_type = PadType.PAD_TOKEN
    else:
        raise ValueError(f"Unknown pad_type = {args.pad_type}")

    ctx = DataContext.get_current()
    ctx.use_push_based_shuffle = True
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    start_time = time.time()
    tokenizer = load_tokenizer(args.tokenizer)
    ds = ray.data.read_json(input_paths)
    # convert to tokens
    ds = ds.map(
        lambda x: jsonl_to_tokens(
            x,
            tokenizer=tokenizer,
            keep_text=args.keep_text,
            discard_meta=args.discard_metadata,
        )
    )
    ds = ds.map_batches(
        cut_to_context,
        batch_size=batch_size,
        fn_kwargs={"pad_type": pad_type, "seqlen": seqlen},
    )
    count = ds.count()
    tokenize_end_time = time.time()
    print(f"Total num contexts = {count}")
    print("Tokenize Finished in", tokenize_end_time - start_time)
    #exit()
    ds = ds.random_shuffle(seed=args.seed)
    num_rows = ds.count()
    ds_indices = ray.data.range(num_rows).map_batches(
        lambda x: x, batch_format="pandas"
    )
    ds = ds_indices.zip(ds)
    ds = ds.map_batches(
        map_write_wds,
        batch_size=wds_chunk_size,
        fn_kwargs={
            "batch_size": wds_chunk_size,
            "folder": args.output.strip("/"),
            "total_count": num_rows,
        },
    ).count()
    end_time = time.time()
    duration = end_time - start_time
    print("Tokenize + Shuffle script Finished in", duration)
    print("==== Driver memory summary ====")
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3)
    print(f"max: {maxrss / 1e9}/GB")
    process = psutil.Process(os.getpid())
    rss = int(process.memory_info().rss)
    print(f"rss: {rss / 1e9}/GB")
    try:
        print(memory_summary(stats_only=True))
    except Exception:
        print("Failed to retrieve memory summary")
        print(traceback.format_exc())
    print("")
