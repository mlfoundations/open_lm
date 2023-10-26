import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch
from tqdm import tqdm
import sys
import numpy as np

def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude', '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False
        
    logging.info(f"Successfully synced with S3 bucket")
    return True

def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        logging.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            logging.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            logging.info(f'Error during remote sync for {k}: {e}')
            return False

    return True

def remote_sync(local_dir, remote_dir, protocol):
    logging.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error('Remote protocol not known')
        return False

def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)

def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True

import json
import fsspec

def get_metadata_file(path):
    of = fsspec.open(path, 'rb')
    with of as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode('utf-8').split('\n')[:-1]]
    return out

def get_shards_for_chunk(num_samples, chunk, path):
    """ Function to get a chunk of shards to train on.

    Chunks are groups of shards with samples roughly equal to the number of samples
    that will be seen during training. This function uses the dataset manifest
    to split the shards into chunks, and assign shards to each chunk.
    """
    metadata = get_metadata_file(path)
    shard_list = []
    curr_shard_list = []
    chunk_count_list = []
    curr_chunk_count = 0
    for m in metadata:
        curr_chunk_count += m['num_chunks']
        curr_shard_list.append(m['shard'])
        if curr_chunk_count >= num_samples:
            shard_list.append(curr_shard_list)
            chunk_count_list.append(curr_chunk_count)
            curr_shard_list = []
            curr_chunk_count = 0
    
    # Append remaining shards
    if len(curr_shard_list) > 0:
        shard_list.append(curr_shard_list)
        chunk_count_list.append(curr_chunk_count)

    return shard_list[chunk % len(shard_list)], chunk_count_list[chunk % len(chunk_count_list)]


def enough_shards(shard_lists, min_shards_needed):
    for sl in shard_lists:
        if len(sl) < min_shards_needed:
            return False
    return True


def enough_samples(num_samples_per_source, needed_samples_per_source):
    for i, number in enumerate(num_samples_per_source):
        if number < needed_samples_per_source[i]:
            return False
    return True


def source_exhausted(paths, shard_list_per_source):
    for i, source in enumerate(paths):
        data = get_metadata_file(source)
        if len(data) < len(shard_list_per_source[i]):
            return True
    return False


def get_string_for_epoch(num_samples, starting_chunk, paths, weights, min_shards_needed):
    if weights is None:
        weights = [1.0 / len(paths) for _ in range(len(paths))]
    needed_samples_per_source = [int(np.ceil(weights[i] * num_samples / sum(weights))) for i in range(len(weights))]
    shard_strings_per_source = []
    next_chunk = starting_chunk
    shard_list_per_source = [[] for _ in range(len(paths))]
    num_samples_per_source = [0 for _ in range(len(paths))]
    while not enough_shards(shard_list_per_source, min_shards_needed) or not enough_samples(num_samples_per_source, needed_samples_per_source):
        for i, source_path in enumerate(paths):
            shard_list_source, num_samples_source = get_shards_for_chunk(needed_samples_per_source[i], next_chunk, source_path)
            shard_list_per_source[i].extend(shard_list_source)
            num_samples_per_source[i] += num_samples_source 
        next_chunk += 1
        if source_exhausted(paths, shard_list_per_source):
            print(shard_list_per_source)
            raise ValueError(
                "Number of shards requested is more than the number of shards available."
                "Consider lowering the number of workers and / or the number of GPUs."
            )

    for i, source_path in enumerate(paths):
        shard_list_source = shard_list_per_source[i]
        num_samples_source = num_samples_per_source[i]
        shard_root_source = '/'.join(source_path.split('/')[:-1]) + '/shard_'
        shard_string_source = shard_root_source + '{' + ",".join(shard_list_source) + '}.tar'
        if source_path.startswith('s3'):
            shard_string_source = f'pipe:aws s3 cp {shard_string_source} -'
        shard_strings_per_source.append(shard_string_source)

    return shard_strings_per_source, num_samples_per_source, next_chunk


if __name__ == '__main__':
    print(sys.argv)
    #path = 's3://s-laion/open_lm_shuffle/rpj_shuffled_mosaic_upsampled_tiktoken_100k_shards_try2/shard_metadata.jsonl'
    #out = get_string_for_epoch(10000000000 // 2048, 10, path)

    
#10000000000