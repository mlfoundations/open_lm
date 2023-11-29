import copy
import io
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import copy
from itertools import cycle, islice

import fsspec
import numpy as np
import torch

from typing import List, Optional
from tqdm import tqdm

from .distributed import is_master


def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir, remote_dir, "--exclude", "*epoch_latest.pt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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
        if "epoch_latest.pt" in k:
            continue

        logging.info(f"Attempting to sync {k}")
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f"Skipping remote sync for {k}.")
            continue

        try:
            logging.info(f"Successful sync for {k}.")
            b[k] = a[k]
        except Exception as e:
            logging.info(f"Error during remote sync for {k}: {e}")
            return False

    return True


def remote_sync(local_dir, remote_dir, protocol):
    logging.info("Starting remote sync.")
    if protocol == "s3":
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == "fsspec":
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error("Remote protocol not known")
        return False


def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)


def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, local_dir, remote_dir, protocol),
    )
    return p


def terminate_sync_process(p: multiprocessing.Process):
    if p is not None and p.is_alive():
        logging.info(f"Terminating remote sync process.")
        p.terminate()


# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)


def _pt_load_s3_cp(file_path, map_location=None):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Failed to fetch model from s3. stderr: {stderr.decode()}")
    return torch.load(io.BytesIO(stdout), map_location=map_location)


def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
        return _pt_load_s3_cp(file_path, map_location)
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


def get_metadata_file(path):
    of = fsspec.open(path, "rb")
    with of as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]
    return out


def get_shards_for_chunk(num_samples, chunk, path):
    """Function to get a chunk of shards to train on.

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
        try:
            curr_chunk_count += m["num_sequences"]
        except KeyError:
            curr_chunk_count += m["num_chunks"]

        curr_shard_list.append(m["shard"])
        if curr_chunk_count >= num_samples:
            shard_list.append(curr_shard_list)
            chunk_count_list.append(curr_chunk_count)
            curr_shard_list = []
            curr_chunk_count = 0

    # Append remaining shards
    if len(curr_shard_list) > 0:
        shard_list.append(curr_shard_list)
        chunk_count_list.append(curr_chunk_count)

    return (
        shard_list[chunk % len(shard_list)],
        chunk_count_list[chunk % len(chunk_count_list)],
    )


def enough_shards(shard_lists: List[List[str]], min_shards_needed: int):
    for sl in shard_lists:
        if len(sl) < min_shards_needed:
            return False
    return True


def enough_samples(num_samples_per_source: List[List[int]], needed_samples_per_source: List[int]):
    for i, number_per_shard in enumerate(num_samples_per_source):
        if sum(number_per_shard) < needed_samples_per_source[i]:
            return False
    return True


def source_exhausted(paths, shard_list_per_source):
    for i, source in enumerate(paths):
        data = get_metadata_file(source)
        if len(data) < len(shard_list_per_source[i]):
            return True
    return False


def log_num_checkpoints(total_steps, args):
    """Log the number of checkpoints that will be made.

    This function counts the number of checkpoints to be made, and logs that number, printing out a warning if that
    number is different than expected.
    """

    steps_done = 0
    tokens_seen = 0
    next_shard_per_source = [0]
    checkpoints_made = 0

    if is_master(args):
        logging.info("Precounting number of steps / tokens seen per checkpoint:")

    while steps_done < total_steps:
        _, num_samples_per_source, next_shard_per_source = get_string_for_epoch(
            args.train_num_samples,
            next_shard_per_source,
            args.dataset_manifest,
            args.train_data_mix_weights,
            args.workers,
            args.world_size,
        )
        global_batch_size = args.world_size * args.batch_size
        steps_epoch = sum([(n // (args.workers * global_batch_size)) * args.workers for n in num_samples_per_source])
        steps_done += steps_epoch
        if steps_done > total_steps:
            steps_done = total_steps
        tokens_seen = steps_done * global_batch_size * args.seq_len
        checkpoints_made += 1

        if is_master(args):
            logging.info(f"==> Checkpoint {checkpoints_made}, steps {steps_done}, tokens seen {tokens_seen}")

    if is_master(args):
        logging.info(
            f"Number of checkpoints to be made: {checkpoints_made}."
            f"Number will be greater in case of unexpected failures leading to the use of more shards"
        )

        if checkpoints_made != args.epochs:
            logging.warning(
                f"{args.epochs} were requested, but {checkpoints_made} will be made. This behavior is a best effort in "
                f"checkpointing for the desired amount of epochs, and depends on the number of workers and gpus used, "
                f"as well as the size of the shards themselves."
            )

    return


def get_string_for_epoch(
    num_samples: int,
    starting_points: List[int],
    paths: List[str],
    weights: Optional[List[float]],
    num_workers_per_gpu: int,
    world_size: int,
    multi_epoch=False,
):
    """See _single_epoch_string for full docstring."""
    if multi_epoch:
        raise NotImplementedError("Multiple passes over the dataset not fully supported yet.")
    else:
        return _single_epoch_string(num_samples, starting_points, paths, weights, num_workers_per_gpu, world_size)


def _multi_epoch_string(num_samples, starting_chunk, paths, weights, min_shards_needed):
    """Multi epoch string training."""

    raise NotImplementedError("Function not fully supported yet.")

    if weights is None:
        weights = [1.0 / len(paths) for _ in range(len(paths))]
    needed_samples_per_source = [int(np.ceil(weights[i] * num_samples / sum(weights))) for i in range(len(weights))]
    shard_strings_per_source = []
    next_chunk = starting_chunk
    shard_list_per_source = [[] for _ in range(len(paths))]
    num_samples_per_source = [0 for _ in range(len(paths))]
    while not enough_shards(shard_list_per_source, min_shards_needed) or not enough_samples(
        num_samples_per_source, needed_samples_per_source
    ):
        for i, source_path in enumerate(paths):
            shard_list_source, num_samples_source = get_shards_for_chunk(
                needed_samples_per_source[i], next_chunk, source_path
            )
            shard_list_per_source[i].extend(shard_list_source)
            num_samples_per_source[i] += num_samples_source
        next_chunk += 1
        if source_exhausted(paths, shard_list_per_source):
            logging.warning(
                "Number of shards requested for a single epoch is more than the number of shards available. "
                "Consider lowering the number of workers and / or the number of GPUs, to avoid continuous "
                "reuse of samples."
            )

    for i, source_path in enumerate(paths):
        shard_list_source = shard_list_per_source[i]
        num_samples_source = num_samples_per_source[i]
        shard_root_source = "/".join(source_path.split("/")[:-1]) + "/"
        if len(shard_list_source) == 1:
            shard_string_source = shard_root_source + shard_list_source[0] + ".tar"
        else:
            shard_string_source = shard_root_source + "{" + ",".join(shard_list_source) + "}.tar"
        if source_path.startswith("s3"):
            shard_string_source = f"pipe:aws s3 cp {shard_string_source} -"
        shard_strings_per_source.append(shard_string_source)

    return shard_strings_per_source, num_samples_per_source, next_chunk


def _single_epoch_string(
    num_samples: int,
    starting_shard_per_source: List[int],
    paths: List[str],
    weights: Optional[List[float]],
    num_workers_per_gpu: int,
    world_size: int,
):
    """Retrieve shards to train on for a particular checkpoint.

    Currently only a single source is fully supported yet.

    Args:
        num_samples: Total number of samples required.
        starting_shard_per_source: First shard per source that has not been consumed yet.
        paths: Paths to source manifests.
        weights: Weighting between sources. If None, it is assumed to be uniform.
        num_workers_per_gpu: Number of workers per gpu process.
        world_size: Total number of gpus used for training.
        factor_drop_lower_avg: While creating the shard list, shards that contain fewer samples than this factor times a
            running average of samples are skipped. This is done to keep the number of samples per worker uniform, to
            decrease the amount of elements discarded.
    """

    num_sources = len(paths)

    if num_sources > 1:
        logging.warning("Multiple sources are not supported fully as of now")

    if weights is None:
        weights = [1.0 / num_sources for _ in range(num_sources)]

    assert len(weights) == num_sources, "One weight is needed per source."

    needed_samples_per_source = [int(np.ceil(weights[i] * num_samples / sum(weights))) for i in range(num_sources)]

    manifests = [get_metadata_file(path) for path in paths]
    shard_strings_per_source = []
    next_shard_per_source = copy.deepcopy(starting_shard_per_source)
    shard_list_per_source = [[] for _ in range(num_sources)]
    num_samples_per_source = [[] for _ in range(num_sources)]

    total_num_workers = num_workers_per_gpu * world_size
    while not enough_shards(shard_list_per_source, total_num_workers) or not enough_samples(
        num_samples_per_source, needed_samples_per_source
    ):
        try:
            for i in range(num_sources):
                # Add shards incrementally
                shard_name = manifests[i][next_shard_per_source[i]]["shard"]
                try:
                    num_samples_shard = manifests[i][next_shard_per_source[i]]["num_sequences"]
                except KeyError:
                    num_samples_shard = manifests[i][next_shard_per_source[i]]["num_chunks"]

                shard_list_per_source[i].append(shard_name)
                num_samples_per_source[i].append(num_samples_shard)

                next_shard_per_source[i] += 1

        except IndexError as e:
            logging.error(
                "Number of shards requested for a single epoch is more than the number of shards available. This means "
                "that the amount of data requested to train on is more than the dataloader can serve. This can either "
                "happen because there are not enough data to begin with, or data being skipped due to rounding errors. "
                "To alleviate the latter, consider making more uniform shards, and using less workers/GPUs. This will "
                "allow for better use of the dataset."
            )
            raise e

    for i in range(num_sources):
        # Ensure the number of shards is a multiple of number of workers, so each worker has the same
        # number of shards.
        #
        # This is a heuristic to minimize how much data we discard when trying to ensure each worker has
        # the same number of samples. Shards tend to have similar number of samples, so an extra shard
        # in a worker will likely get discarded.
        num_multiples = len(shard_list_per_source[i]) // total_num_workers

        shard_list_per_source[i] = shard_list_per_source[i][: num_multiples * total_num_workers]
        num_samples_per_source[i] = num_samples_per_source[i][: num_multiples * total_num_workers]

        # Put back unused shards.
        next_shard_per_source[i] = starting_shard_per_source[i] + len(shard_list_per_source[i])

    num_samples_per_source = [sum(n) for n in num_samples_per_source]

    for i, source_path in enumerate(paths):
        # Combine into a single shard string for training
        shard_list_source = shard_list_per_source[i]
        shard_root_source = "/".join(source_path.split("/")[:-1]) + "/"
        if len(shard_list_source) == 1:
            shard_string_source = shard_root_source + shard_list_source[0] + ".tar"
        else:
            shard_string_source = shard_root_source + "{" + ",".join(shard_list_source) + "}.tar"
        if source_path.startswith("s3"):
            shard_string_source = f"pipe:aws s3 cp {shard_string_source} -"
        shard_strings_per_source.append(shard_string_source)

    return shard_strings_per_source, num_samples_per_source, next_shard_per_source
