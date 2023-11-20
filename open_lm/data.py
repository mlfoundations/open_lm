# This is from open_clip

import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
from functools import partial
import json
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from itertools import islice

from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
from webdataset.mix import RandomMix



def proc_token(x, vocab_size):
    if type(x) is int:
        return x % vocab_size if x < 0 else x

    # FIXME: currently assuming that if not an int 0 is an appropriate token.
    # probably want to throw an error here instead. leaving as 0 for now for
    # backward compatibility with make_2048.py tokenization script.
    return 0


def preprocess_txt(text, vocab_size):
    return [proc_token(x, vocab_size) for x in ast.literal_eval(text.decode())]


def preprocess_json(text, vocab_size):
    text = json.loads(text.decode())
    text = [proc_token(x, vocab_size) for x in text]
    return text


def _batched_fulldata(data, 
                      batchsize=20, 
                      collation_fn=wds.filters.default_collation_fn,
                      partial=True):
    batch = []
    first_batch = None
    for sample in data:
        if len(batch) >= batchsize:
            if first_batch == None:
                first_batch = batch            
            if collation_fn is not None:
                batch = collation_fn(batch)
            yield batch

            batch = []
        batch.append(sample)

    if len(batch) == 0:
        return
    elif len(batch) == batchsize or partial:
        if collation_fn is not None:
            batch = collation_fn(batch)
        yield batch
    elif len(batch) < batchsize and not partial:
        for sample in first_batch:
            batch.append(sample)
            if len(batch) >= batchsize:
                if collation_fn is not None:
                    batch = collation_fn(batch)
                yield batch
                return

batched_fulldata = wds.pipelinefilter(_batched_fulldata)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


# class RandomMix_(RandomMix):
#     def __init__(self, datasets, probs=None, longest=False):
#         super().__init__(datasets, probs=probs, longest=longest)

#     def with_epoch(self, nsamples=-1, nbatches=-1):
#         """Change the epoch to return the given number of samples/batches.

#         The two arguments mean the same thing."""
#         self.repetitions = sys.maxsize
#         self.nsamples = max(nsamples, nbatches)
#         return self


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 20000
_SAMPLE_SHUFFLE_INITIAL = 4000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def filter_lt_seqlen(seq_len, x):
    valid_sample = len(x[0]) > seq_len
    if not valid_sample:
        logging.warning(
            f"Sample sequence length: {len(x[0])} not larger than seq_len: {seq_len}. Skipping sample. NOTE: sample sequence length should be one greater than seq_len."
        )

    return valid_sample


class FiniteDataPipeline(wds.DataPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Create an iterator through the pipeline, repeating and slicing as requested.
        
        This differs from wds.DataPipeline since it allows for slicing even if self.repetitions = 1.
        """
        return islice(self.iterator(), self.nsamples)


def get_wds_dataset(
    args,
    is_train,
    epoch=0,
    floor=True,
    tokenizer=None,
    data_key="json",
    force_num_samples=None,
    multi_epoch=False
):  
    input_shards_ = args.train_data if is_train else args.val_data

    assert input_shards_ is not None

    datasets = []
    all_num_samples = []
    for ii, input_shards in enumerate(input_shards_): # Loop over all shards
        resampled = getattr(args, "dataset_resampled", False) and is_train
        num_shards = None
        if is_train:
            if args.train_num_samples is not None:
                if force_num_samples is not None and force_num_samples[ii] > 0:
                    num_samples = force_num_samples[ii]
                else:
                    if args.train_data_mix_weights is not None:
                        num_samples = int(args.train_num_samples * args.train_data_mix_weights[ii])
                    else:
                        num_samples = args.train_num_samples // len(input_shards_)
            else:
                num_samples, num_shards = get_dataset_size(input_shards)
                if not num_samples:
                    raise RuntimeError(
                        "Currently, the number of dataset samples must be specified for the training dataset. "
                        "Please specify it via `--train-num-samples` if no dataset length info is present."
                    )
        else:
            # Eval will just exhaust the iterator if the size is not specified.
            num_samples = args.val_num_samples or 0

        shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

        if resampled:
            pipeline = [
                ResampledShards2(
                    input_shards,
                    weights=None,
                    deterministic=True,
                    epoch=shared_epoch,
                )
            ]
        else:
            assert (
                args.train_data_upsampling_factors is None
            ), "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
            pipeline = [wds.SimpleShardList(input_shards)]

        # at this point we have an iterator over all the shards
        # disable shuffling if sampling w/o replacement to ensure no repeat
        not_use_shuffle = args.disable_buffer or not resampled
        if is_train:
            if not resampled:
                pipeline.extend(
                    [
                        detshuffle2(
                            bufsize=0 if not_use_shuffle else _SHARD_SHUFFLE_SIZE,
                            initial=0 if not_use_shuffle else _SHARD_SHUFFLE_INITIAL,
                            seed=args.seed,
                            epoch=shared_epoch,
                        ),
                        wds.split_by_node,
                        wds.split_by_worker,
                    ]
                )
            pipeline.extend(
                [
                    # at this point, we have an iterator over the shards assigned to each worker at each node
                    tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                    wds.shuffle(
                        bufsize=0 if not_use_shuffle else _SAMPLE_SHUFFLE_SIZE,
                        initial=0 if not_use_shuffle else _SAMPLE_SHUFFLE_INITIAL,
                        rng=random.Random(args.seed + shared_epoch.get_value()) if args.seed is not None else None,
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker
                    wds.tarfile_to_samples(handler=log_and_continue),
                ]
            )

        map_dict_handler = {"handler": log_and_continue} if args.ignore_parse_errors else {}
        if data_key == "json":
            pipeline.extend(
                [
                    wds.map_dict(json=partial(preprocess_json, vocab_size=args.vocab_size), **map_dict_handler),
                    wds.to_tuple("json"),
                    wds.select(partial(filter_lt_seqlen, args.seq_len)),
                    batched_fulldata(args.batch_size, partial=not is_train),
                ]
            )
        else:
            pipeline.extend(
                [
                    wds.map_dict(txt=partial(preprocess_txt, vocab_size=args.vocab_size), **map_dict_handler),
                    wds.to_tuple("txt"),
                    wds.select(partial(filter_lt_seqlen, args.seq_len)),
                    batched_fulldata(args.batch_size, partial=not is_train),
                ]
            )

        dataset = FiniteDataPipeline(*pipeline)
        datasets.append(dataset)
        all_num_samples.append(num_samples)

    if is_train:
        # TODO: why did we previoulsy wrap with RandomMix_
        dataset = RandomMix(datasets, probs=args.train_data_mix_weights, longest=True)
    else:
        pass

        # dataset = datasets[0]
    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            if num_shards < args.workers * args.world_size:
                print("Please increase --train-num-samples or decrease workers or world size")
                print(f"num_shards: {num_shards}, workers: {args.workers}, world_size: {args.world_size}")
            assert num_shards >= args.workers * args.world_size, "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        total_num_batches = 0
        total_num_samples = 0
        for ii in range(len(datasets)):
            # Calculate batches per worker, round as little as possible.
            num_workers = max(1, args.workers)
            num_worker_batches = round_fn(all_num_samples[ii] / (global_batch_size * num_workers))
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * global_batch_size

            # This forces the dataloader to take num_worker_batches steps per worker, so num_batches total.
            # Note that this internally sets num_repetitions = sys.maxsize, therefore allowing repeats. We are
            # safeguarded by the fact that num_worker_batches is the number of minimum worker batches.
            datasets[ii] = datasets[ii].with_epoch(num_worker_batches)
            
            total_num_batches += num_batches
            total_num_samples += num_samples
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)
        total_num_batches = num_batches
        total_num_samples = num_samples

    print("persistent workers:", args.dataset_manifest is None)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.dataset_manifest is None,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = total_num_batches
    dataloader.num_samples = total_num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(data_path, dataset_type):
    return get_wds_dataset


def get_data(args, epoch=0, tokenizer=None, skip_train=False):
    data = {}

    if skip_train:
        data["train"] = None
    else:
        if args.train_data or args.dataset_type == "synthetic":
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args,
                is_train=True,
                epoch=epoch,
                tokenizer=tokenizer,
                data_key=args.data_key,
            )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, is_train=False, tokenizer=tokenizer, data_key=args.data_key
        )

    return data
