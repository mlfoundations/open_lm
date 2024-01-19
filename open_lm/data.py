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
from itertools import islice
import copy

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def proc_token(x, vocab_size):
    if type(x) is int:
        return x % vocab_size if x < 0 else x

    # FIXME: currently assuming that if not an int 0 is an appropriate token.
    # probably want to throw an error here instead. leaving as 0 for now for
    # backward compatibility with make_2048.py tokenization script.
    return 0


def preprocess_txt(text, vocab_size):
    return [proc_token(x, vocab_size) for x in ast.literal_eval(text.decode())]


# Decoding done in webdataset
def preprocess_json(text, vocab_size):
    text = [proc_token(x, vocab_size) for x in text]
    return text


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


class SyntheticDataset(Dataset):
    def __init__(self, seq_len, vocab_size, dataset_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        return ((torch.rand(self.seq_len + 1, generator=generator) * self.vocab_size).long(),)


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
        """Iterate through up to self.nsamples steps.

        Note: wds.DataPipeline.__iter__ inexplicably only limits the number of samples with self.nsamples if
        self.repetitions != 1. Here, we always slice using self.nsamples, if self.nsamples > 0.
        """

        if self.nsamples > 0:
            return islice(self.iterator(), self.nsamples)
        else:
            return self.iterator()


def get_wds_dataset(args, is_train, epoch=0, floor=True, tokenizer=None, data_key="json", force_num_samples=None):
    """Create a dataloader for a dataset in webdataset format.

    Args:
        args: Object created by the parser defined in open_lm/params.py.
        is_train: Whether the dataset is for training or not.
        epoch: Epoch for which the dataset is created.
        floor: If True, round down samples for the dataloader based on batch size. If False, round up. Defaults to True.
        tokenizer: The tokenizer used in preprocessing (currently unused due to the dataset being already tokenized.)
        data_key: Extension for items in the webdataset tarfiles.
        force_num_samples: If not None, this is a list with the desired number of samples per source.
    """
    input_shards_ = args.train_data if is_train else args.val_data

    assert input_shards_ is not None

    datasets = []
    all_num_samples = []

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    for ii, input_shards in enumerate(input_shards_):
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
        do_shuffle = resampled and not args.disable_buffer
        if is_train:
            if not resampled:
                pipeline.extend(
                    [
                        detshuffle2(
                            bufsize=_SHARD_SHUFFLE_SIZE if do_shuffle else 0,
                            initial=_SHARD_SHUFFLE_INITIAL if do_shuffle else 0,
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
                        bufsize=_SAMPLE_SHUFFLE_SIZE if do_shuffle else 0,
                        initial=_SAMPLE_SHUFFLE_INITIAL if do_shuffle else 0,
                        rng=random.Random(args.seed + shared_epoch.get_value()) if args.seed is not None else None,
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    wds.tarfile_to_samples(handler=wds.reraise_exception),
                    # splitting within a tar is fine for evaluation as no checkpointing
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )

        map_handler = {"handler": log_and_continue} if args.ignore_parse_errors else {}
        batch_size = args.per_gpu_batch_size if is_train else args.per_gpu_val_batch_size

        if data_key == "json" or data_key == "json.gz":
            pipeline.extend(
                [
                    wds.decode(**map_handler),
                    wds.rename(json=data_key),
                    wds.map_dict(json=partial(preprocess_json, vocab_size=args.vocab_size), **map_handler),
                    wds.to_tuple("json", **map_handler),
                    wds.select(partial(filter_lt_seqlen, args.seq_len)),
                    wds.batched(batch_size, partial=not is_train),
                ]
            )
        elif data_key == "txt":
            pipeline.extend(
                [
                    wds.map_dict(txt=partial(preprocess_txt, vocab_size=args.vocab_size), **map_handler),
                    wds.to_tuple("txt", **map_handler),
                    wds.select(partial(filter_lt_seqlen, args.seq_len)),
                    wds.batched(batch_size, partial=not is_train),
                ]
            )
        else:
            raise ValueError(f"Unrecognized data key: {data_key}")

        dataset = FiniteDataPipeline(*pipeline)
        datasets.append(dataset)
        all_num_samples.append(num_samples)

    if is_train:
        # TODO: why did we previoulsy wrap with RandomMix_
        dataset = RandomMix(datasets, probs=args.train_data_mix_weights, longest=True)
        if len(datasets) > 1:
            logging.warning("Source mixing is happening during training. It is preferred to mix during tokenization.")
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
        global_batch_size = batch_size * args.world_size
        total_num_batches = 0
        total_num_samples = 0
        for ii in range(len(datasets)):
            # Calculate batches per worker, round as little as possible.
            num_workers_per_gpu = max(1, args.workers)
            num_worker_batches = round_fn(all_num_samples[ii] / (global_batch_size * num_workers_per_gpu))

            if num_worker_batches == 0:
                raise ValueError(
                    f"The dataloader for source {ii} has received zero batches. This can happen due to rounding if "
                    f"too many GPUs / workers are used for this source, or if the mixing coefficient is too low. "
                    f"Consider addressing the above to fix this."
                )

            num_batches = num_worker_batches * num_workers_per_gpu
            num_samples = num_batches * global_batch_size

            # This forces the dataloader to take num_worker_batches steps per worker, so num_batches total.
            datasets[ii] = datasets[ii].repeat(nepochs=1, nbatches=num_worker_batches)

            total_num_batches += num_batches
            total_num_samples += num_samples
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)
        total_num_batches = num_batches
        total_num_samples = num_samples

    # Start a generator to have control over reproducibility.
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed + shared_epoch.get_value() * args.world_size + args.rank)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=resampled,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = total_num_batches
    dataloader.num_samples = total_num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_synthetic_dataset(args, is_train, epoch, tokenizer, data_key, floor):
    print(f"{args.train_num_samples=}")
    dataset = SyntheticDataset(seq_len=args.seq_len, vocab_size=args.vocab_size, dataset_size=args.train_num_samples)
    print(f"{len(dataset)=}")
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(dataset_type):
    if dataset_type == "synthetic":
        return get_synthetic_dataset
    else:
        return get_wds_dataset


def get_data(args, epoch=0, tokenizer=None, skip_train=False, floor=True):
    data = {}

    if skip_train:
        data["train"] = None
    else:
        if args.train_data or args.dataset_type == "synthetic":
            # train data is treated as a shard list where all data is combined and tained on
            data["train"] = get_dataset_fn(args.dataset_type)(
                args, is_train=True, epoch=epoch, tokenizer=tokenizer, data_key=args.data_key, floor=floor
            )

    if args.val_data:
        # val data is treated as independent val sets to be evaluated
        data["val_list"] = []
        for i, val_data in enumerate(args.val_data):
            args_copy = copy.deepcopy(args)
            args_copy.val_data = [val_data]
            data_val = {
                "val": get_dataset_fn(args.dataset_type)(
                    args_copy, is_train=False, tokenizer=tokenizer, data_key=args.val_data_key[i]
                )
            }
            data["val_list"].append(data_val)

    return data


# preprocessing tokens and sampling chunks


def squash_tok(seqs, mask, pad):
    """This function squashes all non-delimiter elements to the left side of the tensor, and replaces the rest elements with pad tokens in a given sequence.

    Args:
        seqs: torch.Tensor, input tensor with batch size as first dimension
        mask: torch.Tensor, binary mask tensor indicating non-delimiter elements of the same shape as 'seqs'
        pad: int, pad token to be used to fill in the non-masked areas of 'seqs'
    Returns:
        out_tensor: torch.Tensor, resulting tensor after squashing non-delimiter elements and padding on the right

    """
    # Calculate count of non-delimiter (mask values are True) elements in each batch
    valid_token_count = mask.sum(dim=1)

    # Create a tensor filled with pad tokens having the same shape as the input tensor 'seqs'
    out_tensor = torch.full(seqs.shape, pad, dtype=seqs.dtype, device=seqs.device)

    # For each sequence in the batch
    for i in range(seqs.shape[0]):
        # assign n non-delimiter tokens to the first n entries of the output
        # where n is given by indexing valid_token_count along the batch dim
        out_tensor[i, : valid_token_count[i]] = seqs[i, mask[i]]

    return out_tensor


def mask_sequence(chunk, start_idx, args, ignore_tok=-100):
    """Generate inputs and targets, aware of arg.target_mask_left, args.target_mask_individual, args.squash_mask_left
    The function generate an input in the following way:
        1.  get the input as chunck[start_idx : start_idx+seq_len]
        2.  if args.squash_mask_left is specified, remove instance of arg.target_mask_left and slide everything to the left
            In the event that we need to slide things left, we pad the sequence with args.target_mask_individual

    The function generate a target in the following way:
        1.  get the target as chunk[start_idx+1 : start_idx+seq_len]
        2a. if args.squash_mask_left, replace all tokens to the left of the rightmost arg.target_mask_left token with ignore_token
            excluding instances of arg.target_mask_left. then slide everything left and pad with args.target_mask_individual
        2b. if not args.squash_mask_left, replace all tokens to the left of the rightmost arg.target_mask_left token with ignore_token
            including instances of arg.target_mask_left
        3.  replace instances of args.target_mask_individual with ignore_tok


    Args:
        chunk: chunk implicitly containing input and target sequences (batch_size, seq_len+1).
        start_idx: the starting index of the input with sequence length args.seq_len.
        args: An object containing necessary arguments for masking.
              Must include 'seq_len' for specifying sequence length, 'target_mask_left' to specify token to mask,
              to the left of. 'squash_mask_left' a boolean indicating if 'target_mask_left' should be removed
              from the input and target, and 'target_mask_individual' to mask individual targets and pad the
              input to the right should 'target_mask_left' appear and 'squash_mask_left'=True.
        ignore_tok: The token to replace masked tokens with in the target. Defaults to -100.

    Returns:
        inputs: The masked inputs.
        targets: The masked targets.

    Example hand simulation for clarity:
        # L = args.target_mask_left
        # _ = ignore_tok
        # P = pad token (args.target_mask_individual)
        inputs                      = [a, b, c, L, d, e, L, f, P]
        targets                     = [b, c, L, d, e, L, f, P, P]
        targets_mask_left_positions = [0, 0, 1, 0, 0, 1, 0, 0, 0]
        cumsum_mask                 = [2, 2, 2, 1, 1, 1, 0, 0, 0] # warning: >1 L
        tok_mask                    = [1, 1, 1, 1, 1, 1, 0, 0, 0]
        # after tok_mask &= ~targets_mask_left_positions
        tok_mask                    = [1, 1, 0, 1, 1, 0, 0, 0, 0]
        # after targets[tok_mask] = ignore_tok; let _ = ignore_tok
        targets                     = [_, _, L, _, _, L, f, P, P]
        selected_inputs             = [1, 1, 1, 0, 1, 1, 0, 1, 1]
        selected_targets            = [1, 1, 0, 1, 1, 0, 1, 1, 1]
        # after squash, let P = pad token (args.target_mask_individual)
        inputs                      = [a, b, c, d, e, f, P, P, P]
        targets                     = [_, _, _, _, f, P, P, P, P]
        # after  args.target_mask_individual is not None
        inputs                      = [a, b, c, d, e, f, P, P, P]
        targets                     = [_, _, _, _, f, _, _, _, _]

    """

    inputs = torch.clone(chunk[:, start_idx : start_idx + args.seq_len])
    targets = torch.clone(chunk[:, start_idx + 1 : start_idx + args.seq_len + 1])

    if args.target_mask_left is not None:
        targets_mask_left_positions = targets == args.target_mask_left

        # construct cumulative mask for positions before (last) tok (if it appears)
        cumsum_mask = targets_mask_left_positions.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        if torch.any(cumsum_mask > 1):
            logging.warning(
                "> 1 target_mask_left tokens found in a sequence, using the last instance in the sequence for masking."
                "Please ensure data is correct."
            )

        # create mask for positions before (last) tok in each row (batch)
        ignore_mask = cumsum_mask > 0

        if args.squash_mask_left:
            # exclude target_mask_left in the ignore_mask as makes squashing easier
            ignore_mask &= ~targets_mask_left_positions

            targets[ignore_mask] = ignore_tok

            # code to squash the left mask and pad with args.target_mask_individual
            selected_inputs = inputs != args.target_mask_left
            selected_targets = targets != args.target_mask_left

            inputs = squash_tok(inputs, selected_inputs, args.target_mask_individual)
            targets = squash_tok(targets, selected_targets, args.target_mask_individual)

        else:
            targets[ignore_mask] = ignore_tok

    if args.target_mask_individual is not None:
        targets[targets == args.target_mask_individual] = ignore_tok

    return inputs, targets


def sample_chunk(chunk, args):
    if chunk.shape[1] == args.seq_len + 1:
        start_idx = 0
    elif chunk.shape[1] > args.seq_len + 1:
        start_idx = torch.randint(0, chunk.shape[1] - args.seq_len, (1,)).item()
    else:
        raise Exception(f"Invalid sequence length: Sequence length {args.seq_len} > {chunk.shape[1]} Chunk size")

    inputs = chunk[:, start_idx : start_idx + args.seq_len]
    targets = chunk[:, start_idx + 1 : start_idx + args.seq_len + 1]

    # replace elements to be masked with with -100 (pytorch default xent ignore value)
    if args.target_mask_left is not None or args.target_mask_individual is not None:
        inputs, targets = mask_sequence(chunk, start_idx, args)

    return inputs, targets
