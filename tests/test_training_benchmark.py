# test_main.py
from open_lm.main import main, train_one_epoch
import ast
import itertools
import json
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ReduceOp
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    import wandb
except ImportError:
    wandb = None

from open_lm.distributed import is_master
from open_lm.precision import get_autocast

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfidenceIntervalMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.weights = []
        self.sums = []

    def update(self, val, n=1):
        self.weights.append(n)
        self.sums.append(val * n)

    def compute_bootstrap_ci(self, num_samples=2000, interval=95):
        lower = None
        upper = None

        estimates = []
        for _ in range(num_samples):
            acc = 0
            denom = 0
            i = np.random.choice(len(self.sums), size=len(self.sums)).tolist()
            for ii in i:
                acc += self.sums[ii]
                denom += self.weights[ii]

            acc /= denom

            estimates.append(acc)

        half = (100 - interval) / 2

        lower = np.percentile(estimates, half).item()
        upper = np.percentile(estimates, 100 - half).item()

        return lower, upper


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def replace_before_tok(tensor, tok, replaced, excusive=False):
    # NOTE: this implementation supports 0 or 1 instance of tok in a sequence.
    #       if more than one instance appears, the last instace of tok is used.
    #       if exclusive=True every instance of tok will be present in the output

    tok_positions = tensor == tok

    # construct cumulative mask for positions before (last) tok (if it appears)
    cumsum_mask = tok_positions.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    # create mask for positions before (last) tok in each row (batch)
    tok_mask = cumsum_mask > 0

    if excusive:
        # retain tok in the output
        tok_mask &= ~tok_positions

    out = torch.clone(tensor)
    out[tok_mask] = replaced

    return out


def replace_tok(tensor, tok, replaced):
    out = torch.clone(tensor)
    out[out == tok] = replaced

    return out


def sample_chunk(chunk, args):
    if chunk.shape[1] == args.seq_len + 1:
        start_idx = 0
    elif chunk.shape[1] > args.seq_len + 1:
        start_idx = torch.randint(0, chunk.shape[1] - args.seq_len, (1,)).item()
    else:
        raise Exception(f"Invalid sequence length: Sequence length {args.seq_len} > {chunk.shape[1]} Chunk size")

    inputs = chunk[:, start_idx: start_idx + args.seq_len]
    targets = chunk[:, start_idx + 1: start_idx + args.seq_len + 1]

    # replace elements to be masked with with -100 (pytorch default xent ignore value)
    if args.target_mask_left is not None:
        targets = replace_before_tok(targets, args.target_mask_left, -100)
    if args.target_mask_individual is not None:
        targets = replace_tok(targets, args.target_mask_individual, -100)

    return inputs, targets


def simple_train_one_epoch(model, data, loss, epoch, step, optimizer, scaler, scheduler, total_steps, args,
                           tb_writer=None):
    """Trains model for one epoch on the provided data.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the epoch. Note that "epoch" actually is not one full pass through the
            data, but rather the number of tokens specified by `--train-num-samples`, rounded based on shard size.
            As such, the number of steps in an "epoch" can vary, and we have to keep track of steps separately.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    data["train"].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    data_iterator = iter(dataloader)
    for i in itertools.count():
        if not args.skip_scheduler:
            scheduler(step)

        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        if args.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < args.world_size:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                inputs, targets = sample_chunk(texts, args)
                out, _ = model(inputs)

                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())

                total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))

            backward(total_loss, scaler)
        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert args.batch_size % args.accum_freq == 0, "Batch size must be divisible by accum_freq"
            per_batch = args.batch_size // args.accum_freq

            inputs, targets = sample_chunk(texts, args)

            for ii in range(args.accum_freq):
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(model, FSDP) and ii != args.accum_freq - 1:
                    maybe_no_sync = model.no_sync
                with maybe_no_sync():
                    with autocast():
                        inputs_ii = inputs[ii * per_batch: (ii + 1) * per_batch]
                        if inputs_ii.shape[0] == 0:
                            break
                        targets_ii = targets[ii * per_batch: (ii + 1) * per_batch]
                        out, _ = model(inputs_ii)

                        if args.log_logit_mean:
                            logit_m.update(torch.mean(out).item())

                        local_loss = (
                                loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                                * inputs_ii.shape[0]
                                / inputs.shape[0]
                        )
                    backward(local_loss, scaler)
                if ii == 0:
                    total_loss = local_loss
                else:
                    total_loss += local_loss

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()

        global_loss_tensor = total_loss.detach().clone()
        if args.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)

        batch_count = i + 1
        step += 1

        if is_master(args) and (
                i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch or step == total_steps - 1
        ):
            batch_size = len(inputs)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # gathered_loss = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
            # torch.distributed.all_gather(gathered_loss, total_loss)
            # losses_m.update(sum(gathered_loss).item() / args.world_size, batch_size * args.world_size)
            losses_m.update(global_loss_tensor.item(), batch_size)
            samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {losses_m.avg:.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": losses_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": (step + 1) * args.batch_size * args.seq_len * args.world_size,
            }

            if args.log_logit_mean:
                log_data["logit_mean"] = logit_m.val

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, "Please install wandb."
                    wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

            if math.isnan(losses_m.val):
                # case where loss goes to nan, we see this sometimes with bad nodes.
                # in this case we would like to free resources and prevent other issues
                # e.g., saving checkpoints and optmization states that may lead to skipped
                # training on restarts.
                return False, step

            if percent_complete >= 5.0:
                expected_token_sec = 192754.84461
                expected_token_gpu_sec = 64251.61487
                # To check/assert if samples_per_second and samples_per_second_per_gpu are within 10 % of the the expected values
                assert abs(samples_per_second - expected_token_sec) / expected_token_sec < 0.1
                assert abs(samples_per_second_per_gpu - expected_token_gpu_sec) / expected_token_gpu_sec < 0.1
                # print('test passed!!')

    # end for
    return True, step


# Override the train_one_epoch function in main with the simple version
main.train_one_epoch = simple_train_one_epoch


def test_train_simple():
    VALSHARD = "/scratch/08002/gsmyrnis/open_lm_val_data/shard_00000000.tar"

    LR = 0.003
    SAVES = 1
    TOKENS = 654000000
    BATCHSIZE = 32  # 192 # 64, 128
    WARM = 2000  # 200, 400, 1000, 2000
    MODEL = "open_lm_160m"  # "open_lm_1b" #"open_lm_160m"
    WD = 0.033
    ACC = 16  # 24 # 4, 8
    CD = 3e-05  # 4, 8

    main([
        "--train-num-samples", str(TOKENS),
        "--workers", "2",
        "--dataset-manifest", "/scratch/08002/gsmyrnis/open_lm_tokenized/rpj/manifest.jsonl",
        "/scratch/08002/gsmyrnis/open_lm_tokenized/not_rpj/manifest.jsonl",
        "--train-data-mix-weights", "0.725", "0.275",
        "--precision", "amp_bfloat16",
        "--batch-size", str(BATCHSIZE),
        "--grad-checkpointing",
        "--log-every-n-steps", "20",
        "--grad-clip-norm", "1",
        "--lr", str(LR),
        "--warmup", str(WARM),
        "--model", "open_lm_160m",
        "--wd", str(WD),
        "--beta2", "0.95",
        "--epochs", "1",
        "--name", "exp_name",
        "--logs", "$SCRATCH/open_lm_logs",
        "--resume", "latest",
        "--seed", "124",
        "--data-key", "'json'",
        "--accum-freq", str(ACC),
        "--lr-cooldown-end", str(CD),
        "--report-to", "wandb",
        "--wandb-project-name", "lm1"])


# Execute the test
test_train_simple()
