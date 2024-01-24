import logging
import math
import time
import copy

import torch
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from open_lm.data import sample_chunk
from open_lm.distributed import is_master
from open_lm.precision import get_autocast
from open_lm.meters import (
    AverageMeter,
    ConfidenceIntervalMeter,
    gather_meters,
)


@torch.inference_mode()
def evaluate(model, data, start_epoch, args, writer):
    """
    evaluates perplexity on validation data
    """
    if is_master(args):
        print("=> begin evaluation")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(start_epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader

    # NOTE: dataloader.num_batches = 0 corresponds to exhausting iterator by convention
    exhaust_loader = dataloader.num_batches == 0

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    sps_m = AverageMeter()
    spspg_m = AverageMeter()
    losses_seq_ci_m = ConfidenceIntervalMeter()
    losses_tok_ci_m = ConfidenceIntervalMeter()

    end = time.time()
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    # by default the dataloader will be exhausted
    for i, batch in enumerate(dataloader):
        if i == dataloader.num_batches and not exhaust_loader:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)

        data_time_m.update(time.time() - end)

        with autocast():
            inputs, targets = sample_chunk(texts, args)

            out, _, _ = model(inputs)  # [per_gpu_bs, seq_len, vocab_size]

            bs, seq_len = targets.shape

            targets = targets.reshape(-1)
            total_loss = loss(out.reshape(-1, args.vocab_size), targets)  # [bs * seq_len]

            # cross entropy ignores -100 values in loss computation
            mask = targets != -100

            # reshape and average for sequence losses
            sum_loss_per_seq = torch.sum(total_loss.reshape(bs, seq_len), -1)
            num_toks_per_seq = torch.sum(mask.reshape(bs, seq_len), -1).float()
            losses_seq_ci_m.update((sum_loss_per_seq / num_toks_per_seq).cpu().numpy())

            # individual token losses
            losses_tok_ci_m.update(total_loss[mask].cpu().numpy())

            # compute average loss for the mini-batch
            total_loss = total_loss[mask].mean()
            losses_m.update(total_loss.item(), n=inputs.shape[0])

        batch_time_m.update(time.time() - end)
        end = time.time()

        sps_m.update(inputs.numel() * args.world_size / batch_time_m.val)
        spspg_m.update(inputs.numel() / batch_time_m.val)

    if args.distributed:
        dist.barrier()

    if args.world_size > 1:
        # in this case we need to gather the loss. for simplicity we gather the meters only on main proc
        # Save eval loss / etc.
        meters = [
            losses_m,
            batch_time_m,
            data_time_m,
            sps_m,
            spspg_m,
            losses_seq_ci_m,
            losses_tok_ci_m,
        ]

        # meters on master will become global meters, other meters will remain local
        losses_m, batch_time_m, data_time_m, sps_m, spspg_m, losses_seq_ci_m, losses_tok_ci_m = gather_meters(
            meters, args
        )

    if args.distributed:
        dist.barrier()

    lower_seq, upper_seq, lower_tok, upper_tok = -1.0, -1.0, -1.0, -1.0
    if args.val_seq_ci:
        lower_seq, upper_seq = losses_seq_ci_m.compute_bootstrap_ci(args.val_max_pop_ci, args.val_iter_ci)

    if args.val_tok_ci:
        lower_tok, upper_tok = losses_tok_ci_m.compute_bootstrap_ci(args.val_max_pop_ci, args.val_iter_ci)

    num_seqs = sum([len(p) for p in losses_seq_ci_m.points])
    num_toks = sum([len(p) for p in losses_tok_ci_m.points])

    # Save eval loss / etc.
    log_data = {
        "loss": losses_m.avg,
        "data_time": data_time_m.avg,
        "batch_time": batch_time_m.avg,
        "samples_per_second": sps_m.avg,
        "samples_per_second_per_gpu": spspg_m.avg,
        "loss_sequences_lower_95": lower_seq,
        "loss_sequences_upper_95": upper_seq,
        "loss_tokens_lower_95": lower_tok,
        "loss_tokens_upper_95": upper_tok,
        "sequences": num_seqs,
        "tokens": num_toks,
    }
    if args.train_num_samples is not None:
        log_data["train_tokens"] = start_epoch * args.train_num_samples * args.seq_len

    for name, val in log_data.items():
        name = "valid/" + name
        if writer is not None:
            writer.add_scalar(name, val, start_epoch)
        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            wandb.log({name: val, "epoch": start_epoch, "tokens": log_data["tokens"]})

    if is_master(args):
        # meters on masters should be global
        print(f"evaluation on: {args.val_data}")
        print(f"evaluation loss: {losses_m.avg}")
        print(f"num loss point evaluations {losses_m.count}")
        print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
        print(f"num seqs: {num_seqs}")
        print(f"num tokens: {num_toks}")

    log_data["checkpoint_path"] = args.resume
    log_data["val_data"] = args.val_data
    log_data["model"] = args.hf_model if args.hf_model else args.model

    return log_data


def evaluate_loop(model, data_list, start_epoch, args, writer):
    log_data_list = []
    for i, data in enumerate(data_list):
        args_copy = copy.deepcopy(args)
        args_copy.val_data = [args.val_data[i]]
        args_copy.val_data_key = args.val_data_key[i]

        if args.distributed:
            dist.barrier()

        log_data_list.append(evaluate(model, data, start_epoch, args_copy, writer))

        if args.distributed:
            dist.barrier()

    return log_data_list
