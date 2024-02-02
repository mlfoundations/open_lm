import itertools
import logging
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss
    from megablocks.layers.arguments import Arguments as MoEArgs
except ImportError:
    logging.warning(f"Megablocks not installed. To train MoE, install with pip install megablocks.")

try:
    import wandb
except ImportError:
    wandb = None

from open_lm.data import sample_chunk
from open_lm.datapreprocess.ray.tokenize_shuffle import SpecialTokens
from open_lm.distributed import is_master
from open_lm.precision import get_autocast
from open_lm.meters import AverageMeter


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


def get_document_seqlens(inputs, args):
    """Get list of document sequence lengths.

    Return a list of lists. The length of the outer list is equal to the batch size, while the length of the inner list
    is equal to the the number of distinct documents (recognized by EOT tokens). Each element of the inner lists is the
    length of that corresponding document
    """
    if args.mask_across_documents:
        document_seqlens = []
        for idx in range(inputs.shape[0]):
            eot_idx = torch.nonzero(inputs[idx] == SpecialTokens.END_OF_TEXT.value)
            if eot_idx.shape[0] == 0:
                # All tokens come from the same document.
                document_seqlens.append([args.seq_len])
            else:
                start_idx = 0
                seqlens = []
                for k in range(eot_idx.shape[0]):
                    seqlens.append(eot_idx[k].item() - start_idx + 1)
                    start_idx = eot_idx[k].item() + 1
                if start_idx < args.seq_len:
                    seqlens.append(args.seq_len - start_idx)
                document_seqlens.append(seqlens)
    else:
        document_seqlens = None
    return document_seqlens


def train_one_epoch(model, data, loss, epoch, step, optimizer, scaler, scheduler, total_steps, args, tb_writer=None):
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
    load_balancing_losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    data_iterator = iter(dataloader)

    if args.moe_freq > 0:
        # these MoEArgs are necessary for logging load balancing.
        moe_args = MoEArgs(
            hidden_size=model.dim,
            ffn_hidden_size=model.dim * 4,
            moe_num_experts=args.moe_num_experts,
            num_layers=model.n_layers // 2,
            moe_expert_model_parallelism=True,
            moe_top_k=args.moe_top_k,
            device=torch.cuda.current_device(),
            moe_capacity_factor=args.moe_capacity_factor,
            moe_loss_weight=args.moe_loss_weight,
            fp16=False,
            bf16=False,
        )

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
                document_seqlens = get_document_seqlens(inputs, args)
                if args.mask_across_documents:
                    # Some input samples contain EOT as the final token. The prediction after that is meaningless, so it
                    # should not contribute to the loss.
                    ignore_indices = torch.nonzero(inputs == SpecialTokens.END_OF_TEXT.value, as_tuple=True)
                    targets[ignore_indices] = loss.ignore_index

                out, _, _ = model(inputs, document_seqlens=document_seqlens)

                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())

                total_lm_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                total_loss = total_lm_loss
                if args.moe_freq > 0:
                    total_load_balancing_loss = batched_load_balancing_loss(moe_args)
                    clear_load_balancing_loss()
                    total_loss += total_load_balancing_loss

            backward(total_loss, scaler)
        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert args.per_gpu_batch_size % args.accum_freq == 0, "Per-GPU batch size must be divisible by accum_freq"
            per_batch = args.per_gpu_batch_size // args.accum_freq

            inputs, targets = sample_chunk(texts, args)
            if args.mask_across_documents:
                # Some input samples contain EOT as the final token. The prediction after that is meaningless, so it
                # should not contribute to the loss.
                ignore_indices = torch.nonzero(inputs == SpecialTokens.END_OF_TEXT.value, as_tuple=True)
                targets[ignore_indices] = loss.ignore_index

            for ii in range(args.accum_freq):
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(model, FSDP) and ii != args.accum_freq - 1:
                    maybe_no_sync = model.no_sync
                with maybe_no_sync():
                    with autocast():
                        inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                        if inputs_ii.shape[0] == 0:
                            break
                        targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                        document_seqlens = get_document_seqlens(inputs_ii, args)
                        out, _, _ = model(inputs_ii, document_seqlens=document_seqlens)

                        if args.log_logit_mean:
                            logit_m.update(torch.mean(out).item())

                        local_lm_loss = (
                            loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                            * inputs_ii.shape[0]
                            / inputs.shape[0]
                        )
                    local_loss = local_lm_loss
                    if args.moe_freq > 0:
                        local_load_balancing_loss = batched_load_balancing_loss(moe_args)
                        clear_load_balancing_loss()
                        local_loss += local_load_balancing_loss

                    backward(local_loss, scaler)
                if ii == 0:
                    total_lm_loss = local_lm_loss
                    if args.moe_freq > 0:
                        total_load_balancing_loss = local_load_balancing_loss
                else:
                    total_lm_loss += local_lm_loss
                    if args.moe_freq > 0:
                        total_load_balancing_loss += local_load_balancing_loss

            total_loss = total_lm_loss
            if args.moe_freq > 0:
                total_loss += total_load_balancing_loss

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
            if args.moe_freq > 0:
                losses_m.update(global_loss_tensor.item() - total_load_balancing_loss.item(), batch_size)
                load_balancing_losses_m.update(total_load_balancing_loss.item(), batch_size)
            else:
                losses_m.update(global_loss_tensor.item(), batch_size)
            samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
            loss_str = f"Loss: {losses_m.avg:.3f}"
            loss_str += f" LB-Loss: {load_balancing_losses_m.avg:.3f}" if args.moe_freq > 0 else ""
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"{loss_str} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": losses_m.val,
                "load_balancing_loss": load_balancing_losses_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": (step + 1) * args.global_batch_size * args.seq_len,
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

    # end for
    if tb_writer is not None:
        tb_writer.flush()
    return True, step
