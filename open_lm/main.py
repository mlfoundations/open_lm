import atexit
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
import functools
import numpy as np
from pathlib import Path
import json
import math

import fsspec
import torch
from torch import optim
from torch.cuda.amp import GradScaler

import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .data import proc_token
from .model import Block
from .losses import CrossEntropyLossWithZLoss

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_lm.model import create_model
from open_lm.utils.transformers.hf_wrapper import create_wrapped_hf_model
from .data import get_data, get_wds_dataset
from .distributed import is_master, init_distributed_device, broadcast_object
from .logger import setup_logging
from .params import parse_args
from .scheduler import cosine_lr
from .train import train_one_epoch, evaluate
from .file_utils import (
    pt_load,
    check_exists,
    start_sync_process,
    remote_sync,
    get_string_for_epoch,
    log_num_checkpoints,
    terminate_sync_process,
)


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str):
    is_s3 = path.startswith("s3")
    fs, root_path = fsspec.core.url_to_fs(path)
    checkpoints = fs.glob(os.path.join(root_path, "**/epoch_*.pt"))
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return f"s3://{checkpoints[-1]}" if is_s3 else checkpoints[-1]

    return None


def get_state_dict(name):
    checkpoint = pt_load(name, map_location="cpu")
    if "epoch" in checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
    else:
        sd = checkpoint
    return sd


def load_model(args, model):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        global_step = checkpoint.get("step", None)
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        start_epoch, global_step = 0, 0
        model.load_state_dict(checkpoint)
        logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    return start_epoch, global_step


def load_optimizer(args, model, optimizer, scaler):
    potential_checkpoint = args.resume.replace("epoch_", "optimizer_")
    if check_exists(potential_checkpoint):
        checkpoint = pt_load(potential_checkpoint, map_location="cpu")
    else:
        checkpoint = pt_load(args.resume, map_location="cpu")
    if "optimizer" in checkpoint:
        if optimizer is not None:
            osd = checkpoint["optimizer"]
            if args.fsdp:
                osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=osd)
            optimizer.load_state_dict(osd)
            logging.info(f"=> resuming optimizer")
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        logging.info(f"=> WARNING: not resuming optimizer.")


def load_data_chunks(args):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "next_shard_per_source" in checkpoint and "samples_seen" in checkpoint:
        return checkpoint["next_shard_per_source"], checkpoint["samples_seen"]
    else:
        logging.info(
            "=> WARNING: tried to resume a checkpoint without data loading info. Re-starting data loading from the "
            "first shard."
        )
        return [0 for _ in range(len(args.dataset_manifest))], 0


def save_checkpoint(
    args,
    model,
    optimizer,
    scaler,
    completed_epoch,
    evaluation_loss,
    step,
    is_final_checkpoint,
    next_shard_per_source=None,
    samples_seen=None,
):
    cpu_state, optim_state = None, None
    if args.logs and args.logs.lower() != "none" and args.fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)

    if args.save_logs:
        checkpoint_dict_model = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": cpu_state if args.fsdp else model.state_dict(),
            "evaluation_loss": evaluation_loss,
        }
        if next_shard_per_source is not None:
            checkpoint_dict_model["next_shard_per_source"] = next_shard_per_source

        if samples_seen is not None:
            checkpoint_dict_model["samples_seen"] = samples_seen

        if step is not None:
            checkpoint_dict_model["step"] = step

        checkpoint_dict_opt = {
            "epoch": completed_epoch,
            "name": args.name,
            "optimizer": optim_state if args.fsdp else optimizer.state_dict(),
            "evaluation_loss": evaluation_loss,
        }
        if scaler is not None:
            checkpoint_dict_opt["scaler"] = scaler.state_dict()

        checkpoint_dict_stats = {
            "epoch": completed_epoch,
            "name": args.name,
            "is_final_checkpoint": is_final_checkpoint,
            "evaluation_loss": evaluation_loss,
        }

        if (
            completed_epoch == args.epochs
            or is_final_checkpoint
            or (args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0)
        ):
            torch.save(
                checkpoint_dict_model,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
            torch.save(
                checkpoint_dict_opt,
                os.path.join(args.checkpoint_path, f"optimizer_{completed_epoch}.pt"),
            )
            torch.save(
                checkpoint_dict_stats,
                os.path.join(args.checkpoint_path, f"stats_{completed_epoch}.pt"),
            )

        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)
            previous_checkpoint = os.path.join(args.checkpoint_path, f"optimizer_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)
            previous_checkpoint = os.path.join(args.checkpoint_path, f"stats_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)


def main(args):
    args = parse_args(args)

    requires_training = args.train_data or args.dataset_type == "synthetic" or args.dataset_manifest is not None

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.hf_model is not None and args.hf_seq_len is None:
        raise ValueError("If passing --hf-model, must also pass --hf-seq-len to be used for training/fine-tuning.")

    if args.hf_model is not None and args.fsdp and args.hf_fsdp_block is None:
        raise ValueError("If passing --hf-model and --fsdp, must also pass --hf-fspd-block.")

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = None
        if args.hf_model is not None:
            model_name_safe = args.hf_model.replace("/", "-")
        else:
            if Path(args.model).is_file():
                model_name_safe = Path(args.model).stem.replace("/", "-")
            else:
                model_name_safe = args.model.replace("/", "-")

        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            raise ValueError(f"Experiment {args.log_path} already exists. Use --name to specify a new experiment.")

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = "wandb" in args.report_to or "all" in args.report_to
    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name)
            if args.save_most_recent:
                raise ValueError("Cannot use save-most-recent with remote_sync and resume latest.")
            if args.remote_sync_protocol != "s3":
                raise ValueError("Sync protocol not supported when using resume latest.")
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, "checkpoints", LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path)
            if resume_from:
                logging.info(f"Found latest resume checkpoint at {resume_from}.")
            else:
                logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("remote sync successful.")
        else:
            raise ValueError("Remote sync failed.")
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        remote_sync_process.start()

        # make sure that if open_lm throws the remote process is still killed to prevent hanging
        atexit.register(terminate_sync_process, p=remote_sync_process)

    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )

    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)

    model = None
    if args.hf_model is not None:
        model = create_wrapped_hf_model(args)
    else:
        model = create_model(args)

    args.vocab_size = model.vocab_size
    args.seq_len = model.seq_len
    if args.train_num_samples is not None:
        args.train_num_samples //= args.seq_len
    if args.val_num_samples is not None:
        args.val_num_samples //= args.seq_len

    model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info(f"Model (has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters):")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # optionally resume model from a checkpoint
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        start_epoch, global_step = load_model(args, model)
    elif args.pretrained is not None:
        print("=> loading from a pre-trained model.")
        args.resume = args.pretrained
        _epoch, _step = load_model(args, model)
        # this flag continues training from the pre-trained model.
        if args.load_pretrained_state:
            start_epoch, global_step = _epoch, _step
        else:
            args.resume = None
    elif args.average is not None:
        num_models_to_average = len(args.average)
        print(
            "=> Averaging models: ",
            args.average,
            " with coefficients: ",
            args.average_coefficients,
        )
        assert num_models_to_average > 1, "num_models_to_average must be > 1 - else use --pretrained"
        if args.average_coefficients is None:
            args.average_coefficients = [1.0 / num_models_to_average] * num_models_to_average
        else:
            assert len(args.average_coefficients) == num_models_to_average
        state_dict = {k: v * args.average_coefficients[0] for k, v in get_state_dict(args.average[0]).items()}
        for i in range(1, num_models_to_average):
            state_dict_i = get_state_dict(args.average[i])
            for k in state_dict:
                state_dict[k] = state_dict[k] + state_dict_i[k] * args.average_coefficients[i]
        model.load_state_dict(state_dict)

    if requires_training and global_step is None:
        raise ValueError("Key 'step' not found in checkpoint, but required for training.")

    # Add data chunk when resuming (only for dataset without resampling)
    next_shard_per_source = [0 for _ in range(len(args.dataset_manifest))] if args.dataset_manifest is not None else 0
    samples_seen = 0
    if args.resume is not None and args.dataset_manifest is not None:
        next_shard_per_source, samples_seen = load_data_chunks(args)
        if samples_seen >= args.train_num_samples * args.epochs:
            raise RuntimeError("Loaded a checkpoint which has already seen the desired number of tokens.")

    if args.distributed:
        if args.fsdp:
            transformer_layer_cls = None

            if args.hf_model is not None:
                # retrive the user specified block class for fsdp
                for _, target_cls in model.named_modules():
                    if args.hf_fsdp_block in type(target_cls).__name__:
                        transformer_layer_cls = {type(target_cls)}
                        break

                if transformer_layer_cls is None:
                    print(f"--hf-fsdp-block {args.hf_fsdp_block} not found in --hf-model {args.hf_model}")
                    return -1

            else:
                transformer_layer_cls = {Block}
            # from https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
            transformer_auto_wrapper_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            )
            # tries to follow gopher...
            mp_policy = None
            if args.fsdp_amp:
                print("=> using bfloat16 params as part of fsdp amp policy.")
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
            elif args.fsdp_pure_bf16:
                print("=> using pure bfloat16 params as part of fsdp amp policy.")
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            if args.rank == 0:
                print(f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())}")
                print(f"Before FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

            fsdp_kwargs = {}
            assert not (
                args.fsdp_hybrid and args.fsdp_hybrid_o2
            ), "Only --fsdp-hybrid or --fsdp-hybrid-o2 should be set."
            if args.fsdp_backward_prefetch:
                fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            if args.fsdp_hybrid:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
            if args.fsdp_hybrid_o2:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy._HYBRID_SHARD_ZERO2
            print("=> FSDP kwargs: ", fsdp_kwargs)

            # init FSDP
            model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrapper_policy,
                device_id=device,
                mixed_precision=mp_policy,
                cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
                use_orig_params=args.fsdp_use_orig_params,
                limit_all_gathers=args.fsdp_limit_all_gathers,
                **fsdp_kwargs,
            )

            print(f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}")
            print(f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}")
        else:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args["static_graph"] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if requires_training:
        named_parameters = list(model.named_parameters())
        no_decay_params = []  # to be potentially used later
        params = [p for n, p in named_parameters if p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = None
        if args.precision == "amp":
            assert not args.fsdp, "FSDP not supported with amp, only amp_bfloat16"
            scaler = GradScaler()

    # optionally resume optimizer from a checkpoint
    if args.resume is not None:
        load_optimizer(args, model, optimizer, scaler)

    # initialize datasets
    # use tokenizer=None because the data is already pre-tokenized.
    if args.val_data is not None:
        args.val_data = [args.val_data]
    data = get_data(
        args,
        epoch=start_epoch,
        tokenizer=None,
        skip_train=args.dataset_manifest is not None,
        floor=args.dataset_manifest is not None,
    )

    if args.target_mask_left is not None and args.target_mask_individual == args.target_mask_left:
        logging.error(f"--target-mask-left and --target-mask-individual set to same value of {args.target_mask_left}.")
        exit(1)

    if args.target_mask_left is not None:
        # tokens handled with same modulo in dataloading
        args.target_mask_left = proc_token(args.target_mask_left, args.vocab_size)

    if args.target_mask_individual is not None:
        # tokens handled with same modulo in dataloading
        args.target_mask_individual = proc_token(args.target_mask_individual, args.vocab_size)

    if args.torchcompile:
        logging.info("Compiling model...")
        model = torch.compile(model)

    # create scheduler if train
    scheduler = None
    if requires_training:
        if args.dataset_manifest is not None:
            total_steps = (args.train_num_samples * args.epochs) // (args.batch_size * args.world_size)
        else:
            total_steps = (data["train"].dataloader.num_batches) * args.epochs

        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(
                optimizer,
                args.lr,
                args.warmup,
                total_steps,
                args.lr_cooldown_end,
                args.force_min_lr,
            )
        else:
            logging.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
            )
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    if args.wandb and is_master(args):
        assert wandb is not None, "Please install wandb."
        logging.debug("Starting wandb.")
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume=None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log="all")
        wandb.save(params_file)
        logging.debug("Finished loading wandb.")

    if not requires_training:
        checkpoint_root = os.path.dirname(args.resume)

        metrics = evaluate(model, data, start_epoch, args, writer)
        metrics["checkpoint_path"] = args.resume
        metrics["val_data"] = args.val_data
        metrics["model"] = args.hf_model if args.hf_model else args.model

        if is_master(args):
            with fsspec.open(os.path.join(checkpoint_root, "results.jsonl"), "a") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        return

    loss = torch.nn.CrossEntropyLoss()
    if args.z_loss_coefficient != 0.0:
        if is_master(args):
            logging.info("Using CrossEntropyLossWithZLoss.")
        loss = CrossEntropyLossWithZLoss(args.z_loss_coefficient)

    if args.dataset_manifest:
        log_num_checkpoints(total_steps, args)

    # Only enter training loop if there are steps to be done.
    done_training = global_step >= total_steps
    epoch = start_epoch
    while not done_training:
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        if args.dataset_manifest is not None:
            assert not args.dataset_resampled, "dataset_manifest and dataset_resampled are mutually exclusive"
            (
                train_data_string_per_source,
                num_samples_per_source,
                next_shard_per_source,
            ) = get_string_for_epoch(
                args.train_num_samples,
                next_shard_per_source,
                args.dataset_manifest,
                args.train_data_mix_weights,
                args.workers,
                args.world_size,
            )

            if data["train"] is not None:
                del data["train"]
            args.train_data = train_data_string_per_source

            # Draw num_samples_per_source at most from dataset - rounded down to guarantee uniqueness.
            data["train"] = get_wds_dataset(
                args, True, epoch, force_num_samples=num_samples_per_source, data_key=args.data_key, floor=True
            )

        prev_step = global_step
        if is_master(args):
            logging.info(f"=> epoch {epoch}, training on {args.train_data}")

        if args.distributed:
            dist.barrier()

        success, global_step = train_one_epoch(
            model,
            data,
            loss,
            epoch=epoch,
            step=global_step,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            total_steps=total_steps,
            args=args,
            tb_writer=writer,
        )

        if args.distributed:
            dist.barrier()

        done_training = global_step >= total_steps
        steps_done_epoch = global_step - prev_step
        samples_seen = samples_seen + steps_done_epoch * args.batch_size * args.world_size

        if not success:
            logging.info("Training exiting due to NaN value")
            break

        epoch = epoch + 1
        evaluation_loss = -1
        if "val" in data and (epoch % args.val_frequency == 0 or done_training):
            # validate based on frequency and always validate the last checkpoint
            evaluation_loss = evaluate(model, data, epoch, args, writer)["loss"]

        # Saving checkpoints.
        save_checkpoint(
            args,
            model,
            optimizer,
            scaler,
            epoch,
            evaluation_loss,
            step=global_step,
            is_final_checkpoint=done_training,
            next_shard_per_source=next_shard_per_source if args.dataset_manifest is not None else None,
            samples_seen=samples_seen if args.dataset_manifest is not None else None,
        )

        if done_training:
            if is_master(args):
                logging.info("Model has seen the desired number of tokens. Ending training.")
            break

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info("Final remote sync.")
        terminate_sync_process(remote_sync_process)
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol,
        )
        if result:
            logging.info("Final remote sync successful.")
        else:
            logging.info("Final remote sync failed.")


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb"))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
