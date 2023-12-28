import torch
from torch import optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_lm.data import get_data
from open_lm.distributed import init_distributed_device
from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.params import parse_args
from open_lm.scheduler import cosine_lr
from tests.utils import download_val_data


class MockTrainArgs:
    def __init__(self, model, **kwargs):
        data_path = download_val_data("shard_00000000.tar", "./tests/assets/")

        # fmt: off
        args = parse_args([
            "--model", model,
            "--model-norm", "gain_only_layer_norm",
            "--train-data", data_path,
            "--precision", "fp32",
            "--wd", "0.033",
            "--lr", "3e-3",
            "--warmup", "2",
            "--global-batch-size", "8",
            "--accum", "1",
            "--name", "test_model_name",
            "--logs", "./tests/assets/",
            "--workers", "1",
            "--data-key", "json",
            "--seed", "1",
        ])
        # fmt: off
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.vocab_size = 50432
        self.seq_len = 300
        self.wandb = False
        self.fsdp = False
        self.fsdp_amp = False
        self.positional_embedding_type = "rotary"
        self.dist_backend = "nccl"
        self.dist_url = "env://"
        self.dataset_manifest = None
        self.target_mask_left = None
        self.target_mask_individual = None
        self.ignore_parse_errors = False
        self.moe_num_experts = None
        self.moe_freq = 0
        self.moe_weight_parallelism = False
        self.moe_expert_model_parallelism = False
        self.moe_capacity_factor = 1.25
        self.moe_loss_weight = 0.1
        self.moe_top_k = 2
        self.distributed = False
        self.per_gpu_batch_size = self.global_batch_size // self.world_size

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Recalculate batch size if overwritten.
        if "global_batch_size" in kwargs:
            self.per_gpu_batch_size = self.global_batch_size // self.world_size


class MockDataArgs(object):
    def __init__(self):
        data_path = download_val_data("shard_00000000.tar", "./tests/assets/")

        self.train_data = [
            data_path,
        ]
        self.dataset_resampled = True
        self.train_data_mix_weights = None
        self.val_num_samples = 0
        self.train_data_upsampling_factors = None
        self.train_num_samples = 512
        self.disable_buffer = True
        self.seq_len = 300
        self.vocab_size = 50432
        self.global_batch_size = 64
        self.world_size = 1
        self.rank = 0
        self.workers = 2
        self.seed = 42
        self.dataset_manifest = None
        self.target_mask_left = None
        self.target_mask_individual = None
        self.ignore_parse_errors = False
        self.per_gpu_batch_size = self.global_batch_size // self.world_size


def create_train_fixtures(model="open_lm_11m", fsdp=False, **kwargs):
    # Setup data, optimizer, and other basic settings
    args = MockTrainArgs(model, **kwargs)
    args.fsdp = fsdp

    # only want to look at one batch
    args.train_num_samples = args.global_batch_size

    # increase learning rate and remove warmup for maximize change to model weights
    args.lr = 1e-3
    args.warmup = 0

    # create base models
    random_seed()
    if fsdp:
        model = create_model(args)
        model = FSDP(model)
    else:
        model = create_model(args)
        model.reset_parameters()
        model = model.to(args.device)

    # create dataloader
    data = get_data(
        args,
        epoch=0,
        tokenizer=None,
        skip_train=False,
    )

    # create optimizer
    named_parameters = list(model.named_parameters())
    params = [p for _, p in named_parameters if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # create scheduler
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup,
        10,
        args.lr_cooldown_end,
        args.force_min_lr,
    )

    # create loss
    loss = torch.nn.CrossEntropyLoss()

    return args, model, data, optimizer, scheduler, loss
