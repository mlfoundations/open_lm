import torch
from torch import optim

from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.data import get_data
from open_lm.scheduler import cosine_lr
from tests.utils import download_val_data


class MockTrainArgs:
    def __init__(self, model):
        data_path = download_val_data("shard_00000000.tar", "./tests/assets/")

        self.model = model  # part of model config
        self.model_norm = "gain_only_layer_norm"
        self.rotary_old = False
        self.qk_norm = False
        self.train_data = [
            data_path,
        ]
        self.log_logit_mean = False
        self.device = 0
        self.precision = "amp_bfloat16"
        self.wd = 0.033
        self.lr = 3e-3
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8
        self.warmup = 2
        self.skip_scheduler = False
        self.accum_freq = 1
        self.batch_size = 8
        self.grad_clip_norm = 1.0
        self.rank = 0
        self.local_rank = 0
        self.log_every_n_steps = 1e8
        self.dataset_type = "webdataset"
        self.data_key = "json"
        self.ffn_type = "swiglu"
        self.train_num_samples = 250000
        self.train_data_mix_weights = None
        self.train_data_upsampling_factors = None
        self.disable_buffer = False
        self.seed = 1
        self.vocab_size = 50432
        self.seq_len = 300
        self.workers = 1
        self.world_size = 1
        self.dataset_metadata = None
        self.val_data = None
        self.lr_cooldown_end = 3e-5
        self.force_min_lr = 0.0
        self.scaler = None
        self.accum_freq = 1
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.wandb = False
        self.fsdp = False
        self.fsdp_amp = False
        self.positional_embedding_type = "rotary"
        self.dist_backend = "nccl"
        self.dist_url = "env://"
        self.dataset_manifest = None
        self.target_mask_left = None
        self.target_mask_individual = None


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
        self.seq_len = 2048
        self.vocab_size = 50432
        self.batch_size = 64
        self.world_size = 1
        self.workers = 2
        self.dataset_metadata = None


def create_train_fixtures():
    # Setup data, optimizer, and other basic settings
    args = MockTrainArgs("open_lm_11m")

    # only want to look at one batch
    args.train_num_samples = args.batch_size

    # increase learning rate and remove warmup for maximize change to model weights
    args.lr = 2
    args.warmup = 0

    # create base models
    random_seed()
    model = create_model(args).to(args.device)

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
