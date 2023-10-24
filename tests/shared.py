import torch

from tests.testing_utils import download_val_data


class MockArgs:
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
        self.device = "cuda:0"
        self.wandb = False
        self.fsdp = False
        self.fsdp_amp = False
