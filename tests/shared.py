class MockArgs:
    def __init__(self, model):
        self.model = model
        self.model_norm = "gain_only_layer_norm"
        self.rotary_old = False
        self.qk_norm = False
        self.train_data = ""
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
        self.log_every_n_steps = 1e8  # v big as no logging needed
