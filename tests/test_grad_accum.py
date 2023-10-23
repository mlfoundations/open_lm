import torch
from torch import optim
import copy
from datasets import load_dataset

from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.data import get_data
from open_lm.scheduler import cosine_lr
from tests.testing_utils import download
from tests.shared import MockArgs


class TestsForAccumulatingGradients:
    """A class of unittest to accumulating gradients (with and without fsdp)"""

    # Setup data, optimizer, and other basic settings
    args = MockArgs("open_lm_11m")

    # download small shard file from huggingface
    train_data = download("testing_data", "open_lm/open_lm/tests/")
    args.train_data = ["open_lm/open_lm/tests/testing_data",]

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

    def test_grad_acc(self, accum_freq = 2, threshold = 1e-7):
        # create testing models
        model_accum_grad = copy.deepcopy(model).to(args.device) 
        model_no_accum_grad = copy.deepcopy(model_accum_grad).to(args.device) 

        # check that models weights are similar (within some threshold)
        for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
            assert torch.allclose(p1, p2, atol=threshold)
        

        # train on mock data with/without grad accumulation for one epoch
        for model, accum_freq in zip([model_accum_grad, model_no_accum_grad], [accum_freq, 1]):
            args.accum_freq = accum_freq
            train_one_epoch(
                model,
                data,
                loss,
                0,
                optimizer,
                args.scaler, 
                scheduler,
                args
            )
            
        # check that models weights are similar (within some threshold)
            for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
                assert torch.allclose(p1, p2, atol=threshold)
                

    def test_grad_acc_fsdp(self, accum_freq = 2, threshold = 1e-7):
        args.fsdp = True
        args.fsdp_amp = True
        
        # create models
        random_seed()
        model_accum_grad = copy.deepcopy(model).to(args.device) 
        model_no_accum_grad = copy.deepcopy(model_accum_grad).to(args.device) 

        # check that models weights are similar (within some threshold)
        for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
            assert torch.allclose(p1, p2, atol=threshold)

        # train on mock data with/without grad accumulation for one epoch
        for model, accum_freq in zip([model_accum_grad, model_no_accum_grad], [accum_freq, 1]):
            args.accum_freq = accum_freq
            train_one_epoch(
                model,
                data,
                loss,
                0,
                optimizer,
                args.scaler, 
                scheduler,
                args
            )
            
        # check that models weights are similar (within some threshold)
            for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
                assert torch.allclose(p1, p2, atol=threshold)
