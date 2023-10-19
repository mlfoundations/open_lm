import torch
from torch import optim
import copy

from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.data import get_data
from open_lm.scheduler import cosine_lr
from tests.shared import MockArgs


def test_grad_acc(accum_freq = 4, threshold = 1e-1):
    args = MockArgs("open_lm_11m")

    # create models
    random_seed()
    model_accum_grad = create_model(args)
    model_no_accum_grad = copy.deepcopy(model_accum_grad) #should I be using copy.deepcopy or clone?

    # create dataloader
    data = get_data(
        args,
        epoch=0,
        tokenizer=None,
        skip_train=False,
    )

    # create optimizer
    named_parameters = list(model_accum_grad.named_parameters())
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
    sum_layer_weight_diff = []
    for weight_model_1, weight_model_2 in zip(model_accum_grad.state_dict().items(), model_no_accum_grad.state_dict().items()):
        sum_layer_weight_diff.append(torch.sum(weight_model_1[1] - weight_model_2[1])) 
    assert torch.mean(sum_layer_weight_diff) < threshold
        

def test_grad_acc_fsdp():
    # TODO: similar to above but also init fsdp
    pass

test_grad_acc()