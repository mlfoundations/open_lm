import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        autocast_fn = torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
        return lambda: autocast_fn(dtype=torch.bfloat16)
    else:
        return suppress
