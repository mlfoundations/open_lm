import numpy as np
import torch
import logging
from copy import deepcopy


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class ModelAverager(object):
    def __init__(self, model, methods: str):
        self.model = model
        self.avgs_dict = {}
        for method in methods.split(","):
            args = method.split("_")
            # method_name = args[0][:-1]  if args[0].endswith('_') else args[0]
            # freq = int(args[1]) if len(args) > 1 else 1
            self.avgs_dict[method] = Averager(model, args)

    def step(self):
        for avg in self.avgs_dict.values():
            avg.step()


class Averager(object):
    def __init__(self, model, args):
        self.model = model
        self.method = args[0]
        self.update_counter = 1
        self.step_counter = 1
        self.freq = 1 if (len(args) <= 2) else int(args[2])
        if self.method == "none":
            self.av_model = model
            return
        else:
            self.av_model = deepcopy(unwrap_model(model))

        if self.method == "poly":
            self.eta = 0.0 if len(args) <= 1 else float(args[1])
        elif self.method == "ema":
            self.gamma = 0.99 if len(args) <= 1 else float(args[1])

        elif self.method == "cosine":
            pass
        else:
            print(f"Unknown averaging method {self.method}")

    def step(self):
        if self.update_counter != self.freq:
            pass
        else:
            self.update()
        self.update_counter += 1
        if self.update_counter > self.freq:
            self.update_counter = 1
        return

    def update(self):
        method = self.method
        if method == "none":
            return
        t = self.step_counter
        # model_sd is the current model state dict
        # av_sd is the averaged model state dict
        model_sd = self.model.state_dict()
        av_sd = self.av_model.state_dict()
        if self.method == "cosine" or self.method == "degree":
            pass
        first_k_av_sd = list(av_sd.keys())[0]
        for k in model_sd.keys():
            av_sd_k = k
            if k.startswith("module") and not first_k_av_sd.startswith("module"):
                av_sd_k = k[len("module.") :]
            if isinstance(av_sd[av_sd_k], (torch.LongTensor, torch.cuda.LongTensor)):
                # these are buffers that store how many batches batch norm has seen so far
                av_sd[av_sd_k].copy_(model_sd[k])
                continue
            if method == "poly":
                # the update rule is: new_average = (1 - (eta + 1) / (eta + t)) * old_average + (eta + 1) / (eta + t) * current_model
                # which is eq(10) in https://arxiv.org/pdf/1212.1824.pdf
                av_sd[av_sd_k].mul_(1 - ((self.eta + 1) / (self.eta + t))).add_(
                    model_sd[k], alpha=(self.eta + 1) / (self.eta + t)
                )
            if method == "ema":
                # the update rule is: new_average = (1 - gamma) * old_average + gamma * current_model
                av_sd[av_sd_k].mul_(self.gamma).add_(model_sd[k], alpha=1 - self.gamma)
        self.step_counter += 1

    def reset(self):
        self.step_counter = 2

    @property
    def averaged_model(self):
        return self.av_model

    def get_state_dict_avg(self):
        state_dict = {
            "update_counter": self.update_counter,
            "step_counter": self.step_counter,
            "freq": self.freq,
            "av_model_sd": unwrap_model(self.av_model).state_dict(),
            "method": self.method,
            "eta": self.eta if hasattr(self, "eta") else None,
            "gamma": self.gamma if hasattr(self, "gamma") else None,
            "suffix_steps": self.suffix_steps if hasattr(self, "suffix_steps") else None,
            "power": self.power if hasattr(self, "power") else None,
            "start": self.start if hasattr(self, "start") else None,
        }
        return state_dict

    def load_state_dict_avg(self, state_dict):
        self.update_counter = state_dict["update_counter"]
        self.step_counter = state_dict["step_counter"]
        self.freq = state_dict["freq"]
        self.method = state_dict["method"]
        self.av_model.load_state_dict(state_dict["av_model_sd"])
        if hasattr(self, "eta"):
            self.eta = state_dict["eta"]
        if hasattr(self, "gamma"):
            self.gamma = state_dict["gamma"]
        if hasattr(self, "suffix_steps"):
            self.suffix_steps = state_dict["suffix_steps"]
        if hasattr(self, "power"):
            self.power = state_dict["power"]
        if hasattr(self, "start"):
            self.start = state_dict["start"]
