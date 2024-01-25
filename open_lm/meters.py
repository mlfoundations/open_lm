import numpy as np
import torch
import torch.distributed as dist
import logging

from open_lm.distributed import is_master


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfidenceIntervalMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = []
        self.points_array = None

    def update(self, val):
        self.points.append(val)

    def compute_bootstrap_ci(self, max_population, num_iterations, interval=95):
        lower = None
        upper = None

        self.points_array = np.concatenate(self.points)

        num_points = self.points_array.shape[0]

        population_size = self.points_array.shape[0]
        if max_population is not None:
            population_size = min(max_population, population_size)

        estimates = []
        for _ in range(num_iterations):
            i = np.random.choice(num_points, size=population_size)
            estimate = np.sum(self.points_array[i]) / population_size
            estimates.append(estimate.item())

        half = (100 - interval) / 2

        lower = np.percentile(estimates, half).item()
        upper = np.percentile(estimates, 100 - half).item()

        return lower, upper


def combine_average_meters(meter_list):
    combined_meter = AverageMeter()

    # arbitarily get latest val as the val from the last
    combined_meter.val = meter_list[-1].val
    combined_meter.sum = sum([m.sum for m in meter_list])
    combined_meter.count = sum([m.count for m in meter_list])
    combined_meter.avg = combined_meter.sum / combined_meter.count

    return combined_meter


def combine_ci_meters(meter_list):
    combined_meter = ConfidenceIntervalMeter()
    for m in meter_list:
        combined_meter.points.extend(m.points)

    return combined_meter


def gather_meters(meters, args):
    out_meters = []
    for m in meters:
        combine_fn = None
        if isinstance(m, AverageMeter):
            combine_fn = combine_average_meters
        if isinstance(m, ConfidenceIntervalMeter):
            combine_fn = combine_ci_meters

        # buffer for a gather on all meters
        if is_master(args):
            # no need to gather unless its on master
            ms = [None for _ in range(args.world_size)]
            dist.gather_object(m, ms)
            out_meters.append(combine_fn(ms))
        else:
            # meters on all others are assumed to be local
            dist.gather_object(m)
            out_meters.append(m)

        dist.barrier()

    return out_meters
