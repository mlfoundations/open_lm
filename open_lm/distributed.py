# This is from open_clip.
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
has_FSDP = True
def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) >= 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) >= 1
    return False


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    fairseq-specific checkpoint saving/loading logic.

    Args:
        is_moe (bool): if True, use MoE-specific checkpointing logic
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, is_moe: bool = False, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        if is_moe is None:
            if torch.distributed.get_rank() == 0:
                from openlm import pdb; pdb.set_trace()
            else:
                import time; time.sleep(1000)
        assert is_moe is not None
        super().__init__(*args, **kwargs)
        self.is_moe = is_moe
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        if self.flatten_parameters:
            return self.module.module
        else:
            return self.module

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        
        if self.use_sharded_state:
            return super().local_state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        elif self.is_moe:
            return super().state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            if self.rank == 0:
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
            else:
                # We must call state_dict() due to use of communication
                # primitives. But we don't use the result.
                super().state_dict()
                return destination or {}

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        if self.use_sharded_state:
            return super().load_local_state_dict(state_dict, strict=strict)
        elif self.is_moe:
            return super().load_state_dict(state_dict, strict=strict)
        else:
            state_dict = dist_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group
            )
            return super().load_state_dict(state_dict, strict=strict)


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    if args.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(args.world_size)]
    dist.all_gather_object(objects, obj)
    return objects
