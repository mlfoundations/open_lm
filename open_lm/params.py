import argparse
import ast
import copy
import json
import logging
import yaml


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def add_model_args(parser):
    """Add arguments that change the underlying architecture.

    These arguments need to be added to the eval code. Ideally, these should be moved to our model configs when we make
    a backward-incompatible release."""
    parser.add_argument(
        "--model-norm",
        type=str,
        default="default_layer_norm",
        choices=[
            "default_layer_norm",
            "lp_layer_norm",
            "gain_only_lp_layer_norm",
            "gain_only_layer_norm",
            "no_wb_layer_norm",
            "rms_norm",
        ],
        help="Type of normalization to employ in the model. This might be overridden by the model config.",
    )
    parser.add_argument(
        "--ffn-type",
        choices=["swiglu", "gelu"],
        default="swiglu",
        help="Type of feedforward layer to use. This might be overridden by the model config.",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        default=False,
        help="apply --model-norm to qk as in: https://arxiv.org/abs/2302.05442. This might be overridden by the model config.",
    )
    parser.add_argument(
        "--positional-embedding-type",
        type=str,
        choices=["rotary", "head_rotary", "llama_rotary"],
        default="rotary",
        help="Type of positional embedding to use. This might be overridden by the model config.",
    )


def check_replacement_type(replacement, original):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.

    Taken from YACS: https://github.com/rbgirshick/yacs/blob/32d5e4ac300eca6cd3b839097dde39c4017a1070/yacs/config.py#L494
    """
    # The types must match (with some exceptions)
    if type(original) == type(replacement):
        return True

    # If either of them is None, accept the type.
    if replacement is None or original is None:
        return True

    return False


def maybe_load_config(parser, args):
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config", type=str)
    args, unknown_args = config_parser.parse_known_args(args)
    if not args.config:
        return None

    assert not unknown_args, "No arguments can be passed if --config is provided."
    logging.info(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        elif args.config.endswith(".json"):
            config = json.load(f)
        else:
            raise ValueError(f"Unknown config format: {args.config}")

    default_args = vars(parser.parse_args([]))
    default_arg_keys = default_args.keys()
    updated_args = copy.deepcopy(default_args)

    for config_key, config_value in config.items():
        config_key = config_key.replace("-", "_")
        if config_key not in default_arg_keys:
            raise ValueError(f"Unknown config key: {config_key}")
        default_value = default_args[config_key]
        is_valid = check_replacement_type(replacement=config_value, original=default_value)
        if not is_valid:
            raise ValueError(
                f"Type mismatch (config: {type(config_value)} vs. argparse: {type(default_value)}) with values "
                f"(config: {config_value} vs. argparse: {default_value}) for config. key: {config_key}"
            )
        updated_args[config_key] = config_value

    return updated_args


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        nargs="+",
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    )
    parser.add_argument(
        "--val-data",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path to file(s) with validation data. Note: each space seperated entry will be processed seperately and writen as seperate entries "
            "in a results.jsonl file."
        ),
    )
    parser.add_argument(
        "--data-key",
        type=str,
        default="txt",
        help="what is the extension",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "auto", "synthetic"],
        default="auto",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        nargs="+",
        default=None,
        help="Uses manifest to construct a train set.",
    )
    parser.add_argument(
        "--disable-buffer",
        action="store_true",
        default=False,
        help="Turns off the shuffle buffer.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
    )
    parser.add_argument("--optimizer", default="adamw", help="Optimizer.")
    parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument(
        "--z-loss-coefficient",
        type=float,
        default=0.0,
        help="regularization term to make sure logits not too big, based on: https://github.com/google-research/t5x/blob/main/t5x/losses.py#L33-L38",
    )
    parser.add_argument(
        "--log-logit-mean",
        default=False,
        action="store_true",
        help="Whether to log the logit mean to wandb etc.",
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
        help="End learning rate for cooldown schedule. Default: 0",
    )
    parser.add_argument(
        "--lr-cooldown-power",
        type=float,
        default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
    )
    parser.add_argument(
        "--force-min-lr",
        type=float,
        default=0.0,
        help="Force the LR to stop decaying at this value.",
    )
    parser.add_argument("--save-frequency", type=int, default=1, help="How often to save checkpoints.")
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--torchcompile",
        action="store_true",
        default=False,
        help="Compile the model, requires torch >=2.0.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val-data (in epochs). Last epoch validated if val-data provided.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Batch size to be used with val-data.",
    )
    parser.add_argument(
        "--val-data-key",
        type=str,
        nargs="+",
        default=None,
        help="what is the extension fore each val-data source.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="open_lm_1b",
        help="Name of the model_config to use. Can also pass a custom json config.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Huggingface model/tokenizer name for AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--hf-seq-len",
        type=int,
        default=None,
        help="Sequence length for use with a --hf-model.",
    )
    parser.add_argument(
        "--hf-fsdp-block",
        type=str,
        default=None,
        help="transformer_layer_cls name in a --hf-model used for fsdp's transformer_auto_wrap_policy.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--load-pretrained-state",
        default=False,
        action="store_true",
        help="Include the opt and schedule state when loading a pre-trained model.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        default=False,
        action="store_true",
        help="CPU offloading for FSDP and checkpoint saving. This does not work with gradient accumulation.",
    )
    parser.add_argument(
        "--fsdp-use-orig-params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. This does not work for OPT models. Enables param_groups for weight_decay.",
    )
    parser.add_argument(
        "--fsdp-amp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-pure-bf16",
        default=False,
        action="store_true",
        help="Use pure bf16 FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp-backward-prefetch",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-hybrid",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-hybrid-o2",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-checkpoint",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fsdp-limit-all-gathers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--report-to",
        default="",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument("--wandb-notes", default="", type=str, help="Notes if logging with wandb")
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="open-lm",
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--average",
        type=str,
        nargs="+",
        default=None,
        help=("Apply model average on these checkpoints with the specified coefficients by --average-coefficients."),
    )
    parser.add_argument(
        "--average-coefficients",
        type=float,
        nargs="+",
        default=None,
        help=("Average the model weights with the specified coefficients, model weights specified by --average."),
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clip.")
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help="Which model arch to distill from, if any.",
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help="Which pre-trained weights to distill from, if any.",
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help="Replace the network linear layers from the bitsandbytes library. " "Allows int8 training/inference, etc.",
    )
    parser.add_argument(
        "--target-mask-left",
        type=int,
        default=None,
        help="Mask the loss to the left of a specified token (including the specified token).",
    )
    parser.add_argument(
        "--target-mask-individual",
        type=int,
        default=None,
        help="Mask the loss for a special pad token. Useful for sequences shorter than sequence lenght.",
    )
    parser.add_argument(
        "--ignore-parse-errors",
        action="store_true",
        default=False,
        help="If true, ignore parse errors in data loading. This should ideally be False, as errors in dataloading can point to bigger issues in your dataset. However, this can be useful when training on a large dataset which has a couple errors.",
    )

    add_model_args(parser)

    config = maybe_load_config(parser, args)
    if config is not None:
        args = argparse.Namespace(**config)
        logging.info(f"Loaded config from file: {args=}")
    else:
        args = parser.parse_args(args)

    if args.dataset_type == "synthetic":
        assert args.train_data is None, "--train-data must not be specified if --dataset-type='synthetic'"
        assert args.dataset_manifest is None, "--dataset-manifest must not be specified if --dataset-type='synthetic'"

    if args.val_data is not None and args.val_batch_size is None:
        # if not set explicitly make sure that the val batch size is set to the micro batch size

        args.val_batch_size = args.batch_size // args.accum_freq

    return args
