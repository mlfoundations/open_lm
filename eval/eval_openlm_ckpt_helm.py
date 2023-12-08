import argparse
import os
import shutil
import yaml
from wrap_checkpoint_for_hf import do_wrap
from open_lm.params import add_model_args
from pathlib import Path

BASE_PATH = Path("eval/helm_deployments/")

HELM_MODEL_DEPLOYMENTS_PATH = BASE_PATH / "prod_env/model_deployments.yaml"
HELM_TOKENIZER_CONFIG_PATH = BASE_PATH / "prod_env/tokenizer_configs.yaml"
MODEL_WRAPPER_PATH = BASE_PATH / "huggingface_wrappers/"

RUN_SPEC_PATH = BASE_PATH / "run_specs_dec2023.conf"


def add_yaml_config(model_name: str, out_dir: Path, tokenizer: str, params):
    """
    Adds a model to the model_deployments.yaml file in the helm_deployments/ directory.

    Args:
        model_name (str): The name of the model to be added.
        out_dir (Path): The output directory where the model is saved.
        tokenizer (str): The name of the tokenizer used by the model.
        params: Additional parameters for the model.

    Returns:
        None
    """
    if os.path.exists(HELM_MODEL_DEPLOYMENTS_PATH):
        config = yaml.load(open(HELM_MODEL_DEPLOYMENTS_PATH), Loader=yaml.FullLoader)
        config["model_deployments"] = [x for x in config["model_deployments"] if x["name"] != model_name]
    else:
        config = {
            "model_deployments": [],
        }
    new_config = {
        "name": model_name,
        "tokenizer_name": tokenizer,
        "max_sequence_length": params.seq_len,
        "client_spec": {
            "class_name": "helm.proxy.clients.huggingface_client.HuggingFaceClient",
            "args": {
                "pretrained_model_name_or_path": str(out_dir.resolve()),
            },
        },
    }
    config["model_deployments"].append(new_config)
    with open(HELM_MODEL_DEPLOYMENTS_PATH, "w") as f:
        yaml.dump(config, f)


def ensure_tokenizer_config():
    """
    Ensure the existence of the tokenizer configuration file. Necessary for HELM to run.
    If the file does not exist, create it with default configuration.
    """
    if not HELM_TOKENIZER_CONFIG_PATH.exists():
        config = {
            "tokenizer_configs": [
                {
                    "name": "open_lm",
                    "tokenizer_spec": {
                        "class_name": "helm.proxy.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                        "args": {},
                    },
                }
            ],
        }
        HELM_MODEL_DEPLOYMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HELM_TOKENIZER_CONFIG_PATH, "w") as f:
            yaml.dump(config, f)


def generate_run_specs(model_name: str, out_dir: Path):
    config = (RUN_SPEC_PATH).read_text()
    (out_dir / "run_specs.conf").write_text(config.replace("{model_name}", model_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the type of model to use. (`open_lm_1b` for example)"
    )
    parser.add_argument("--experiment", type=str, default="default", help="Name of the experiment.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint.")
    parser.add_argument(
        "--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Name of the tokenizer to use."
    )
    parser.add_argument("--copy-model", action="store_true", help="Copy the model to the evaluation directory.")
    parser.add_argument("--overwrite-model", action="store_true", help="Overwrite the wrapper if it already exists.")
    parser.add_argument(
        "--use-existing-model", action="store_true", help="Use the existing wrapper if it already exists."
    )
    add_model_args(parser)
    args = parser.parse_args()

    experiment = args.experiment
    model = args.model

    out_dir = MODEL_WRAPPER_PATH / experiment / model
    model_name = f'open_lm/{experiment}-{model.replace("/", "_")}'

    if os.path.exists(out_dir) and args.overwrite_model:
        print(f"Overwriting model {out_dir}...")
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir):
        if not args.ignore_existing_model:
            raise ValueError(f"Output directory {out_dir} already exists.")
    else:
        # Model not found. Generate all necessary files to run HELM on this model
        checkpoint = args.checkpoint
        tokenizer = args.tokenizer
        copy_model = args.copy_model
        if not os.path.exists(checkpoint):
            raise ValueError(f"Checkpoint {checkpoint} does not exist.")

        from open_lm.model import create_params

        params = create_params(args)
        do_wrap(args.checkpoint, params, out_dir, tokenizer, copy_model)

        ensure_tokenizer_config()
        add_yaml_config(model_name, out_dir, tokenizer, params)
        generate_run_specs(model_name, out_dir)

    helm_args = [
        "helm-run",
        "--conf-paths",
        str(out_dir.resolve() / "run_specs.conf"),
        "--suite",
        experiment,
        # For each dataset, evaluate only on 1000 instances
        "--max-eval-instances",
        "1000",
        # Running using only one thread to fix run time
        "-n1",
        # # For the Slurm environment. Must change HELM codebase for your environment.
        # "--runner-class-name",
        # "helm.benchmark.slurm_runner.SlurmRunner"
    ]
    print(f"Running HELM with cmd: \n{' '.join(helm_args)}")
    os.chdir(BASE_PATH)
    os.execvp(helm_args[0], helm_args)
