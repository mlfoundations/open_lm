import json
import tempfile
import pytest
import yaml
from contextlib import contextmanager

from open_lm.main import main
from open_lm.params import parse_args


@contextmanager
def create_config(config_dict, file_type="json"):
    assert file_type in ("json", "yaml")
    with tempfile.NamedTemporaryFile(mode="w", suffix="." + file_type) as f:
        if file_type == "json":
            json.dump(config_dict, f)
        elif file_type == "yaml":
            yaml.safe_dump(config_dict, f)
        f.seek(0)
        yield f


def get_cmdline_config1():
    samples = 1000
    batch_size = 2
    # fmt: off
    cmdline = [
        "--train-num-samples", str(samples),
        "--global-batch-size", str(batch_size),
        "--dataset-type", "synthetic",
        "--model", "open_lm_test_tiny",
        "--epochs", "1",
    ]
    config_dict = {
        "train-num-samples": samples,
        "global-batch-size": batch_size,
        "dataset-type": "synthetic",
        "model": "open_lm_test_tiny",
        "epochs": 1,
    }
    # fmt: on
    return cmdline, config_dict


@pytest.mark.parametrize("filetype", ["json", "yaml"])
def test_config_params1(filetype):
    cmdline, config_dict = get_cmdline_config1()
    cmdline_args = parse_args(cmdline)
    with create_config(config_dict, filetype) as f:
        config_args = parse_args(["--config", f.name])
    assert vars(cmdline_args) == vars(config_args), "Config and command line match failed"


@pytest.mark.parametrize("filetype", ["json", "yaml"])
def test_wrong_type_throws(filetype):
    config_dict = {"train-num-samples": "100"}
    with create_config(config_dict, filetype) as f:
        try:
            parse_args(["--config", f.name])
        except ValueError as e:
            assert "Type mismatch" in str(e)


@pytest.mark.parametrize("filetype", ["json", "yaml"])
def test_extra_config_key_throws(filetype):
    config_dict = {"this-key-should-not-exist": "100"}
    with create_config(config_dict, filetype) as f:
        try:
            parse_args(["--config", f.name])
        except ValueError as e:
            assert "Unknown config" in str(e)


@pytest.mark.parametrize("filetype", ["json", "yaml"])
def test_extra_arg_after_config_throws(filetype):
    config_dict = {"this-key-should-not-exist": "100"}
    with create_config(config_dict, filetype) as f:
        try:
            parse_args(["--config", f.name, "--train-data", "foo"])
        except AssertionError as e:
            assert "--config is provided" in str(e)
