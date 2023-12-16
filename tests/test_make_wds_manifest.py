import pytest
from open_lm.utils import make_wds_manifest as mwm
from tests.utils import download_dl_test_data
import os
import json

""" Test strategy:
Given some tarfiles with "correct" manifests on huggingface, let's assert 
that we can recreate them 
"""


@pytest.mark.parametrize("source_dir", ["source_1", "source_2"])
def test_make_manifest_from_source(source_dir):
    download_dl_test_data("tests/assets")

    MOCK_MANIFEST = "tests/assets/%s/mock_manifest.jsonl" % source_dir
    if os.path.exists(MOCK_MANIFEST):
        os.unlink(MOCK_MANIFEST)

    args = ["--data-dir", "tests/assets/%s" % source_dir, "--manifest-filename", "mock_manifest.jsonl"]
    mwm.main(args)

    true_manifest = "tests/assets/%s/manifest.jsonl" % source_dir
    with open(true_manifest, "r") as true_file:
        with open(MOCK_MANIFEST, "r") as mock_file:
            assert true_file.read() == mock_file.read()

    if os.path.exists(MOCK_MANIFEST):
        os.unlink(MOCK_MANIFEST)


def test_make_toplevel_manifest():
    download_dl_test_data("tests/assets")

    MOCK_MANIFEST = "tests/assets/mock_manifest.jsonl"
    if os.path.exists(MOCK_MANIFEST):
        os.unlink(MOCK_MANIFEST)

    args = ["--data-dir", "tests/assets/", "--manifest-filename", "mock_manifest.jsonl"]
    mwm.main(args)

    lines = [json.loads(_) for _ in open(MOCK_MANIFEST, "r").readlines()]
    assert lines == [{"shard": "shard_00000000", "num_sequences": 120}]

    if os.path.exists(MOCK_MANIFEST):
        os.unlink(MOCK_MANIFEST)
