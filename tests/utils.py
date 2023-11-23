import os
from tqdm import tqdm
import urllib.request
import hashlib
import warnings
import tarfile
import json

from pathlib import Path
from huggingface_hub import snapshot_download
import torch

from open_lm.utils import make_wds_manifest as mwm


def download_val_data(name: str, root: str = None):
    # modified from oai _download clip function

    if root is None:
        raise RuntimeError(f"{root} must not be None")

    cloud_checkpoints = {
        "shard_00000000.tar": {
            "url": "https://huggingface.co/datasets/mlfoundations/open_lm_example_data/resolve/main/example_train_data/shard_00000000.tar",
            "sha256": "f53d2cbaf5ffc0532aaefe95299e1ef5e1641f0a1cbf7ae12642f71eaa892d30",
        },
    }

    if name not in cloud_checkpoints:
        raise ValueError(
            f"unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}"
        )

    os.makedirs(root, exist_ok=True)

    expected_sha256 = cloud_checkpoints[name]["sha256"]
    download_target = os.path.join(root, f"{name}")
    url = cloud_checkpoints[name]["url"]

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def download_dl_test_data(root: str = "./tests/assets"):
    """Downloads test files if the data doesn't exist in HF cache."""

    snapshot_args = dict(
        repo_id="mlfoundations/open_lm_test_data_v2",
        local_dir=root,
        repo_type="dataset",
    )

    snapshot_download(**snapshot_args)


def make_tar(tar_num, num_lines, source_num=0, dir_name=None):
    fname = lambda i: '%08d_chunk_%s.json' % (tar_num, i)
    
    if dir_name != None:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        
    tarname = os.path.join(dir_name, '%08d.tar' % tar_num)
    if os.path.exists(tarname):
        return

    fnames = []
    with tarfile.open(tarname, 'w') as tar:
        for line in range(num_lines):
            base_line = [666 for _ in range(2049)]
            base_line[0] = source_num
            base_line[1] = tar_num
            base_line[2] = line
            this_file = fname(line)
            with open(this_file, 'w') as f:
                f.write(json.dumps(base_line))
            tar.add(this_file)
            fnames.append(this_file)
    
        
    for f in fnames:
        try:
            os.unlink(f)
        except:
            pass

        
def make_source(source_num, size_per_tar, total_size):
    num_tars = total_size // size_per_tar
    if total_size % size_per_tar != 0:
        num_tars += 1
    
    base_dir = "tests/assets"
    os.makedirs(base_dir, exist_ok=True)

    num_remaining = total_size    
    for tar_num in range(num_tars):
        this_tar = min(num_remaining, size_per_tar)		        
        make_tar(tar_num, this_tar, source_num=source_num, dir_name="tests/assets/source_id_%02d" % source_num)
        num_remaining -= this_tar

    args = ["--data-dir", "tests/assets/source_id_%02d" % source_num]
    mwm.main(args)


def make_fake_tarfiles():
    """ Makes sources for dataloader tests.
    Running main will...
    - generate 2 sources, titled 'source_id_00', 'source_id_01'
    - each source has 7 .tar files, each with 100 sequences (except the last which has 66)
    - each sequence has the first three tokens as (source_num, tar_num, line_num)

    This way we'll be able to identify where each sequence came from when we test...
    """
    for i in range(2):
        make_source(i, 100, 666)