import os
from tqdm import tqdm
import urllib.request
import hashlib
import warnings

def download(name: str, root: str = None):
    # modified from oai _download clip function

    if root is None:
        raise RuntimeError(f"{root} must not be None")

    cloud_checkpoints = {
        "testing_data": {
            # 'url': 'file://' + os.path.abspath('./assets/nsfw_torch.pt'),
            "url": "https://huggingface.co/datasets/mlfoundations/open_lm_example_data/blob/main/example_train_data/shard_00000000.tar",
            "sha256": "ec9fd5ed97b00815e0dbb7af82f528385c659ae7772f4fcaa82b4574a2ace689",
        },
    }

    if name not in cloud_checkpoints:
        raise ValueError(
            f"unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}"
        )

    os.makedirs(root, exist_ok=True)

    expected_sha256 = cloud_checkpoints[name]["sha256"]
    download_target = os.path.join(root, f"{name}.tar")
    url = cloud_checkpoints[name]["url"]

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

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

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target

