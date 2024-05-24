import json
import os
import pytest
import webdataset as wds


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    os.system("rm -rf test_output/")
    os.system("rm -rf test_input/")
    os.system("aws s3 rm --recursive s3://dcnlp-west-test/tokenize_shuffle_test_output/simple/")


def test_tokenize_shuffle_simple():
    content_len = 2048
    NUM_TOKENS = 86058
    NUM_PAGES = 160
    NUM_JSONLS = 16
    EOS = 1
    PAD = 0

    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --output test_output/ --seqlen {content_len}"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    eos_tokens = 0
    padded_sequences = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
        eos_tokens += x["json.gz"].count(EOS)
        padded_sequences += 1 if x["json.gz"][-1] == PAD else 0

    # assert total == NUM_TOKENS
    assert eos_tokens == NUM_PAGES
    assert padded_sequences == NUM_JSONLS

    with open("test_output/manifest.jsonl", "rb") as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]

    # assert out[0]["shard"] == "00000001"
    # assert out[0]["num_sequences"] == NUM_TOKENS // (content_len + 1)


def test_tokenize_shuffle_overide_eos_and_pad():
    content_len = 2048
    NUM_TOKENS = 86058
    NUM_PAGES = 160
    NUM_JSONLS = 16
    EOS = 1
    PAD = 0

    # Swap the identity of EOS and PAD special tokens to test whether --eos_overwrite and --pad_overwrite flags work correctly.
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --output test_output/ --seqlen {content_len} --eos_overwrite {EOS} --pad_overwrite {PAD}"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    eos_tokens = 0
    padded_sequences = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
        eos_tokens += x["json.gz"].count(EOS)
        padded_sequences += 1 if x["json.gz"][-1] == PAD else 0

    # assert total == NUM_TOKENS
    assert eos_tokens == NUM_PAGES
    assert padded_sequences == NUM_JSONLS

    with open("test_output/manifest.jsonl", "rb") as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]

    # assert out[0]["shard"] == "00000001"
    # assert out[0]["num_sequences"] == NUM_TOKENS // (content_len + 1)


@pytest.mark.parametrize("content_key,NUM_TOKENS", [("npy", 4860228), ("txt", 24588), ("json:duration", 8196)])
def test_tokenize_shuffle_tar(content_key, NUM_TOKENS):
    content_len = 2048

    params = f"--content_key {content_key}"
    if content_key == "npy":
        params += " --pretokenized"

    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/webvid_tiny/ {params} --output test_output/ --seqlen {content_len}"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
    assert total == NUM_TOKENS


def test_tokenize_shuffle_simple_do_sample():
    content_len = 2048
    NUM_TOKENS = 32784
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --output test_output/ --seqlen {content_len} --do_sample"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
    assert total == NUM_TOKENS


@pytest.mark.s3
def test_tokenize_shuffle_s3_write():
    content_len = 2048
    NUM_TOKENS = 86058
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --seqlen {content_len} --output s3://dcnlp-west-test/tokenize_shuffle_test_output/simple/"
    )
    os.system("aws s3 sync  s3://dcnlp-west-test/tokenize_shuffle_test_output/simple/ test_output/")
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
    assert total == NUM_TOKENS
    assert exit_value == 0

    with open("test_output/manifest.jsonl", "rb") as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]

    assert out[0]["shard"] == "00000001"
    assert out[0]["num_sequences"] == NUM_TOKENS // (content_len + 1)


def test_tokenize_shuffle_local_read_local_write():
    content_len = 2048
    NUM_TOKENS = 24508089
    # download a small test json file and store at ./test_input
    os.system("mkdir test_input")
    os.system("mkdir test_output")
    os.system(
        "wget -O ./test_input/wikipedia_sample.jsonl https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/wikipedia_sample.jsonl"
    )
    # run tokenize script
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input ./test_input --content_key text --seqlen {content_len} --output ./test_output/"
    )
    tars = [os.path.join("test_output", fname) for fname in os.listdir("test_output") if fname.endswith(".tar")]
    total = 0
    for tar in tars:
        ds = wds.WebDataset(tar).decode()
        for x in ds:
            assert len(x["json.gz"]) == content_len + 1
            total += len(x["json.gz"])
    assert total == NUM_TOKENS
    assert exit_value == 0
