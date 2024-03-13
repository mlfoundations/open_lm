import json
import os
import pytest
import webdataset as wds
import numpy as np


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    os.system("rm -rf test_output/")
    os.system("rm -rf test_input/")
    os.system("aws s3 rm --recursive s3://dcnlp-west-test/tokenize_shuffle_test_output/simple/")


def test_tokenize_shuffle_simple():
    content_len = 2048
    NUM_TOKENS = 86058
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --output test_output/ --seqlen {content_len}"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
    # assert total == NUM_TOKENS

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
        params += " --vocab_size 16384"

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


@pytest.mark.parametrize("num_sources", [2, 3])
@pytest.mark.parametrize("generation_length", [1024, 2048, 2500])
def test_mixing_no_sampling(num_sources, generation_length):
    content_len = 2048
    docs_a = 1000
    docs_b = 500
    docs_c = 2000

    # Tokens for gpt-neox tokenizer (default)
    token_a = 247
    token_b = 270
    token_c = 260

    # Store some fake sources in ./test_input
    os.system("mkdir test_input")
    os.system("mkdir test_input/source_a/")
    os.system("mkdir test_input/source_b/")
    os.system("mkdir test_input/source_c/")
    os.system("mkdir test_output")

    for i in range(docs_a // 100):
        with open(f"test_input/source_a/input_{i:08d}.jsonl", "w") as f:
            # This will create 2048 copies of the " a" string
            data = {"text": " " + " ".join(["a" for _ in range(generation_length)])}
            json_string = json.dumps(data)
            for _ in range(100):
                f.write(json_string)
                f.write("\n")

    for i in range(docs_b // 100):
        with open(f"test_input/source_b/input_{i:08d}.jsonl", "w") as f:
            data = {"text": " " + " ".join(["b" for _ in range(generation_length)])}
            json_string = json.dumps(data)
            for _ in range(100):
                f.write(json_string)
                f.write("\n")

    if num_sources == 3:
        for i in range(docs_c // 100):
            with open(f"test_input/source_c/input_{i:08d}.jsonl", "w") as f:
                data = {"text": " " + " ".join(["c" for _ in range(generation_length)])}
                json_string = json.dumps(data)
                for _ in range(100):
                    f.write(json_string)
                    f.write("\n")

    # run tokenize script
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input ./test_input --content_key text --seqlen {content_len} --output ./test_output/"
    )

    tars = [os.path.join("test_output", fname) for fname in os.listdir("test_output") if fname.endswith(".tar")]
    total_a = total_b = total_c = 0
    for tar in tars:
        ds = wds.WebDataset(tar).decode()
        for x in ds:
            assert len(x["json.gz"]) == content_len + 1
            tokens = np.array(x["json.gz"])
            total_a += np.sum(tokens == token_a)
            total_b += np.sum(tokens == token_b)
            total_c += np.sum(tokens == token_c)

    assert total_a == docs_a * generation_length
    assert total_b == docs_b * generation_length
    if num_sources == 3:
        assert total_c == docs_c * generation_length

    assert exit_value == 0


@pytest.mark.parametrize("generation_length", [1024, 2048, 2500])
def test_mixing_sampling(generation_length):
    content_len = 2048
    docs_a = 10000
    docs_b = 10000

    # Tokens for gpt-neox tokenizer (default)
    token_a = 247
    token_b = 270

    # Store some fake sources in ./test_input
    os.system("mkdir test_input")
    os.system("mkdir test_input/source_a/")
    os.system("mkdir test_input/source_b/")
    os.system("mkdir test_output")

    for i in range(docs_a // 100):
        with open(f"test_input/source_a/input_{i:08d}.jsonl", "w") as f:
            # This will create 2048 copies of the " a" string
            data = {"text": " " + " ".join(["a" for _ in range(generation_length)])}
            json_string = json.dumps(data)
            for _ in range(100):
                f.write(json_string)
                f.write("\n")

    for i in range(docs_b // 100):
        with open(f"test_input/source_b/input_{i:08d}.jsonl", "w") as f:
            data = {"text": " " + " ".join(["b" for _ in range(generation_length)])}
            json_string = json.dumps(data)
            for _ in range(100):
                f.write(json_string)
                f.write("\n")

    # run tokenize script
    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input ./test_input --content_key text --seqlen {content_len} --output ./test_output/ --do_sample --default_dataset_yaml ./tests/assets/test_sample.yaml"
    )
    assert exit_value == 0

    tars = [os.path.join("test_output", fname) for fname in os.listdir("test_output") if fname.endswith(".tar")]
    total_a = total_b = 0
    for tar in tars:
        ds = wds.WebDataset(tar).decode()
        for x in ds:
            assert len(x["json.gz"]) == content_len + 1
            tokens = np.array(x["json.gz"])
            total_a += np.sum(tokens == token_a)
            total_b += np.sum(tokens == token_b)

    # Sampling for source a should be 2.0, so it should be exactly 2
    assert total_a == 2 * docs_a * generation_length

    # Source b is sampled with probability 0.5, so the number of documents from source b follows Bin(10000, 0.5).
    # Via (multiplicative) Chernoff bounds, for margin delta the error probability is 2 * exp(-delta**2 * mu / 3)
    # In this case for error probability <= 1e-4, we need delta * mu = sqrt(-3 * ln(0.5e-10) / mu) * mu ~= 386
    # TODO (gsmyrnis): I think you can get a better bound here.
    mixing_error = 386
    assert total_b <= (0.5 * docs_b + mixing_error) * generation_length
    assert total_b >= (0.5 * docs_b - mixing_error) * generation_length
