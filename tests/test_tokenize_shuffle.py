import json
import os
import pytest
import webdataset as wds


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    os.system("rm -rf test_output/")
    os.system("aws s3 rm --recursive s3://dcnlp-west-test/tokenize_shuffle_test_output/simple/")


def test_tokenize_shuffle_simple():
    content_len = 2048
    NUM_TOKENS = 381114

    aws_dir = os.path.expanduser("~/.aws")

    # Download dummy creds if they don't exist
    if not os.path.exists(os.path.join(aws_dir, "credentials")):
        os.makedirs(aws_dir, exist_ok=True)
        os.system(
            f"wget https://gist.githubusercontent.com/Vaishaal/f109bfab6a194a93040ae2a19b6be251/raw/7d8026ae234d77ba1ca29b1f9d114c6780308ae4/dummy_creds -O {aws_dir}/credentials"
        )        

    exit_value = os.system(
        f"python open_lm/datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3_tiny/ --content_key content --output test_output/ --seqlen {content_len}"
    )
    assert exit_value == 0
    ds = wds.WebDataset("test_output/00000001.tar").decode()
    total = 0
    for x in ds:
        assert len(x["json.gz"]) == content_len + 1
        total += len(x["json.gz"])
    assert total == NUM_TOKENS

    with open("test_output/manifest.jsonl", "rb") as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]
    
    assert out[0]["shard"] == "00000001"
    assert out[0]["num_sequences"] == NUM_TOKENS // (content_len + 1)

@pytest.mark.s3
def test_tokenize_shuffle_s3_write():
    content_len = 2048
    NUM_TOKENS = 381114
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
