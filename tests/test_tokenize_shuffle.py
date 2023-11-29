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
