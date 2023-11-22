import os


def test_tokenize_shuffle_simple():
    exit_value = os.system(
        "python datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3/ --content_key content --subset 10 --output test_output/"
    )
    os.system("rm -rf test_output/")
    assert exit_value == 0


def test_tokenize_shuffle_force_parallelism():
    exit_value = os.system(
        "python datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3/ --content_key content --subset 10 --output test_output/ --force_parallelism 16"
    )
    os.system("rm -rf test_output/")
    assert exit_value == 0


def test_tokenize_shuffle_s3_write():
    exit_value = os.system(
        "python datapreprocess/ray/tokenize_shuffle.py --input s3://dcnlp-west-test/tokenize_shuffle_test/C4_V3/ --content_key content --subset 10 --output s3://dcnlp-hub/test_output/ --force_parallelism 16"
    )
    os.system("aws s3 rm --recursive s3://dcnlp-hub/test_output/")
    assert exit_value == 0
