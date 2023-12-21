import boto3
import os
import tarfile
import glob
import shutil
import json
import random

s3 = boto3.client("s3")
bucket_name, prefix = "dcnlp-hub", "C4_V3_tokenized/"
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
all_files = [obj["Key"] for objects in pages for obj in objects.get("Contents", [])]
random.shuffle(all_files)
all_files = all_files[6:100]


total_tokens = 0
num_errors = 0
for i, file in enumerate(all_files):
    try:
        os.makedirs(f"tmp/token_jsons_{i}", exist_ok=True)
        output_path = f'tmp/{file.split("/")[-1]}'
        s3.download_file(bucket_name, file, output_path)

        with tarfile.open(output_path, "r") as tar:
            tar.extractall(path=f"tmp/token_jsons_{i}/", numeric_owner=True)

        num_tokens = len(glob.glob(f"tmp/token_jsons_{i}/*.json")) * 2048
        for tokens_file in glob.glob(f"tmp/token_jsons_{i}/*.json"):
            with open(tokens_file, "r") as file:
                tokens = json.load(file)
            assert len(tokens) == 2048, "Token length is wrong"
    except:
        print("Error on file:", file)
        num_tokens = 0
        num_errors += 1

    # os.rmdir("/tmp/token_jsons/")
    shutil.rmtree(f"tmp/token_jsons_{i}", ignore_errors=True)
    os.remove(output_path)
    total_tokens += num_tokens
    print(f"Reached tar {i}, Num Tokens = {num_tokens}, Total = {total_tokens/1e9} Billion")


print("Total tokens", total_tokens)
print("Num errors", num_errors)
