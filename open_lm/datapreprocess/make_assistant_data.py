import jsonlines
import glob
import tiktoken
import os
import threading
from webdataset import ShardWriter
import random
import time
import boto3
import io
import zstandard as zstd
from contextlib import contextmanager
import argparse
from pathlib import Path
from transformers import GPTNeoXTokenizerFast


QUEUE_MAX = 10000
BUFFER_MIN = 1000
BUFFER_MAX = 200000
CHUNK_SIZE = 2048 + 1
SHARD_SIZE = 267
SLEEP_TIME = 1
S3_BUCKET = "s-laion"
S3_SUFFIX = "validation_data_tokenized/"
S3_BASE = f"s3://<your-bucket>"

eot_token = "<|endoftext|>"
pad_token = "<|pad|>"


def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:12d}", "txt": str(chunk)})


def upload_to_s3_and_remove(fname):
    fname_split = fname.split("/")
    s3_path = S3_BASE + fname_split[-2] + "/" + fname_split[-1]
    cmd = f"aws s3 cp {fname} {s3_path} && rm {fname}"
    print("COMMAND:", cmd)
    os.system(cmd)


@contextmanager
def get_item_reader(file_name):
    if file_name.endswith(".jsonl"):
        with jsonlines.open(file_name) as reader:
            yield reader
    else:
        dctx = zstd.ZstdDecompressor()
        with open(file_name, "rb") as compressed_file:
            with dctx.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                    with jsonlines.Reader(text_reader) as jsonl_reader:
                        yield jsonl_reader


def process_files(file_list, buffer, enc, buffer_lock):
    remaining_tokens = []
    queue = []

    def dump_queue_to_buffer():
        with buffer_lock:
            while queue:
                buffer.append(queue.pop(0))

    for file_name in file_list:
        print("Processing", file_name)

        with get_item_reader(file_name) as item_reader:
            for item in item_reader:
                string = item["text"]
                try:
                    tokens = remaining_tokens + enc(string) + [eot_token]
                    remaining_tokens = []
                except:
                    print("Failed to encode string.")
                    continue

                for i in range(0, len(tokens), CHUNK_SIZE):
                    chunk = tokens[i : i + CHUNK_SIZE]
                    if len(chunk) < CHUNK_SIZE:
                        remaining_tokens = chunk
                    else:
                        if len(buffer) > BUFFER_MAX:
                            time.sleep(1)
                            continue

                        if buffer_lock.locked():
                            if len(queue) < QUEUE_MAX:
                                queue.append(chunk)
                            else:
                                time.sleep(1)
                        else:
                            if queue:
                                dump_queue_to_buffer()
                            with buffer_lock:
                                buffer.append(chunk)


def consumer(my_id, output_dir, threads, buffer, buffer_lock, num_consumers, upload_to_s3=False):
    output_directory = f"{output_dir}/{CHUNK_SIZE - 1}-v1/{my_id}"
    os.makedirs(output_directory, exist_ok=True)
    shard_writer = ShardWriter(os.path.join(output_directory, "shard-%07d.tar"), maxcount=SHARD_SIZE)

    chunks = []

    start_time = time.time()

    while any(t.is_alive() for t in threads):
        time.sleep(SLEEP_TIME)
        with buffer_lock:
            lenb = len(buffer)
            print("Length of buffer", lenb)
            if lenb >= BUFFER_MIN:
                while buffer and len(chunks) < SHARD_SIZE:
                    random_index = random.randint(0, len(buffer) - 1)
                    chunks.append(buffer[random_index])
                    buffer.pop(random_index)  # Remove the selected element

        if len(chunks) == SHARD_SIZE:
            print(f"I am {my_id} and I am writing a shard.", len(buffer))
            write_to_shard(chunks, shard_writer)
            # print("FNAME", shard_writer.fname)
            chunks = []
            time_for_shard = time.time() - start_time
            print("shards / s", num_consumers / time_for_shard)
            print("tokens / s", num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard)
            print(
                "hours req for 1.2T tokens",
                1_200_000_000_000 / (num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard) / 3600,
            )

            start_time = time.time()

    # Process the remaining items in the buffer after all threads have completed
    while buffer:
        with buffer_lock:
            while buffer and len(chunks) < SHARD_SIZE:
                random_index = random.randint(0, len(buffer) - 1)
                chunks.append(buffer[random_index])
                buffer.pop(random_index)  # Remove the selected element

        write_to_shard(chunks, shard_writer)
        chunks = []


def tokenize_eleutherai(tokenizer, string):
    return tokenizer(string).input_ids


def main(
    input_files,
    output_dir,
    tokenizer="EleutherAI/gpt-neox-20b",
    num_workers=32,
    num_consumers=8,
    upload_to_s3=False,
):
    os.makedirs(f"{output_dir}/tars-{CHUNK_SIZE - 1}-v1", exist_ok=True)

    input_files = [glob.glob(input_file) for input_file in input_files]
    input_files = [x for y in input_files for x in y]

    # Shuffle the input files
    random.shuffle(input_files)

    print("Input files", input_files)

    enc = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    tokenize = lambda x: tokenize_eleutherai(enc, x)
    buffer = []  # Use list instead of queue.Queue
    buffer_lock = threading.Lock()

    files_per_worker = len(input_files) // num_workers
    threads = []
    for i in range(num_workers):
        start = i * files_per_worker
        end = (i + 1) * files_per_worker if i < num_workers - 1 else len(input_files)
        t = threading.Thread(
            target=process_files,
            args=(input_files[start:end], buffer, tokenize, buffer_lock),
        )
        t.start()
        threads.append(t)

    consumer_threads = []
    for i in range(num_consumers):
        t = threading.Thread(
            target=consumer,
            args=(
                i,
                output_dir,
                threads,
                buffer,
                buffer_lock,
                num_consumers,
                upload_to_s3,
            ),
        )
        t.start()
        consumer_threads.append(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-consumers", type=int, default=8)
    parser.add_argument("--upload-to-s3", action="store_true")

    args = parser.parse_args()

    main(
        args.input_files,
        args.output_dir,
        args.tokenizer,
        args.num_workers,
        args.num_consumers,
        args.upload_to_s3,
    )
