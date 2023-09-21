import jsonlines
import glob
import tiktoken
import os
import io
import tempfile
import fsspec
import tarfile
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
from braceexpand import braceexpand

import numpy as np


# VOCAB_SIZE = 1024
VOCAB_SIZE = 16384
QUEUE_MAX = 10000
BUFFER_MIN = 100000
BUFFER_MAX = 200000
CHUNK_SIZE = 16 * (256 + 1) + 1
SHARD_SIZE = 8192
SLEEP_TIME = 1
S3_BUCKET = 'stability-west'
S3_SUFFIX = 'pretraining_data/'
# S3_BASE = f's3://stability-west/acav/openlm_tokens/{CHUNK_SIZE - 1}-v1/'
S3_BASE = f's3://stability-west/webvid-10M_openlm_tokens/{CHUNK_SIZE - 1}-v1/'


def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:12d}", "txt": str(chunk)})

def upload_to_s3_and_remove(fname):
    fname_split = fname.split('/')
    s3_path = S3_BASE + fname_split[-2] + '/' + fname_split[-1]
    cmd = f'aws s3 cp {fname} {s3_path} && rm {fname}'
    print('COMMAND:', cmd)
    os.system(cmd)

@contextmanager
def get_item_reader(file_name):
    if file_name.endswith('.jsonl'):
        with jsonlines.open(file_name) as reader:
            yield reader
    else:
        dctx = zstd.ZstdDecompressor()
        with open(file_name, 'rb') as compressed_file:
            with dctx.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(reader, encoding='utf-8') as text_reader:
                    with jsonlines.Reader(text_reader) as jsonl_reader:
                        yield jsonl_reader

def process_files(file_list, buffer, buffer_lock):
    remaining_tokens = []
    queue = []

    def dump_queue_to_buffer():
        with buffer_lock:
            while queue:
                buffer.append(queue.pop(0))

    folder = "/".join(file_list[0].split("/")[0:-1])
    fs, output_path = fsspec.core.url_to_fs(folder)

    for file_name in file_list:
        print('Processing', file_name)

        try:
            with tempfile.TemporaryDirectory() as tempdir:
                tar_bytes = io.BytesIO(fs.open(file_name).read())
                with tarfile.open(fileobj=tar_bytes) as tar:
                    tar.extractall(tempdir)

                nps = glob.glob(os.path.join(tempdir, "*.npy")) 
                total_tokens = []
                for np_arr in nps:
                    vid = np.load(np_arr)
                    # Append EOF token
                    vid = np.hstack([vid, np.full((vid.shape[0], 1), VOCAB_SIZE)])
                    vid = vid.reshape(-1)
                    total_tokens.append(vid)

                for tokens in total_tokens:
                    tokens = tokens.tolist()
                    for i in range(0, len(tokens), CHUNK_SIZE):
                        chunk = tokens[i:i + CHUNK_SIZE]
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
        except FileNotFoundError as e:
            print(f"{file_name} does not exist")



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
            print('Length of buffer', lenb)
            if lenb >= BUFFER_MIN:
                while buffer and len(chunks) < SHARD_SIZE:
                    random_index = random.randint(0, len(buffer) - 1)
                    chunks.append(buffer[random_index])
                    buffer.pop(random_index)  # Remove the selected element
        
        if len(chunks) == SHARD_SIZE:
            print(f'I am {my_id} and I am writing a shard.', len(buffer))
            write_to_shard(chunks, shard_writer)
            if upload_to_s3:
                upload_to_s3_and_remove(shard_writer.fname)
            #print("FNAME", shard_writer.fname)
            chunks = []
            time_for_shard = time.time() - start_time
            print('shards / s', num_consumers / time_for_shard)
            print('tokens / s', num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard)
            print('hours req for 1.2T tokens', 1_200_000_000_000 / (num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard) / 3600)
        
            start_time = time.time()

    # Process the remaining items in the buffer after all threads have completed
    while buffer:
        with buffer_lock:
            while buffer and len(chunks) < SHARD_SIZE:
                random_index = random.randint(0, len(buffer) - 1)
                chunks.append(buffer[random_index])
                buffer.pop(random_index)  # Remove the selected element

        write_to_shard(chunks, shard_writer)
        if upload_to_s3:
            upload_to_s3_and_remove(shard_writer.fname)
        chunks = []


def main(input_files, output_dir, num_workers=32, num_consumers=8, upload_to_s3=False):

    os.makedirs(f"{output_dir}/tars-{CHUNK_SIZE - 1}-v1", exist_ok=True)

    if "*" in input_files:
        input_files = [glob.glob(input_file) for input_file in input_files]
        input_files = [x for y in input_files for x in y]
    else:
        input_files = [braceexpand(f) for f in input_files]
        input_files = [x for y in input_files for x in y]

    # Shuffle the input files
    random.shuffle(input_files)

    print('Input files', input_files)

    buffer = []  # Use list instead of queue.Queue
    buffer_lock = threading.Lock()

    files_per_worker = len(input_files) // num_workers
    threads = []
    for i in range(num_workers):
        start = i * files_per_worker
        end = (i + 1) * files_per_worker if i < num_workers - 1 else len(input_files)
        t = threading.Thread(target=process_files, args=(input_files[start:end], buffer, buffer_lock))
        t.start()
        threads.append(t)

    consumer_threads = []
    for i in range(num_consumers):
        t = threading.Thread(target=consumer, args=(i, output_dir, threads, buffer, buffer_lock, num_consumers, upload_to_s3))
        t.start()
        consumer_threads.append(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-consumers", type=int, default=8)
    parser.add_argument("--upload-to-s3", action='store_true')

    args = parser.parse_args()

    main(args.input_files, args.output_dir, args.num_workers, args.num_consumers, args.upload_to_s3)
