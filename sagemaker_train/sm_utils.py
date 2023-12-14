import os


def get_remote_sync(s3_remote_sync):
    if s3_remote_sync is not None:
        return s3_remote_sync
    else:
        return os.environ["S3_REMOTE_SYNC"]
