from huggingface_hub import snapshot_download


def main():

    snapshot_args = dict(
        repo_id="mlfoundations/open_lm_test_data",
        local_dir="tests/assets",
        local_dir_use_symlinks=False,
        repo_type="dataset",
    )

    snapshot_download(**snapshot_args)


if __name__ == "__main__":
    main()