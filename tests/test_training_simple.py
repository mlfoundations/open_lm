from open_lm.main import main


def test_train_simple():
    seq_len = 2048
    num_batches = 5
    batch_size = 1
    # fmt: off
    main([
        "--train-num-samples", str(num_batches * seq_len),
        "--batch-size", str(batch_size),
        "--dataset-type", "synthetic",
        "--model", "open_lm_11m",
        "--epochs", "1",
    ])
    # fmt: on
