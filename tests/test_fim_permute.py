import numpy as np
from open_lm.train import permute_segment


def test_resize_token_embeddings():
    prefix_tok_id = 9999
    suffix_tok_id = 9998
    middle_tok_id = 9997

    test_cases = [
        (np.array(range(100)), 1.0),
        (np.array(range(2000)), 0.0),
        (np.random.choice(2000, 500), 1.0),
        (np.random.choice(9876, 888), 0.0),
        (np.random.choice(9000, 9000), 1.0),
    ]

    for sample, fim_rate in test_cases:
        new_sample = permute_segment(sample, fim_rate, prefix_tok_id, suffix_tok_id, middle_tok_id)

        if fim_rate == 0.0:
            assert (new_sample == sample).all()

        if fim_rate == 1.0:
            prefix_idx = np.argwhere(new_sample == prefix_tok_id)
            suffix_idx = np.argwhere(new_sample == suffix_tok_id)
            middle_idx = np.argwhere(new_sample == middle_tok_id)

            assert len(prefix_idx) == 1
            assert len(suffix_idx) == 1
            assert len(middle_idx) == 1

            assert prefix_idx[0][0] == 0
            assert prefix_idx[0][0] <= suffix_idx[0][0]
            assert suffix_idx[0][0] <= middle_idx[0][0]
            assert middle_idx[0][0] <= new_sample.shape[0]

    # Test that the fim_rate frequency works as expected
    test_cases_2 = []
    n = 10000
    for i in range(n):
        test_cases_2.append(np.random.choice(5000, 5000))

    diff_counter = 0
    for sample in test_cases_2:
        new_sample = permute_segment(sample, 0.2, prefix_tok_id, suffix_tok_id, middle_tok_id)
        if (new_sample == sample).all() == False:
            diff_counter += 1
    fim_changed = diff_counter / n
    assert fim_changed >= 0.19 and fim_changed <= 0.21
