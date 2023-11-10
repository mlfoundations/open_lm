from open_lm.data import get_wds_dataset
from tests.shared import MockDataArgs


def test_dataloader_no_crash():
    # basic test to make sure the datalaoder does not crash
    args = MockDataArgs()
    di = get_wds_dataset(args, True)

    for _ in di.dataloader:
        pass

    assert True
