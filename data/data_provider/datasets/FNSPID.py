from data.dependencies.StandardIrregularCSV.dataset import StandardIrregularDataset
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data(StandardIrregularDataset):
    """
    FNSPID financial dataset (irregular time series)

    - number of variables: 6
    - number of records: 10
    - split_strategy: time
    - recommended: seq_len=730, pred_len=30, batch_size=32
    """

    CSV_FILENAME = "FNSPID_standard.csv"
