from data.dependencies.StandardIrregularCSV.dataset import StandardIrregularDataset
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data(StandardIrregularDataset):
    """
    EPA Air Quality dataset (irregular time series)

    - number of variables: 4
    - number of records: 8
    - split_strategy: time
    - recommended: seq_len=336, pred_len=24, batch_size=32
    """

    CSV_FILENAME = "EPA-Air_standard.csv"
