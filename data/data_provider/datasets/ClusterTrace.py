from data.dependencies.StandardIrregularCSV.dataset import StandardIrregularDataset
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data(StandardIrregularDataset):
    """
    ClusterTrace dataset (irregular time series)

    - number of variables: 3
    - number of records: 3
    - split_strategy: time
    - recommended: seq_len=300, pred_len=30
    """

    CSV_FILENAME = "ClusterTrace_standard.csv"
