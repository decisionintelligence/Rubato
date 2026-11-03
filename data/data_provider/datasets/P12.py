from data.dependencies.StandardIrregularCSV.dataset import StandardIrregularDataset
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data(StandardIrregularDataset):
    """
    PhysioNet 2012 Challenge dataset (irregular time series).
    - number of variables: 41
    - split_strategy: time
    - recommended: seq_len=24, pred_len=12
    """

    CSV_FILENAME = "physionet_standard.csv"
