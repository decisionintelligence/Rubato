from data.dependencies.StandardIrregularCSV.dataset import StandardIrregularDataset
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)


class Data(StandardIrregularDataset):
    """
    MIMIC-III clinical dataset (irregular time series, standard CSV format).
    - number of variables: 96
    - split_strategy: time
    - recommended: seq_len=24, pred_len=12
    """

    CSV_FILENAME = "mimic_standard.csv"
