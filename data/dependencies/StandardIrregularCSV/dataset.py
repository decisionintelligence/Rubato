"""
Base Dataset class for standard irregular CSV datasets (CESNET, EPA_AIR, FNSPID, PAMP2).
Uses time-based splitting (split_by_time) following the irregular_data_loader.py pattern.
"""

import math
import os

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.dependencies.StandardIrregularCSV.loader import (
    StandardIrregularCSV,
    split_by_time,
    split_by_sample,
    time_based_chunk,
)
from data.dependencies.tsdm.PyOmniTS.tsdmDataset import (
    collate_fn,
    collate_fn_patch,
    collate_fn_tpatch,
)
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class StandardIrregularDataset(Dataset):
    """
    Reusable Dataset class for any dataset stored in the standard irregular CSV format.

    Uses time-based splitting (60%/20%/20% by global time range) and time-based chunking
    with stride = history / 20.

    Subclasses only need to set:
    - CSV_FILENAME: name of the CSV file under dataset_root_path
    """

    CSV_FILENAME: str = None  # override in subclass

    def __init__(
        self,
        configs: ExpConfigs,
        flag: str = "train",
        **kwargs,
    ):
        self.configs = configs
        assert flag in ["train", "val", "test", "test_all"]
        self.flag = flag

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.dataset_root_path = configs.dataset_root_path

        self.N_SAMPLES = None
        self.N_SAMPLES_TRAIN = None
        self.N_SAMPLES_VAL = None
        self.N_SAMPLES_TEST = None

        self._preprocess()
        self._get_sample_index()

    def __getitem__(self, index):
        sample_dict: dict[str, Tensor] = self.data[index]
        sample_dict["sample_ID"] = self.sample_index[index]
        sample_dict["_configs"] = self.configs
        sample_dict["_L_TOTAL"] = self.seq_len + self.pred_len

        return sample_dict

    def __len__(self):
        return len(self.data)

    def _get_csv_path(self):
        return os.path.join(self.dataset_root_path, self.CSV_FILENAME)

    def _preprocess(self):
        if self.configs.task_name not in ["short_term_forecast", "long_term_forecast"]:
            raise NotImplementedError()

        csv_path = self._get_csv_path()
        raw_data = StandardIrregularCSV(csv_path=csv_path)

        # Split strategy: "time" (global time cutoff) or "sample" (by record_id, t-PatchGNN)
        split_strategy = getattr(self.configs, "split_strategy", "time")
        single_chunk = bool(getattr(self.configs, "single_chunk_on_sample", 0))

        if split_strategy == "sample":
            train_data_raw, val_data_raw, test_data_raw = split_by_sample(
                raw_data.data, train_ratio=0.6, val_ratio=0.2
            )
            logger.info(f"[{self.__class__.__name__}] split_by_sample (t-PatchGNN): "
                        f"train_records={len(train_data_raw)}, val_records={len(val_data_raw)}, "
                        f"test_records={len(test_data_raw)}, single_chunk={single_chunk}")
        else:
            train_data_raw, val_data_raw, test_data_raw = split_by_time(
                raw_data.data, train_ratio=0.6, val_ratio=0.2
            )

        history = self.seq_len
        pred_len = self.pred_len

        train_data = time_based_chunk(train_data_raw, history=history, pred_len=pred_len,
                                      single_chunk_on_sample=single_chunk)
        val_data = time_based_chunk(val_data_raw, history=history, pred_len=pred_len,
                                    single_chunk_on_sample=single_chunk)
        test_data = time_based_chunk(test_data_raw, history=history, pred_len=pred_len,
                                     single_chunk_on_sample=single_chunk)

        # Normalization stats from train split only
        norm_method = getattr(self.configs, "normalize_method", "zscore")
        if norm_method == "minmax":
            self.data_min, self.data_max = self._compute_minmax(train_data)
            logger.info(f"[{self.__class__.__name__}] Normalization (minmax): min={self.data_min}, max={self.data_max}")
            self._normalize_minmax(train_data)
            self._normalize_minmax(val_data)
            self._normalize_minmax(test_data)
            # Keep placeholders for compatibility
            self.data_mean = self.data_min
            self.data_std = self.data_max - self.data_min
        else:
            self.data_mean, self.data_std = self._compute_stats(train_data)
            logger.info(f"[{self.__class__.__name__}] Normalization (zscore): mean={self.data_mean}, std={self.data_std}")
            self._normalize_chunks(train_data)
            self._normalize_chunks(val_data)
            self._normalize_chunks(test_data)

        self.N_SAMPLES = len(train_data) + len(val_data) + len(test_data)
        self.N_SAMPLES_TRAIN = len(train_data)
        self.N_SAMPLES_VAL = len(val_data)
        self.N_SAMPLES_TEST = len(test_data)

        logger.info(
            f"[{self.__class__.__name__}] split({split_strategy}): "
            f"train={self.N_SAMPLES_TRAIN}, val={self.N_SAMPLES_VAL}, "
            f"test={self.N_SAMPLES_TEST}"
        )

        if self.flag != "val":
            # determine max irr lengths for padding in collate_fn
            test_all_data = train_data + val_data + test_data
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            self.patch_len_max_irr = 0

            SEQ_LEN = self.configs.seq_len
            PRED_LEN = self.configs.pred_len
            PATCH_LEN = self.configs.patch_len

            for sample in test_all_data:
                if sample["x"].shape[0] > self.seq_len_max_irr:
                    self.seq_len_max_irr = sample["x"].shape[0]
                if sample["y"].shape[0] > self.pred_len_max_irr:
                    self.pred_len_max_irr = sample["y"].shape[0]

                if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                    n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                    n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)

                    patch_i_end_previous = 0
                    for i in range(n_patch):
                        observations = sample["x_mark"] < ((i + 1) * PATCH_LEN / (SEQ_LEN + PRED_LEN))
                        patch_i_end = observations.sum()
                        sample_mask = slice(patch_i_end_previous, patch_i_end)
                        x_patch_i = sample["x"][sample_mask]
                        if len(x_patch_i) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(x_patch_i)
                        patch_i_end_previous = patch_i_end

                    patch_j_end_previous = 0
                    for j in range(n_patch_y):
                        observations = sample["y_mark"] < ((SEQ_LEN + (j + 1) * PATCH_LEN) / (SEQ_LEN + PRED_LEN))
                        patch_j_end = observations.sum()
                        sample_mask = slice(patch_j_end_previous, patch_j_end)
                        y_patch_j = sample["y"][sample_mask]
                        if len(y_patch_j) > self.patch_len_max_irr:
                            self.patch_len_max_irr = len(y_patch_j)
                        patch_j_end_previous = patch_j_end

            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                n_patch: int = math.ceil(SEQ_LEN / PATCH_LEN)
                n_patch_y: int = math.ceil(PRED_LEN / PATCH_LEN)
                self.seq_len_max_irr = max(self.seq_len_max_irr, self.patch_len_max_irr * n_patch)
                self.pred_len_max_irr = max(self.pred_len_max_irr, self.patch_len_max_irr * n_patch_y)

            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
            if self.configs.collate_fn in ["collate_fn_patch", "collate_fn_tpatch"]:
                self.configs.patch_len_max_irr = self.patch_len_max_irr
                logger.debug(f"{self.configs.patch_len_max_irr=}")
            logger.debug(f"{self.configs.seq_len_max_irr=}")
            logger.debug(f"{self.configs.pred_len_max_irr=}")

        if self.flag == "test_all":
            self.data = train_data + val_data + test_data
        elif self.flag == "train":
            self.data = train_data
        elif self.flag == "val":
            self.data = val_data
        elif self.flag == "test":
            self.data = test_data

    @staticmethod
    def _compute_stats(chunks):
        """Compute per-variable mean and std from chunk list (using mask)."""
        all_vals = []
        all_masks = []
        for sample in chunks:
            all_vals.append(torch.cat([sample["x"], sample["y"]], dim=0))
            all_masks.append(torch.cat([sample["x_mask"], sample["y_mask"]], dim=0))
        all_vals = torch.cat(all_vals, dim=0)   # [N_total, D]
        all_masks = torch.cat(all_masks, dim=0) # [N_total, D]

        # masked mean/std per variable
        n_vars = all_vals.shape[-1]
        mean = torch.zeros(n_vars)
        std = torch.ones(n_vars)
        for i in range(n_vars):
            valid = all_vals[:, i][all_masks[:, i] == 1]
            if len(valid) > 0:
                mean[i] = valid.mean()
                std[i] = valid.std()
                if std[i] == 0:
                    std[i] = 1.0
        return mean, std

    @staticmethod
    def _compute_minmax(chunks):
        """Compute per-variable min and max from chunk list (using mask)."""
        if not chunks:
            return torch.zeros(1), torch.ones(1)
        n_vars = chunks[0]["x"].shape[-1]
        data_min = torch.full((n_vars,), float("inf"))
        data_max = torch.full((n_vars,), float("-inf"))
        for sample in chunks:
            all_vals = torch.cat([sample["x"], sample["y"]], dim=0)
            all_masks = torch.cat([sample["x_mask"], sample["y_mask"]], dim=0)
            for i in range(n_vars):
                valid = all_vals[:, i][all_masks[:, i] == 1]
                if len(valid) > 0:
                    data_min[i] = torch.min(data_min[i], valid.min())
                    data_max[i] = torch.max(data_max[i], valid.max())
        # Replace inf for variables with no observations
        data_min[torch.isinf(data_min)] = 0.0
        data_max[torch.isinf(data_max)] = 1.0
        return data_min, data_max

    def _normalize_minmax(self, chunks):
        """In-place min-max normalization to [0, 1]; respects mask (zeros out masked)."""
        scale = (self.data_max - self.data_min).clamp(min=1e-8)
        for sample in chunks:
            sample["x"] = (sample["x"] - self.data_min) / scale
            sample["x"][sample["x_mask"] == 0] = 0.0
            sample["y"] = (sample["y"] - self.data_min) / scale
            sample["y"][sample["y_mask"] == 0] = 0.0

    def _normalize_chunks(self, chunks):
        """In-place z-score normalization of x and y values, respecting mask."""
        for sample in chunks:
            sample["x"] = (sample["x"] - self.data_mean) / self.data_std
            sample["x"][sample["x_mask"] == 0] = 0.0
            sample["y"] = (sample["y"] - self.data_mean) / self.data_std
            sample["y"][sample["y_mask"] == 0] = 0.0

    def _get_sample_index(self):
        sample_index_all = torch.arange(self.N_SAMPLES)
        if self.flag == "train":
            self.sample_index = sample_index_all[: self.N_SAMPLES_TRAIN]
        elif self.flag == "val":
            self.sample_index = sample_index_all[
                self.N_SAMPLES_TRAIN : self.N_SAMPLES_TRAIN + self.N_SAMPLES_VAL
            ]
        elif self.flag == "test":
            self.sample_index = sample_index_all[
                self.N_SAMPLES_TRAIN + self.N_SAMPLES_VAL :
            ]
        elif self.flag == "test_all":
            self.sample_index = sample_index_all
        else:
            raise NotImplementedError(f"Unknown {self.flag=}")
