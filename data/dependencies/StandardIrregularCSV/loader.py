"""
Loader for standard irregular CSV format used by CESNET, EPA_AIR, FNSPID, PAMP2.

CSV schema: record_id, time, value_0, mask_0, value_1, mask_1, ...

Returns list of (record_id, tt, vals, mask) tuples where:
- tt: [T] float tensor of time stamps
- vals: [T, D] float tensor of values
- mask: [T, D] float tensor of observation masks
"""

import csv
from collections import defaultdict

import torch


class StandardIrregularCSV:
    def __init__(self, csv_path: str, device=torch.device("cpu")):
        self.data = self._load(csv_path, device)

    def _load(self, csv_path, device):
        # group rows by record_id, preserving order
        records = defaultdict(list)
        record_order = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # parse header to find value/mask column pairs
            value_cols = []
            mask_cols = []
            for i, col in enumerate(header):
                if col.startswith("value_"):
                    value_cols.append(i)
                elif col.startswith("mask_"):
                    mask_cols.append(i)

            n_vars = len(value_cols)
            assert n_vars == len(mask_cols), f"Mismatched value/mask columns: {n_vars} vs {len(mask_cols)}"

            n_cols_expected = len(header)
            n_skipped = 0
            for row in reader:
                if len(row) != n_cols_expected:
                    n_skipped += 1
                    continue
                rid = row[0]
                if rid not in records:
                    record_order.append(rid)

                t = float(row[1])
                vals = []
                masks = []
                for vi, mi in zip(value_cols, mask_cols):
                    v = float(row[vi]) if row[vi] != "" else 0.0
                    m = float(row[mi]) if row[mi] != "" else 0.0
                    vals.append(v)
                    masks.append(m)

                records[rid].append((t, vals, masks))
            if n_skipped:
                import warnings
                warnings.warn(f"StandardIrregularCSV: skipped {n_skipped} malformed row(s) in {csv_path}")

        # convert to tensors
        data = []
        for rid in record_order:
            rows = records[rid]
            rows.sort(key=lambda r: r[0])  # sort by time
            tt = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)
            vals = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=device)
            mask = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=device)
            data.append((rid, tt, vals, mask))

        return data

    @property
    def n_variables(self):
        if self.data:
            return self.data[0][2].shape[-1]
        return 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def split_by_time(total_dataset, train_ratio=0.6, val_ratio=0.2):
    """
    Split dataset by global time range into train/val/test.
    Follows the same logic as irregular_data_loader.split_by_time.
    """
    all_times = [data[1] for data in total_dataset if data[1] is not None and data[1].numel() > 0]
    if not all_times:
        return [], [], []

    global_min_time = min([torch.min(t) for t in all_times])
    global_max_time = max([torch.max(t) for t in all_times])
    time_span = global_max_time - global_min_time

    train_cutoff = global_min_time + time_span * train_ratio
    val_cutoff = global_min_time + time_span * (train_ratio + val_ratio)

    train_data, val_data, test_data = [], [], []

    for record_id, tt, vals, mask in total_dataset:
        if tt is None or tt.numel() == 0:
            continue

        train_mask = tt < train_cutoff
        if torch.any(train_mask):
            train_data.append((record_id, tt[train_mask], vals[train_mask], mask[train_mask]))

        val_mask = (tt >= train_cutoff) & (tt < val_cutoff)
        if torch.any(val_mask):
            val_data.append((record_id, tt[val_mask], vals[val_mask], mask[val_mask]))

        test_mask = tt >= val_cutoff
        if torch.any(test_mask):
            test_data.append((record_id, tt[test_mask], vals[test_mask], mask[test_mask]))

    return train_data, val_data, test_data


def split_by_sample(total_dataset, train_ratio=0.6, val_ratio=0.2, random_state=42):
    """
    Sample-based split (by record_id) matching t-PatchGNN behavior.
    Uses sklearn train_test_split (80/20, then 75/25 → 60/20/20 effective).
    """
    from sklearn import model_selection
    # 80/20 first split
    seen_data, test_data = model_selection.train_test_split(
        list(total_dataset), train_size=train_ratio + val_ratio, random_state=random_state, shuffle=True
    )
    # 75/25 inner split → 60/20
    train_ratio_inner = train_ratio / (train_ratio + val_ratio)
    train_data, val_data = model_selection.train_test_split(
        seen_data, train_size=train_ratio_inner, random_state=random_state, shuffle=False
    )
    return train_data, val_data, test_data


def time_based_chunk(data, history, pred_len, single_chunk_on_sample=False):
    """
    Time-based chunking: slides a window over each record using time values.

    Follows the same logic as irregular_data_loader.unified_time_chunk:
    - stride = int(history / 20.0)
    - For each window [st, st+history+pred_len), observations with tt < st+history
      are history (x), observations with tt >= st+history are future (y).
    - Time marks are normalized to [0, 1] over the full window span.
    - If single_chunk_on_sample=True, only produce one chunk per record (st=0)
      — matches t-PatchGNN P12/MIMIC behavior.

    Returns list of dicts with keys: x, x_mark, x_mask, y, y_mark, y_mask, sample_ID.
    """
    stride = max(1, int(history / 20.0))
    chunk_data = []
    sample_id = 0

    for record_id, tt, vals, mask in data:
        if tt.numel() == 0:
            continue

        t_max = int(tt.max())

        if single_chunk_on_sample:
            start_times = [0]
        else:
            start_times = range(0, t_max - history + 1, stride)

        for st in start_times:
            et = st + history + pred_len
            if et >= t_max:
                idx_mask = (tt >= st) & (tt <= et)
            else:
                idx_mask = (tt >= st) & (tt < et)

            selected_tt = tt[idx_mask]
            hist_mask_in_selected = selected_tt < (st + history)
            future_mask_in_selected = selected_tt >= (st + history)

            if not (torch.any(hist_mask_in_selected) and torch.any(future_mask_in_selected)):
                continue

            idx = torch.where(idx_mask)[0]
            shifted_tt = tt[idx] - st

            # split into x (history) and y (future)
            n_hist = hist_mask_in_selected.sum()

            x_tt = shifted_tt[:n_hist]
            y_tt = shifted_tt[n_hist:]
            x_vals = vals[idx[:n_hist]]
            y_vals = vals[idx[n_hist:]]
            x_mask = mask[idx[:n_hist]]
            y_mask = mask[idx[n_hist:]]

            # normalize time marks to [0, 1]
            t_total = float(history + pred_len)
            x_mark = x_tt / t_total
            y_mark = y_tt / t_total

            chunk_data.append({
                "sample_ID": sample_id,
                "x_mark": x_mark,
                "y_mark": y_mark,
                "x": x_vals,
                "y": y_vals,
                "x_mask": x_mask,
                "y_mask": y_mask,
            })
            sample_id += 1

    return chunk_data
