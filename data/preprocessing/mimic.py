"""
Preprocess raw MIMIC-III time-series CSV into the `mimic_standard.csv` format
expected by `data/dependencies/StandardIrregularCSV/loader.py`.

Expected raw input layout (under <root>/raw/):
    full_dataset.csv  — the raw MIMIC-III time-series CSV with columns
                        Time, Value.<feature>, Mask.<feature>, ... and an
                        index column for record_id.

Usage:
    python -m data.preprocessing.mimic --root storage/datasets/MIMIC

After running, `<root>/mimic_standard.csv` is written. Move/symlink it to
`storage/datasets/MIMIC/mimic_standard.csv` if not already there.
"""
import argparse
import os
import pandas as pd
import torch


def save_to_standard_csv(records, output_path, feature_names):
    print(f"Writing standard CSV: {output_path}")
    rows = []
    n_features = len(feature_names)

    for record_id, tt, vals, mask in records:
        tt_np = tt.cpu().numpy()
        vals_np = vals.cpu().numpy()
        mask_np = mask.cpu().numpy()

        data = {"record_id": record_id, "time": tt_np}
        for i in range(n_features):
            data[f"value_{i}"] = vals_np[:, i]
            data[f"mask_{i}"] = mask_np[:, i]
        rows.append(pd.DataFrame(data))

    if not rows:
        print("Warning: no records to write.")
        return

    final = pd.concat(rows, ignore_index=True)
    final.to_csv(output_path, index=False)
    print(f"Done. Total {len(final)} rows.")


class MIMIC:
    def __init__(self, root, device=torch.device("cpu")):
        self.root = root
        self.device = device

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    def process(self):
        filename = os.path.join(self.raw_folder, "full_dataset.csv")
        os.makedirs(self.raw_folder, exist_ok=True)
        print(f"Processing {filename}")

        full = pd.read_csv(filename, index_col=0)
        value_cols = [c for c in full.columns if c.startswith("Value")]
        mask_cols = [("Mask" + c[5:]) for c in value_cols]

        patients = []
        for record_id, group in full.groupby(level=0):
            tt = torch.tensor(group["Time"].values).float() / 60.0
            vals = torch.tensor(group[value_cols].values).float()
            mask = torch.tensor(group[mask_cols].values).float()
            patients.append((record_id, tt, vals, mask))

        feature_names = [c.replace("Value.", "") for c in value_cols]
        save_to_standard_csv(
            patients,
            os.path.join(self.raw_folder, "mimic_standard.csv"),
            feature_names,
        )
        print(f"Total records: {len(patients)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="storage/datasets/MIMIC",
        help="Dataset root. Expects raw/full_dataset.csv inside.",
    )
    args = parser.parse_args()
    MIMIC(root=args.root).process()
    print(
        f"\nNext: ensure '{os.path.join(args.root, 'raw', 'mimic_standard.csv')}' is "
        f"moved/copied to '{os.path.join(args.root, 'mimic_standard.csv')}' "
        f"(the path the model loader expects)."
    )


if __name__ == "__main__":
    main()
