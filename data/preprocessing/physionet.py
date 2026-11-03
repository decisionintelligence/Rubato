"""
Download and preprocess PhysioNet Challenge 2012 (P12) into the
`physionet_standard.csv` format expected by
`data/dependencies/StandardIrregularCSV/loader.py`.

Downloads sets a, b, c from physionet.org, extracts, parses each patient file,
and writes a single standardized CSV.

Usage:
    python -m data.preprocessing.physionet --root storage/datasets/P12 --download

After running, `<root>/raw/physionet_standard.csv` is written. Move/symlink it
to `storage/datasets/P12/physionet_standard.csv`.

Adapted from https://github.com/rtqichen/time-series-datasets.
"""
import argparse
import os
import tarfile
import torch
from torchvision.datasets.utils import download_url


def save_to_standard_csv(records, output_path, feature_names):
    import pandas as pd

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


class PhysioNet:
    urls = [
        "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
        "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download",
        "https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download",
    ]

    params = [
        "Age", "Gender", "Height", "ICUType", "Weight", "Albumin", "ALP", "ALT",
        "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine", "DiasABP",
        "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
        "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2",
        "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
        "TroponinI", "TroponinT", "Urine", "WBC",
    ]
    params_dict = {k: i for i, k in enumerate(params)}

    def __init__(self, root, quantization=None, device=torch.device("cpu")):
        self.root = root
        self.reduce = "average"
        self.quantization = quantization
        self.device = device

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    def download_and_process(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        all_patients = []

        for url in self.urls:
            filename = url.rpartition("/")[2]
            download_url(url, self.raw_folder, filename, None)
            with tarfile.open(os.path.join(self.raw_folder, filename), "r:gz") as tar:
                tar.extractall(self.raw_folder)

            print(f"Processing {filename}")
            dirname = os.path.join(self.raw_folder, filename.split(".")[0])
            for txtfile in sorted(os.listdir(dirname)):
                record_id = txtfile.split(".")[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()

                prev_time = 0
                tt = [0.0]
                vals = [torch.zeros(len(self.params))]
                mask = [torch.zeros(len(self.params))]
                nobs = [torch.zeros(len(self.params))]
                for line in lines[1:]:
                    time, param, val = line.split(",")
                    time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60.0
                    if self.quantization is not None and self.quantization != 0:
                        time = round(time / self.quantization) * self.quantization

                    if time != prev_time:
                        tt.append(time)
                        vals.append(torch.zeros(len(self.params)))
                        mask.append(torch.zeros(len(self.params)))
                        nobs.append(torch.zeros(len(self.params)))
                        prev_time = time

                    if param in self.params_dict:
                        idx = self.params_dict[param]
                        n = nobs[-1][idx]
                        if self.reduce == "average" and n > 0:
                            prev = vals[-1][idx]
                            vals[-1][idx] = (prev * n + float(val)) / (n + 1)
                        else:
                            vals[-1][idx] = float(val)
                        mask[-1][idx] = 1
                        nobs[-1][idx] += 1
                    else:
                        assert param in ("RecordID", ""), f"Unexpected param {param}"

                tt = torch.tensor(tt)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                all_patients.append((record_id, tt, vals, mask))

        save_to_standard_csv(
            all_patients,
            os.path.join(self.raw_folder, "physionet_standard.csv"),
            self.params,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="storage/datasets/P12",
        help="Dataset root. Raw .tar.gz and extracted files go under <root>/raw/.",
    )
    parser.add_argument(
        "--quantization",
        type=float,
        default=0.0,
        help="Round timestamps to this resolution (hours). 0 = no quantization.",
    )
    args = parser.parse_args()
    PhysioNet(root=args.root, quantization=args.quantization).download_and_process()
    print(
        f"\nNext: ensure '{os.path.join(args.root, 'raw', 'physionet_standard.csv')}' is "
        f"moved/copied to '{os.path.join(args.root, 'physionet_standard.csv')}'."
    )


if __name__ == "__main__":
    main()
