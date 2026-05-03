import argparse
import csv
import glob
import os

from utils import ensure_dir, resolve_path


SUMMARY_COLUMNS = [
    "dataset",
    "metrics_csv",
    "nmi_argmax_Q",
    "acc_argmax_Q",
    "nmi_spectral_pi",
    "purity_at_5_pi",
    "ncut_gt_pi",
    "ncut_pred_pi",
    "ncut_pred_over_gt_pi",
    "spec_gap_c",
    "spec_gap_c1",
    "spec_gap_ratio",
]


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Append final trainer metrics to Phase 0 baseline summary.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--metrics_csv", type=str, default="")
    parser.add_argument("--emb_root", type=str, default=os.path.join(cur_dir, "emb"))
    parser.add_argument("--output_path", type=str, default=os.path.join(cur_dir, "diagnostics", "baseline_metrics_summary.md"))
    return parser.parse_args()


def latest_metrics_csv(emb_root: str, dataset: str) -> str:
    pattern = os.path.join(emb_root, dataset, f"{dataset}_TGC_*_metrics.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No metrics CSV found under {pattern}")
    return max(paths, key=os.path.getmtime)


def read_final_row(path: str):
    with open(path, "r", encoding="utf-8") as reader_file:
        rows = list(csv.DictReader(reader_file))
    if not rows:
        raise ValueError(f"No rows in metrics CSV: {path}")
    return rows[-1]


def pick(row, key):
    value = row.get(key, "")
    return value if value not in (None, "") else "nan"


def append_summary(path: str, out_row):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as writer:
        if not exists:
            writer.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
            writer.write("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |\n")
        writer.write("| " + " | ".join(str(out_row[col]) for col in SUMMARY_COLUMNS) + " |\n")


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.output_path = resolve_path(cur_dir, args.output_path)
    metrics_csv = resolve_path(cur_dir, args.metrics_csv) if args.metrics_csv else latest_metrics_csv(args.emb_root, args.dataset)
    row = read_final_row(metrics_csv)
    out_row = {
        "dataset": args.dataset,
        "metrics_csv": metrics_csv,
        "nmi_argmax_Q": pick(row, "nmi_argmax_s"),
        "acc_argmax_Q": pick(row, "acc_argmax_s"),
        "nmi_spectral_pi": pick(row, "nmi_spectral_pi"),
        "purity_at_5_pi": pick(row, "purity_at_5_pi"),
        "ncut_gt_pi": pick(row, "ncut_gt_pi"),
        "ncut_pred_pi": pick(row, "ncut_pred_pi"),
        "ncut_pred_over_gt_pi": pick(row, "ncut_pred_over_gt_pi"),
        "spec_gap_c": pick(row, "spec_gap_c"),
        "spec_gap_c1": pick(row, "spec_gap_c1"),
        "spec_gap_ratio": pick(row, "spec_gap_ratio"),
    }
    append_summary(args.output_path, out_row)
    print(f"updated {args.output_path}")


if __name__ == "__main__":
    main()
