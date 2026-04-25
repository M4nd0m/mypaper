import argparse
import csv
import os
from datetime import datetime
from typing import Dict, Iterable, List

from evaluate import evaluate_clustering
from utils import ensure_dir, resolve_path


CSV_COLUMNS = [
    "run_time",
    "dataset",
    "epoch",
    "evaluated_nodes",
    "ACC",
    "NMI",
    "F1",
    "ARI",
    "prediction_path",
    "label_path",
]


def get_args():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Evaluate one or more clustering prediction files and write the metrics to CSV."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=["school", "dblp", "brain", "patent", "arXivAI", "arXivCS"],
        help="dataset names to evaluate",
    )
    parser.add_argument("--epoch", type=int, default=100, help="training epoch used in prediction filenames")
    parser.add_argument("--data_root", type=str, default=os.path.join(cur_dir, "dataset"), help="dataset root")
    parser.add_argument("--emb_root", type=str, default=os.path.join(cur_dir, "emb"), help="prediction output root")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=os.path.join(cur_dir, "results", "evaluation_results.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append rows to an existing CSV instead of overwriting it",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="skip datasets whose label or prediction file is missing",
    )
    return parser.parse_args()


def iter_eval_rows(args) -> Iterable[Dict[str, object]]:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for dataset in args.datasets:
        label_path = os.path.join(args.data_root, dataset, "node2label.txt")
        pred_path = os.path.join(args.emb_root, dataset, f"{dataset}_TGC_{args.epoch}_pred.txt")

        missing_paths = [path for path in (label_path, pred_path) if not os.path.exists(path)]
        if missing_paths:
            message = f"Missing files for dataset={dataset}: {', '.join(missing_paths)}"
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)

        result = evaluate_clustering(dataset, label_path, pred_path)
        yield {
            "run_time": run_time,
            "dataset": dataset,
            "epoch": int(args.epoch),
            "evaluated_nodes": result["evaluated_nodes"],
            "ACC": f"{result['ACC']:.6f}",
            "NMI": f"{result['NMI']:.6f}",
            "F1": f"{result['F1']:.6f}",
            "ARI": f"{result['ARI']:.6f}",
            "prediction_path": result["prediction_path"],
            "label_path": result["label_path"],
        }


def write_csv(csv_path: str, rows: List[Dict[str, object]], append: bool) -> None:
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    mode = "a" if append else "w"
    write_header = (not append) or (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode, newline="", encoding="utf-8-sig") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = get_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_path(cur_dir, args.data_root)
    args.emb_root = resolve_path(cur_dir, args.emb_root)
    args.csv_path = resolve_path(cur_dir, args.csv_path)

    rows = list(iter_eval_rows(args))
    write_csv(args.csv_path, rows, args.append)

    print(f"Wrote {len(rows)} evaluation row(s) to {args.csv_path}")
    for row in rows:
        print(
            f"{row['dataset']}: ACC={row['ACC']}, NMI={row['NMI']}, "
            f"F1={row['F1']}, ARI={row['ARI']}"
        )


if __name__ == "__main__":
    main()
