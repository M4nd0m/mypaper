import argparse
import subprocess
import sys


ABLATIONS = {
    "original": {
        "objective_mode": "original",
    },
    "cut_main_no_aux": {
        "objective_mode": "cut_main",
        "lambda_temp": "0.0",
        "lambda_batch": "0.0",
    },
    "cut_main_temp": {
        "objective_mode": "cut_main",
        "lambda_temp": "0.01",
        "lambda_batch": "0.0",
    },
    "cut_main_batch": {
        "objective_mode": "cut_main",
        "lambda_temp": "0.0",
        "lambda_batch": "0.01",
    },
    "cut_main_full": {
        "objective_mode": "cut_main",
        "lambda_temp": "0.01",
        "lambda_batch": "0.01",
    },
}


def get_args():
    parser = argparse.ArgumentParser(description="Run minimal HTNcut ablations.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lambda_cut", type=float, default=0.1)
    parser.add_argument("--lambda_bal", type=float, default=0.005)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--extra_args", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = get_args()
    for name, cfg in ABLATIONS.items():
        cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            args.dataset,
            "--epoch",
            str(args.epoch),
            "--run_tag",
            name,
            "--objective_mode",
            cfg["objective_mode"],
            "--warmup_epochs",
            str(args.warmup_epochs),
            "--eval_interval",
            str(args.eval_interval),
        ]
        if cfg["objective_mode"] == "original":
            cmd.extend(["--lambda_cut", str(args.lambda_cut)])
            cmd.extend(["--lambda_bal", "0.0"])
        else:
            cmd.extend(["--lambda_bal", str(args.lambda_bal)])
        if "lambda_temp" in cfg:
            cmd.extend(["--lambda_temp", cfg["lambda_temp"]])
        if "lambda_batch" in cfg:
            cmd.extend(["--lambda_batch", cfg["lambda_batch"]])
        cmd.extend(args.extra_args)

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
