"""Ablation study runner for TCDI-Net.

Runs all 10 ablation variants sequentially and collects results into a
summary CSV at ``result/ablation_results.csv``.

Usage:
    python ablation.py /path/to/dataset --dataset cviu17 --epochs 100

All arguments except ``--ablation`` are forwarded to main.py.
"""

import argparse
import subprocess
import sys
import time

import pandas as pd

from iqanet import ABLATION_CONFIGS

VARIANTS = list(ABLATION_CONFIGS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="TCDI-Net ablation study runner")
    parser.add_argument("data", help="path to dataset root")
    parser.add_argument("--dataset", default="cviu17", help="dataset name")
    parser.add_argument("--epochs", type=int, default=100, help="epochs per variant")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dr-mode", action="store_true", help="enable DR mode")
    parser.add_argument("--lr-dir", default=None, help="LR images directory (for DR mode)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=1300)
    parser.add_argument("--th", type=int, default=4)
    parser.add_argument("--comment", default="", help="suffix for checkpoint names")
    parser.add_argument("--variants", nargs="+", default=VARIANTS,
                        help=f"variants to run (default: all {len(VARIANTS)})")
    return parser.parse_args()


def run_variant(variant: str, args: argparse.Namespace) -> dict:
    """Run a single ablation variant via main.py subprocess."""
    cmd = [
        sys.executable, "main.py", args.data,
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--workers", str(args.workers),
        "--train-size", str(args.train_size),
        "--th", str(args.th),
        "--ablation", variant,
        "--comment", f"{args.comment}_ablation_{variant}",
    ]
    if args.dr_mode:
        cmd.append("--dr-mode")
        if args.lr_dir:
            cmd.extend(["--lr-dir", args.lr_dir])

    print(f"\n{'='*60}")
    print(f"Running ablation variant: {variant}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    t_start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t_start

    return {
        "variant": variant,
        "returncode": result.returncode,
        "elapsed_min": round(elapsed / 60, 1),
    }


def main():
    args = parse_args()
    variants = args.variants
    if isinstance(variants, str):
        variants = [variants]

    print(f"Running ablation study: {len(variants)} variants, {args.epochs} epochs each")
    print(f"Variants: {variants}")

    results = []
    for i, variant in enumerate(variants):
        res = run_variant(variant, args)
        results.append(res)
        status = "OK" if res["returncode"] == 0 else f"FAIL (rc={res['returncode']})"
        print(f"[{i+1}/{len(variants)}] {variant}: {status}  ({res['elapsed_min']} min)")

    # Collect per-variant result CSVs
    summary_rows = []
    for variant in variants:
        csv_path = f"result/{args.dataset}_results.csv"
        variant_csv = f"result/{args.dataset}_ablation_{variant}.csv"
        try:
            df = pd.read_csv(csv_path)
            last = df.iloc[-1]
            summary_rows.append({
                "variant": variant,
                "test_PLCC": last["test_PLCC"],
                "test_SRCC": last["test_SRCC"],
                "test_RMSE": last["test_RMSE"],
                "train_PLCC": last["train_PLCC"],
                "train_SRCC": last["train_SRCC"],
                "train_RMSE": last["train_RMSE"],
            })
            # Preserve individual result
            import shutil
            shutil.copy(csv_path, variant_csv)
        except FileNotFoundError:
            print(f"Warning: result CSV not found for variant {variant}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = "result/ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAblation summary saved to {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
