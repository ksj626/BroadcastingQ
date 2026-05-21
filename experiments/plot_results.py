from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_eval_metrics(path: Path) -> tuple[list[int], list[float]]:
    steps = []
    returns = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            returns.append(float(row["eval_return_mean"]))
    return steps, returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot eval returns from one or more runs.")
    parser.add_argument("runs", nargs="+", help="Run output directories.")
    parser.add_argument("--out", default="outputs/eval_returns.png", help="Output image path.")
    args = parser.parse_args()

    for run in args.runs:
        run_dir = Path(run)
        steps, returns = read_eval_metrics(run_dir / "eval_metrics.csv")
        plt.plot(steps, returns, label=run_dir.name)
    plt.xlabel("Environment steps")
    plt.ylabel("Mean eval return")
    plt.legend()
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)


if __name__ == "__main__":
    main()
