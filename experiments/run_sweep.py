from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small sweep of application-local configs.")
    parser.add_argument("configs", nargs="+", help="YAML config paths to run.")
    args = parser.parse_args()
    for config in args.configs:
        subprocess.run([sys.executable, "main.py", "--config", config], check=True)


if __name__ == "__main__":
    main()
