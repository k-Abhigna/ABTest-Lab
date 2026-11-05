"""Analyse an experiment CSV (or the built-in simulated one) and write a readout.

    python -m abtestlab.run                      # simulated checkout test
    python -m abtestlab.run --srm-bug            # same test with a broken split
    python -m abtestlab.run --csv my_test.csv    # your own data
CSV columns: group (control/treatment), converted, revenue, pre_revenue, refunded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze import analyze, format_report
from .simulate import simulate_experiment

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--srm-bug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.csv:
        df, name = pd.read_csv(args.csv), args.csv.stem
    else:
        df = simulate_experiment(seed=args.seed, srm_bug=args.srm_bug)
        name = "checkout_test_srm_bug" if args.srm_bug else "checkout_test"
        (ROOT / "data").mkdir(exist_ok=True)
        df.to_csv(ROOT / "data" / f"{name}.csv", index=False)
    result = analyze(df)
    report = format_report(result)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"readout_{name}.md").write_text(report, encoding="utf-8")
    print(report)
    return result


if __name__ == "__main__":
    main()
