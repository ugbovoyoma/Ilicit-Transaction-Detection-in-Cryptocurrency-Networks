"""
CLI entry point to run the modular illicit transaction detection pipeline.

Usage:
    python run_pipeline.py              # Runs baselines only
    python run_pipeline.py --deep       # Includes deep models (LSTM + GraphSAGE)
    python run_pipeline.py --no-analysis
"""

from __future__ import annotations

import argparse

from illicit_pipeline.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular illicit transaction detection pipeline")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Include deep learning models (LSTM + GraphSAGE). Skipped by default to keep runtime reasonable.",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip exploratory analysis prints.",
    )
    args = parser.parse_args()

    results = run_full_pipeline(include_deep=args.deep, run_analysis=not args.no_analysis)

    print("\n=== Pipeline Results ===")
    for name, res in results.items():
        print(f"\n{name}")
        if hasattr(res, "metrics"):
            for metric, value in res.metrics.items():
                print(f"  {metric}: {value:.4f}")
        if getattr(res, "report", None):
            print("\nClassification report:")
            print(res.report)


if __name__ == "__main__":
    main()

