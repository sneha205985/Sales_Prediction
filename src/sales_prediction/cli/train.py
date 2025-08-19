from __future__ import annotations

import argparse
from typing import Optional

from sales_prediction.data import load_advertising_csv, split_features_and_target
from sales_prediction.modeling import save_artifacts, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sales prediction model")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to advertising CSV")
    parser.add_argument("--target-column", type=str, default="Sales", help="Target column name")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save artifacts")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_advertising_csv(args.csv_path)
    X, y = split_features_and_target(df, target_column=args.target_column)

    result = train_and_evaluate(
        X,
        y,
        cv=args.cv,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    report_path, model_path = save_artifacts(result, args.output_dir)

    print(f"Saved report to: {report_path}")
    print(f"Saved model to:  {model_path}")
    print(f"Metrics: {result.metrics}")
    print(f"Best params: {result.best_params}")


if __name__ == "__main__":
    main()

