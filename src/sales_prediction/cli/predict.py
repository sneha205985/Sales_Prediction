from __future__ import annotations

import argparse
import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sales using a trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .joblib")
    parser.add_argument("--report-path", type=str, required=True, help="Path to model_report.json for feature names")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv-input", type=str, help="CSV file with columns TV,Radio,Newspaper")
    group.add_argument("--tv", type=float, help="TV spend")
    parser.add_argument("--radio", type=float, help="Radio spend")
    parser.add_argument("--newspaper", type=float, help="Newspaper spend")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save predictions CSV")
    return parser.parse_args()


def _load_feature_names(report_path: str) -> List[str]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    feature_names = report.get("feature_names")
    if not feature_names:
        raise ValueError("feature_names not found in report")
    return feature_names


def main() -> None:
    args = parse_args()
    pipeline = joblib.load(args.model_path)
    feature_names = _load_feature_names(args.report_path)

    if args.csv_input:
        df = pd.read_csv(args.csv_input)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in input CSV: {missing}")
        X = df[feature_names]
        preds = pipeline.predict(X)
        out_df = df.copy()
        out_df["PredictedSales"] = preds
        if args.out:
            out_df.to_csv(args.out, index=False)
            print(f"Saved predictions to: {args.out}")
        else:
            print(out_df.head())
    else:
        if args.tv is None or args.radio is None or args.newspaper is None:
            raise ValueError("For single prediction, provide --tv, --radio, and --newspaper")
        # Build a one-row DataFrame with expected feature names to avoid warnings
        single = {"TV": args.tv, "Radio": args.radio, "Newspaper": args.newspaper}
        X_single = pd.DataFrame([single], columns=feature_names)
        preds = pipeline.predict(X_single)
        print(float(preds[0]))


if __name__ == "__main__":
    main()

