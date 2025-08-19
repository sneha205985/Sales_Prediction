from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    best_estimator: Pipeline
    best_params: Dict[str, Any]
    cv_best_score: float
    metrics: Dict[str, float]
    feature_names: List[str]
    model_name: str


def _make_json_safe(value: Any) -> Any:
    """Recursively convert complex objects to JSON-serializable representations."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_make_json_safe(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    # Fallback: string representation
    return str(value)

def _build_pipeline_and_param_grid(random_state: int) -> Tuple[Pipeline, List[Dict[str, Any]]]:
    """Create a single pipeline and a parameter grid spanning multiple models.

    The pipeline consists of optional scaling followed by a 'model' step. The grid
    enumerates LinearRegression, Ridge, Lasso, and RandomForestRegressor with reasonable
    hyperparameters.
    """
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    param_grid: List[Dict[str, Any]] = [
        # Linear Regression
        {
            "scaler": [StandardScaler()],
            "model": [LinearRegression()],
        },
        # Ridge Regression
        {
            "scaler": [StandardScaler()],
            "model": [Ridge()],
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
            "model__solver": ["auto", "svd", "lsqr", "saga"],
        },
        # Lasso Regression
        {
            "scaler": [StandardScaler()],
            "model": [Lasso(max_iter=10000, random_state=random_state)],
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        # Random Forest (no scaling)
        {
            "scaler": ["passthrough"],
            "model": [RandomForestRegressor(random_state=random_state)],
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
    ]

    return pipeline, param_grid


def train_and_evaluate(
    X,
    y,
    *,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = "neg_root_mean_squared_error",
) -> TrainResult:
    """Split data, perform model selection with cross-validation, and evaluate.

    Returns a TrainResult with the fitted best estimator and metrics on the holdout test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline, param_grid = _build_pipeline_and_param_grid(random_state=random_state)

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_estimator: Pipeline = search.best_estimator_

    # Evaluate on the test set
    y_pred = best_estimator.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    model_name = type(best_estimator.named_steps["model"]).__name__

    return TrainResult(
        best_estimator=best_estimator,
        best_params=search.best_params_,
        cv_best_score=float(search.best_score_),
        metrics=metrics,
        feature_names=list(X.columns),
        model_name=model_name,
    )


def save_artifacts(
    result: TrainResult,
    output_dir: str,
    *,
    report_filename: str = "model_report.json",
    model_filename: str = "best_model.joblib",
) -> Tuple[str, str]:
    """Persist the fitted model and a JSON report with metadata and metrics.

    Returns paths to the saved report and model.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, model_filename)
    report_path = os.path.join(output_dir, report_filename)

    joblib.dump(result.best_estimator, model_path)

    report = {
        "model_name": result.model_name,
        "best_params": _make_json_safe(result.best_params),
        "cv_best_score": result.cv_best_score,
        "metrics": result.metrics,
        "feature_names": result.feature_names,
        "model_path": model_path,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path, model_path

