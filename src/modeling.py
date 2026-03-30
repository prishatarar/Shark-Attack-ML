"""Model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def make_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)


def _binary_y(y: pd.Series) -> pd.Series:
    return y.map({"N": 0, "Y": 1})


def evaluate_classifier(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_train_pred = fitted.predict(X_train)
    y_test_pred = fitted.predict(X_test)

    results: dict[str, Any] = {
        "model": fitted,
        "train_report": classification_report(y_train, y_train_pred, output_dict=True, zero_division=0),
        "test_report": classification_report(y_test, y_test_pred, output_dict=True, zero_division=0),
    }

    if hasattr(fitted, "predict_proba"):
        classes = list(fitted.classes_)
        positive_class = "Y" if "Y" in classes else 1
        positive_idx = classes.index(positive_class)
        train_proba = fitted.predict_proba(X_train)[:, positive_idx]
        test_proba = fitted.predict_proba(X_test)[:, positive_idx]

        if positive_class == "Y":
            y_train_binary = _binary_y(y_train)
            y_test_binary = _binary_y(y_test)
        else:
            y_train_binary = y_train
            y_test_binary = y_test

        results["train_pr_auc"] = average_precision_score(y_train_binary, train_proba)
        results["test_pr_auc"] = average_precision_score(y_test_binary, test_proba)

    return results


def summarize_results(results_by_name: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, metrics in results_by_name.items():
        test_report = metrics["test_report"]
        train_report = metrics["train_report"]
        rows.append(
            {
                "model": name,
                "test_accuracy": test_report["accuracy"],
                "test_f1_fatal": test_report.get("Y", {}).get("f1-score", np.nan),
                "test_f1_nonfatal": test_report.get("N", {}).get("f1-score", np.nan),
                "test_pr_auc": metrics.get("test_pr_auc", np.nan),
                "train_accuracy": train_report["accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values(["test_f1_fatal", "test_pr_auc"], ascending=False)


def get_baseline_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(class_weight="balanced", random_state=42),
    }


def cross_validate_pr_auc(model: Any, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, random_state: int = 42) -> tuple[float, float]:
    y_binary = _binary_y(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        classes = list(fitted.classes_)
        positive_class = "Y" if "Y" in classes else 1
        positive_idx = classes.index(positive_class)
        proba = fitted.predict_proba(X.iloc[val_idx])[:, positive_idx]
        target = y_binary.iloc[val_idx] if positive_class == "Y" else y.iloc[val_idx]
        scores.append(average_precision_score(target, proba))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
