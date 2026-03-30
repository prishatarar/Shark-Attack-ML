"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_attacks_data(path: str | Path) -> pd.DataFrame:
    """Load the raw shark attack dataset with the same defaults used in the notebooks."""
    df = pd.read_csv(path, encoding="ISO-8859-1").iloc[:-3, :].copy()
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return df
