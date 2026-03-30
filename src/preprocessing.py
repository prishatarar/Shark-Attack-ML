"""Dataset preprocessing pipeline."""

from __future__ import annotations

import pandas as pd

from .data import load_attacks_data
from .features import categorize_time, classify_shark, extract_date, get_season, replace_activity
from .mappings import FATAL_MAPPING, HEMISPHERE_MAPPING, OCEAN_MAPPING, SEX_MAPPING, TYPE_MAPPING


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook-style column normalization."""
    df = df.copy()
    df["Type"] = df["Type"].replace(TYPE_MAPPING)
    df["Hemisphere"] = df["Country"].replace(HEMISPHERE_MAPPING)
    df["ocean"] = df["Country"].map(OCEAN_MAPPING)
    df["Sex "] = df["Sex "].replace(SEX_MAPPING)
    df["Fatal"] = df["Fatal (Y/N)"].replace(FATAL_MAPPING)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered columns used in both notebooks."""
    df = df.copy()
    df["Activity"] = df["Activity"].apply(replace_activity)
    df["Classified Species"] = df["Species "].apply(classify_shark)
    df["Date"] = pd.to_datetime(
        df["Case Number"].apply(extract_date), format="%Y.%m.%d", errors="coerce"
    )
    df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).astype("Int64")
    df["time_category"] = df["Time"].apply(categorize_time)
    df["Season"] = df["Date"].apply(get_season)
    return df


def prepare_attacks_dataframe(path: str) -> pd.DataFrame:
    """End-to-end data preparation entrypoint for notebooks and scripts."""
    df = load_attacks_data(path)
    df = clean_columns(df)
    df = add_engineered_features(df)
    return df


def build_model_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the model-ready design matrix from the cleaned dataframe."""
    feature_columns = [
        "Fatal",
        "Type",
        "Activity",
        "Sex ",
        "Hemisphere",
        "ocean",
        "Classified Species",
        "is_weekend",
        "time_category",
        "Season",
    ]
    model_df = df[feature_columns].dropna().copy()
    model_df = model_df[model_df["Fatal"].isin(["Y", "N"])]
    y = model_df["Fatal"]
    X = model_df.drop(columns=["Fatal"])
    X_encoded = pd.get_dummies(
        X,
        columns=[
            "Type",
            "Activity",
            "Sex ",
            "ocean",
            "Classified Species",
            "time_category",
            "Season",
        ],
        drop_first=False,
    )
    return X_encoded, y, model_df
