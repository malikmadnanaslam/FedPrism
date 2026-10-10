"""Leakage-aware preprocessing for the SAPEx-D dataset.

The released CSV is one-hot encoded. ``prepare`` returns the feature matrix
without fitting any transform. Use ``fit_scale`` *after* each train/validation
split so scaler parameters are learned only from the corresponding training
partition (including inner cross-fitting folds).
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.preprocessing import MinMaxScaler

import config as cfg

TARGETS = ["Exact CGPA", "Letter Grade"]
PERSONALITY = ["Openness", "Conscientiousness", "Extraversion",
               "Agreeableness", "Neuroticism"]


def _load_raw():
    df = pd.read_csv(cfg.DATASET_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def categorize_cgpa(cgpa: pd.Series) -> pd.Series:
    """Map continuous CGPA to the institution-defined performance bands."""
    def _cat(x):
        if cfg.GOOD_LOWER <= x <= cfg.GOOD_UPPER:
            return "Good"
        if cfg.AVG_LOWER <= x < cfg.GOOD_LOWER:
            return "Average"
        if cfg.POOR_LOWER <= x < cfg.AVG_LOWER:
            return "Poor"
        return "Unknown"
    return cgpa.apply(_cat)


def build_features(df: pd.DataFrame):
    """Drop outcome fields and return the 86 encoded predictors."""
    return df.drop(columns=[c for c in TARGETS if c in df.columns])


def prepare():
    """Load records and return *unscaled* X, categories, binary labels, CGPA.

    Returns
    -------
    X : pd.DataFrame
        Encoded predictor matrix; no dataset-wide statistics are fitted.
    y_cat : pd.Series
        Good / Average / Poor categories.
    y_bin : np.ndarray
        0 = Good or Average; 1 = Poor (at-risk).
    cgpa : pd.Series
        Continuous CGPA target used for feature ranking.
    """
    df = _load_raw().dropna().reset_index(drop=True)
    X = build_features(df).copy()
    cgpa = df["Exact CGPA"].astype(float)
    y_cat = categorize_cgpa(cgpa)
    y_bin = np.where(y_cat == "Poor", cfg.ANOMALY_LABEL, cfg.NORMAL_LABEL)
    return X, y_cat, y_bin, cgpa


def fit_scale(X_train: pd.DataFrame, X_other: pd.DataFrame):
    """Fit min--max scaling on training rows only; transform another split.

    Binary one-hot columns are kept as 0/1. All non-Boolean columns (ordinal
    codes and personality scores) are scaled using parameters from X_train.
    Returns two float DataFrames with the original column order.
    """
    train = X_train.astype(float).copy()
    other = X_other.astype(float).copy()
    bool_cols = [c for c in train.columns if is_bool_dtype(train[c].dtype)]
    scale_cols = [c for c in train.columns if c not in bool_cols]

    for c in bool_cols:
        train[c] = train[c].astype(float)
        other[c] = other[c].astype(float)

    if scale_cols:
        scaler = MinMaxScaler()
        train.loc[:, scale_cols] = scaler.fit_transform(
            train.loc[:, scale_cols].astype(float))
        other.loc[:, scale_cols] = scaler.transform(
            other.loc[:, scale_cols].astype(float))

    return train.astype(float), other.astype(float)
