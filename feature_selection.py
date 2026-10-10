"""Leakage-aware F-ANOVA feature selection."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

import config as cfg


def select_features(X: pd.DataFrame, y: np.ndarray, k=cfg.TOP_ACADEMIC_FEATURES,
                    include_personality=True):
    """Select top-k non-personality features by F-score plus optional traits.

    Call this function on a training partition only. ``y`` is continuous CGPA.
    When ``include_personality`` is False, the five Big-Five features are
    excluded to provide a controlled personality-feature ablation.
    Returns (selected DataFrame, all feature F-scores).
    """
    cols = list(X.columns)
    f_scores, _ = f_regression(X, y)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf, posinf=np.finfo(float).max)
    scores = pd.Series(f_scores, index=cols).sort_values(ascending=False)

    personality = [c for c in cfg.PERSONALITY_TRAITS if c in cols]
    non_personality = [c for c in scores.index if c not in cfg.PERSONALITY_TRAITS]
    selected = non_personality[:k]
    if include_personality:
        selected = selected + personality
    return X[selected], scores
