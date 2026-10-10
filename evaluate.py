"""Evaluation metrics and publication-quality figures."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

import config as cfg


def rmse(y_true, y_score):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_score)) ** 2)))


def mse(y_true, y_score):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_score)) ** 2))


def classification_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_alarm_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) else 0.0,
    }
    if y_score is not None:
        s = np.asarray(y_score, dtype=float)
        m["roc_auc"] = round(float(roc_auc_score(y_true, s)), 4)
        # Legacy Brier-style quantities retained for clients expecting them.
        s01 = (s - s.min()) / (np.ptp(s) + 1e-12)
        m["rmse"] = round(rmse(y_true, s01), 4)
        m["mse"] = round(mse(y_true, s01), 4)
    return m


def calibrate_threshold(y_true, scores):
    """Find a threshold maximizing Youden's J on training or OOF scores.

    The experiment runner passes five-fold out-of-fold training scores, never
    held-out test scores. This avoids test leakage and the degenerate
    predict-all-positive solution that F1 calibration can favor."""
    best_t, best_j = None, -1.0
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 60)))
    for t in candidates:
        pred = (scores >= t).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200,
})


def plot_confusion_matrices(matrices, names, path):
    """matrices: list of 2x2 confusion matrices; names: model names."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.4))
    if len(matrices) == 1:
        axes = [axes]
    for ax, cm, name in zip(axes, matrices, names):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["On-track (0)", "At-risk (1)"])
        ax.set_yticklabels(["On-track (0)", "At-risk (1)"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(scores, path, top=15):
    s = scores.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh(s.index, s.values, color="#3b6ea5")
    ax.set_xlabel("F-score (ANOVA)")
    ax.set_title("Top contributing features to CGPA")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(metrics_df, path):
    df = metrics_df[["accuracy", "precision", "recall", "f1"]].T
    ax = df.plot(kind="bar", figsize=(6.4, 3.6), rot=0,
                 color=["#8ca9c9", "#6a8fb5", "#4a76a5", "#2b5d8a"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_dp_ae_sensitivity(epsilons, ae_roc_auc, path):
    """Plot held-out AE performance for the separate AE-only DP sweep."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(epsilons, ae_roc_auc, marker="o", color="#2b5d8a")
    ax.set_xlabel("Client-level privacy budget $\\varepsilon$")
    ax.set_ylabel("Autoencoder held-out ROC-AUC")
    ax.set_title("AE-only client-level DP sensitivity")
    ax.text(0.99, 0.02, "Not an end-to-end privacy claim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_framework(path):
    """Schematic of the in-process IID simulation and its limited DP scope."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf1f8", ec="#2b5d8a", fs=8.5):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           fc=fc, ec=ec, lw=1.4)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2, style="-|>"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=12, color="#555", lw=1.3)
        ax.add_patch(a)

    # Simulated IID shards, not actual departments
    for i, name in enumerate(["Simulated shard 1", "Simulated shard 2", "Simulated shard 3", "Simulated shard 4"]):
        box(0.25, 4.4 - i * 1.1, 1.7, 0.85, name)
    box(2.15, 3.0, 2.2, 2.3, "Client-local fitting\nAutoencoder updates\nLocal RF / IF models", fs=7.5)

    # Aggregation server
    box(4.85, 3.4, 2.0, 1.5, "Trusted in-process coordinator\nFedAvg AE (primary unnoised)\nRF / IF scores averaged centrally", fc="#fdeaea", ec="#a52b2b", fs=7.0)

    # Global model + recommendation
    box(7.15, 4.1, 2.5, 1.0, "OOF score stacking\nand held-out evaluation", fc="#eaf6ea", ec="#2e7d32")
    box(7.15, 2.5, 2.5, 1.0, "Candidate prompts\nfor human review", fc="#fff7e0", ec="#b8860b")

    for i in range(4):
        arrow(1.95, 4.8 - i * 1.1, 2.15, 4.3)
    arrow(4.35, 4.15, 4.85, 4.15)
    arrow(6.85, 4.55, 7.15, 4.6)
    arrow(6.85, 3.95, 7.15, 3.4)
    arrow(8.4, 3.5, 8.4, 2.0, style="<|-|>")
    ax.text(8.55, 2.75, "advice", fontsize=7.5, color="#555")

    ax.text(5.8, 2.4, "Separate sensitivity run: client-level DP on AE only; no secure aggregation.",
            ha="center", fontsize=6.5, color="#a52b2b")
    ax.text(5.0, 0.55, "Single-process IID simulation; not an end-to-end private deployment",
            ha="center", fontsize=9.5, style="italic", color="#333")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
