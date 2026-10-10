"""Generate supplementary and evaluation figures from reproducible outputs."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

import config as cfg
import preprocess

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "figure.dpi": 200})
BLUE, RED, GREEN, GREY = "#2b5d8a", "#c0504d", "#2e7d32", "#8ca9c9"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _figure_cgpa(cgpa, y_cat):
    colors = {"Poor": RED, "Average": BLUE, "Good": GREEN}
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.hist(cgpa, bins=24, color=GREY, edgecolor="white", alpha=.85)
    ax.axvspan(cfg.POOR_LOWER, cfg.AVG_LOWER, color=RED, alpha=.13)
    ax.axvspan(cfg.AVG_LOWER, cfg.GOOD_LOWER, color=BLUE, alpha=.13)
    ax.axvspan(cfg.GOOD_LOWER, cfg.GOOD_UPPER, color=GREEN, alpha=.13)
    counts = y_cat.value_counts()
    for x, label in [(1.95, "Poor"), (3.15, "Average"), (3.82, "Good")]:
        ax.text(x, ax.get_ylim()[1] * .82,
                f"{label}\n(n={counts.get(label, 0)})",
                color=colors[label], ha="center", fontsize=8)
    ax.set(xlabel="Exact CGPA", ylabel="Students",
           title="SAPEx-D CGPA distribution and performance bands")
    _save(fig, "cgpa_distribution.png")


def _figure_categories(y_cat):
    order = ["Good", "Average", "Poor"]
    counts = y_cat.value_counts().reindex(order, fill_value=0)
    colors = [GREEN, BLUE, RED]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bars = ax.bar(order, counts.values, color=colors)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 4,
                f"{value} ({value/len(y_cat):.1%})", ha="center", fontsize=8)
    ax.set(ylabel="Students", title="CGPA-derived performance categories")
    _save(fig, "category_distribution.png")


def _figure_demographics(raw):
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 2.9))
    genders = [int(raw["Gender_Male"].sum()), int(raw["Gender_Female"].sum())]
    axes[0].bar(["Male", "Female"], genders, color=[BLUE, RED])
    for i, v in enumerate(genders): axes[0].text(i, v+4, str(v), ha="center", fontsize=8)
    axes[0].set_title("Gender"); axes[0].set_ylabel("Students")
    age = raw["Age"].value_counts().sort_index()
    age_counts = [int(age.get(1, 0)), int(age.get(2, 0))]
    axes[1].bar(["18–21", "22–25"], age_counts, color=GREY)
    for i, v in enumerate(age_counts): axes[1].text(i, v+4, str(v), ha="center", fontsize=8)
    axes[1].set_title("Age group")
    labels = ["A", "A−", "B+", "B", "B−", "C+", "C", "C−"]
    counts = [int(raw["Letter Grade"].value_counts().get(i, 0)) for i in range(1, 9)]
    axes[2].bar(labels, counts, color=BLUE)
    axes[2].set_title("Letter grade")
    for ax in axes: ax.tick_params(axis="x", labelsize=7)
    _save(fig, "demographics.png")


def _figure_feature_importance(result):
    scores = result["feature_scores"].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.barh(scores.index, scores.values, color=BLUE)
    ax.set(xlabel="Training-set F-score", title="Top selected features (F-ANOVA)")
    _save(fig, "feature_importance.png")


def _figure_correlation(raw):
    columns = [
        "Scholar-Ship Type", "How Do you rate your interest in your major course",
        "Attendance To Classes", "Weekly Study Hours",
        "Reading Frequency(Scientific Books / Journals)", "Taking Notes in Classes",
        "Listening in Classes", "Discussion improves my interest & success in the course",
        "Your Attendance in Seminars/ Conference  Related to the Department",
        "Conscientiousness", "Openness", "Agreeableness", "Extraversion",
        "Neuroticism", "Any Artistic or Sports Activity?", "Total Salary if Available?",
    ]
    names, values = [], []
    aliases = {
        "Scholar-Ship Type": "Scholarship type",
        "How Do you rate your interest in your major course": "Interest in major",
        "Attendance To Classes": "Class attendance",
        "Weekly Study Hours": "Weekly study hours",
        "Reading Frequency(Scientific Books / Journals)": "Scientific reading",
        "Taking Notes in Classes": "Note-taking",
        "Listening in Classes": "Listening in class",
        "Discussion improves my interest & success in the course": "Discussion engagement",
        "Your Attendance in Seminars/ Conference  Related to the Department": "Seminar attendance",
        "Any Artistic or Sports Activity?": "Arts/sports activity",
        "Total Salary if Available?": "Salary category",
    }
    target = raw["Exact CGPA"].astype(float)
    for col in columns:
        if col in raw.columns:
            r = raw[col].astype(float).corr(target)
            if np.isfinite(r):
                values.append(float(r)); names.append(aliases.get(col, col))
    order = np.argsort(values)
    vals = np.asarray(values)[order]
    cols = [RED if v < 0 else GREEN for v in vals]
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    ax.barh(np.asarray(names)[order], vals, color=cols, alpha=.88)
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Pearson r with CGPA (descriptive)",
           title="Unadjusted feature–CGPA associations")
    _save(fig, "correlation.png")


def _figure_personality(raw, y_cat):
    traits = cfg.PERSONALITY_TRAITS
    data = raw[traits].copy()
    data["category"] = np.asarray(y_cat)
    means = data.groupby("category")[traits].mean()
    labels = ["Openness", "Conscientiousness", "Extraversion",
              "Agreeableness", "Neuroticism"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.5, 4.2), subplot_kw={"polar": True})
    for cat, color, label in [("Good", GREEN, "Good"), ("Poor", RED, "Poor")]:
        vals = means.loc[cat, traits].to_numpy(dtype=float) / 100.0
        vals = np.append(vals, vals[0])
        ax.plot(angles, vals, color=color, lw=2,
                label=f"{label} (n={(np.asarray(y_cat)==cat).sum()})")
        ax.fill(angles, vals, color=color, alpha=.13)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticks([.25, .5, .75])
    ax.set_title("Descriptive Big-Five means by CGPA band", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=7.5)
    _save(fig, "personality_radar.png")


def _figure_framework():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.set_xlim(0, 8.0); ax.set_ylim(0, 6.7); ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf1f8", ec=BLUE, fs=7.4):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               fc=fc, ec=ec, lw=1.25)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.2)

    def arrow(x1, y1, x2, y2, color="#555", lw=1.05):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, color=color, lw=lw))

    # Training-only selection, then the four simulated IID training shards.
    box(.18, 4.28, 1.82, 1.60,
        "Training-only input selection\n86 encoded candidates\n(81 non-personality + 5 Big Five)\nSelect 30 + 5 = 35 inputs\n(30 in no-Big-Five ablation)",
        fc="#f1f5fa", fs=7.0)
    ax.text(2.88, 6.20, "Four simulated IID clients", ha="center",
            va="center", fontsize=7.8, weight="bold", color=BLUE)
    shard_y = [5.35, 4.53, 3.71, 2.89]
    shard_n = [99, 99, 99, 98]
    for i, (y, n) in enumerate(zip(shard_y, shard_n), start=1):
        box(2.22, y, 1.32, .60, f"Client {i}\n{n} training rows", fs=6.8)
        arrow(2.00, 5.08, 2.22, y+.30, color="#667788", lw=.9)

    box(3.82, 4.28, 1.82, 1.60,
        "Client-local models\nAutoencoder: 35→16→8→16→35\nRandom Forest + Isolation Forest\n35 inputs per detector\n5 FedAvg rounds; 10 local epochs",
        fc="#eaf1f8", fs=6.8)
    box(5.92, 4.28, 1.82, 1.60,
        "In-process coordinator\nFedAvg autoencoder updates\nMean of four IF scores\nMean of four RF risk probabilities\nThree detector scores",
        fc="#fdeaea", ec=RED, fs=6.8)

    # Model-training and score-aggregation flow.
    for y in [5.65, 4.83, 4.01, 3.19]:
        arrow(3.54, y, 3.82, 5.08, color="#667788", lw=.9)
    arrow(5.64, 5.08, 5.92, 5.08)

    # OOF evaluation and human-reviewed preliminary guidance.
    box(.35, 1.12, 2.40, 1.24,
        "Leakage-controlled evaluation\nFive-fold OOF training scores\nLogistic stack uses 3 scores\nThresholds set on OOF only;\nheld-out test scored once",
        fc="#eaf6ea", ec=GREEN, fs=7.0)
    box(3.18, 1.12, 2.30, 1.24,
        "Preliminary, behavior-focused prompts\nCandidate guidance only\nHuman review required\nNot evaluated as an intervention",
        fc="#fff7e0", ec="#b8860b", fs=7.0)
    box(5.92, 1.12, 1.82, 1.24,
        "Separate client-level DP run\nAutoencoder only; clip C=1\n5 rounds; δ=10⁻⁵; Gaussian noise",
        fc="#fff4f1", ec=RED, fs=6.6)
    arrow(6.35, 4.28, 2.75, 2.36, color=GREEN, lw=1.15)
    arrow(2.75, 1.74, 3.18, 1.74, color="#555")
    arrow(6.83, 4.28, 6.83, 2.36, color=RED, lw=1.1)

    ax.text(4.0, .67,
            "Primary FedPRISM run is unnoised; no cryptographic secure aggregation; coordinator sees updates.",
            ha="center", va="center", fontsize=7.0, color=RED, weight="bold")
    ax.text(4.0, .22,
            "Single-process IID simulation—not a deployed cross-institution network",
            ha="center", va="center", fontsize=8.0, style="italic", color="#333")
    _save(fig, "framework.png")

def _figure_convergence(history):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(np.arange(1, len(history)+1), history, marker="o", color=BLUE, lw=1.7)
    ax.set(xlabel="FedAvg round", ylabel="Training reconstruction MSE",
           title="Federated autoencoder training loss")
    ax.grid(True, alpha=.25)
    _save(fig, "fedavg_convergence.png")


def _figure_confusion(result):
    names = ["Autoencoder", "Isolation Forest", "Random Forest", "FedPRISM"]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
    for ax, name in zip(axes, names):
        cm = confusion_matrix(result["y_test"], result["predictions"][name], labels=[0,1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
        ax.set_title(name, fontsize=8)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["On-track", "At-risk"], rotation=30, fontsize=6)
        ax.set_yticklabels(["On-track", "At-risk"], fontsize=6)
        ax.set_xlabel("Predicted", fontsize=7)
    axes[0].set_ylabel("Actual", fontsize=7)
    _save(fig, "confusion_matrices.png")


def _figure_model_comparison(metrics):
    names = ["Autoencoder", "Isolation Forest", "Random Forest", "FedPRISM"]
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics_to_plot)); width = .18
    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    colors = ["#8ca9c9", "#6a8fb5", GREEN, RED]
    for i, name in enumerate(names):
        vals = [metrics[name][m] for m in metrics_to_plot]
        ax.bar(x + (i-1.5)*width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0,1.04); ax.set_ylabel("Held-out score")
    ax.set_title("Leakage-controlled held-out performance")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "model_comparison.png")


def _figure_roc(result):
    fig, ax = plt.subplots(figsize=(5.4,4.4))
    for name, color in [("Autoencoder",BLUE), ("Isolation Forest",GREY),
                        ("Random Forest",GREEN), ("FedPRISM",RED)]:
        fpr,tpr,_ = roc_curve(result["y_test"], result["scores"][name])
        ax.plot(fpr,tpr,lw=1.7,color=color,
                label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"k--",lw=.8,alpha=.5)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate",
           title="Held-out ROC curves")
    ax.legend(fontsize=7,loc="lower right")
    _save(fig,"roc_curves.png")


def _figure_far(metrics):
    names = ["Autoencoder", "Isolation Forest", "Random Forest", "FedPRISM"]
    x = np.arange(len(names))
    vals = [metrics[n]["false_alarm_rate"] for n in names]
    lo = [metrics[n]["ci95"]["false_alarm_rate"][0] for n in names]
    hi = [metrics[n]["ci95"]["false_alarm_rate"][1] for n in names]
    yerr = np.array([[v-l for v,l in zip(vals,lo)], [h-v for v,h in zip(vals,hi)]])
    fig, ax = plt.subplots(figsize=(6.2,3.4))
    ax.bar(x, vals, color=[BLUE,GREY,GREEN,RED], alpha=.9)
    ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Autoencoder", "Isolation\nForest", "Random\nForest", "FedPRISM"], fontsize=7)
    ax.set_ylim(0, max(.6, max(hi)+.08)); ax.set_ylabel("False-alarm rate (FP / [FP + TN])")
    ax.set_title("Held-out false alarms with stratified-bootstrap 95% CIs")
    _save(fig,"false_alarm_rate.png")


def _figure_ablation(ablation):
    names = list(ablation.keys())
    labels = {
        "Autoencoder": "Autoencoder",
        "Isolation Forest": "Isolation Forest",
        "Random Forest": "Random Forest",
        "FedPRISM": "FedPRISM",
        "Equal-weight blend": "Equal-weight blend",
        "FedPRISM without Autoencoder": "FedPRISM without\nAutoencoder",
        "FedPRISM without Isolation Forest": "FedPRISM without\nIsolation Forest",
        "FedPRISM without Random Forest": "FedPRISM without\nRandom Forest",
        "FedPRISM without Big Five (30 features)": "FedPRISM without\nBig Five (30 features)",
    }
    short = [labels.get(n, n) for n in names]
    acc = [ablation[n]["accuracy"] for n in names]
    f1 = [ablation[n]["f1"] for n in names]
    x=np.arange(len(names)); w=.38
    fig,ax=plt.subplots(figsize=(9.4,4.2))
    ax.bar(x-w/2,acc,w,label="Accuracy",color=BLUE)
    ax.bar(x+w/2,f1,w,label="F1",color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(short,rotation=35,ha="right",fontsize=6.5)
    ax.set_ylim(0,1.02); ax.set_ylabel("Held-out score")
    ax.set_title("FedPRISM detector and feature ablations")
    ax.legend(fontsize=8)
    _save(fig,"ablation.png")


def _figure_dp(dp_rows):
    rows = dp_rows[1:]
    labels = [r["epsilon_total"] for r in rows]
    vals = [r["ae_test_roc_auc"] for r in rows]
    lo = [r["ae_auc_ci95"][0] for r in rows]
    hi = [r["ae_auc_ci95"][1] for r in rows]
    x=np.arange(len(rows))
    yerr=np.array([[v-l for v,l in zip(vals,lo)],[h-v for v,h in zip(vals,hi)]])
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    ax.errorbar(x,vals,yerr=yerr,fmt="o-",capsize=3,color=BLUE,lw=1.8)
    ax.set_xticks(x); ax.set_xticklabels([f"ε={v}" for v in labels])
    ax.set_ylim(0,1); ax.set_ylabel("AE ROC-AUC on held-out test set")
    ax.set_title("Client-level DP sensitivity of the AE component only")
    ax.text(.99,.02,"Not end-to-end DP for FedPRISM",transform=ax.transAxes,
            ha="right",va="bottom",fontsize=7,color=RED)
    ax.grid(axis="y",alpha=.25)
    _save(fig,"privacy_utility.png")


def _figure_reconstruction(result, X_raw, train_idx, test_idx):
    Xtr_raw = X_raw.iloc[train_idx].reset_index(drop=True)
    Xte_raw = X_raw.iloc[test_idx].reset_index(drop=True)
    Xtr_scaled,Xte_scaled = preprocess.fit_scale(Xtr_raw,Xte_raw)
    feats=result["features"]
    Xte=Xte_scaled.loc[:,feats].to_numpy(float)
    err=result["models"]["ae"].anomaly_scores(Xte)
    y=result["y_test"]
    upper=np.percentile(err,99)
    bins=np.linspace(0,upper,26)
    fig,ax=plt.subplots(figsize=(5.8,3.5))
    ax.hist(err[y==0],bins=bins,alpha=.6,color=BLUE,label="On-track")
    ax.hist(err[y==1],bins=bins,alpha=.6,color=RED,label="At-risk")
    med0=float(np.median(err[y==0])); med1=float(np.median(err[y==1]))
    ax.axvline(med0,color=BLUE,ls="--",lw=1,label=f"On-track median={med0:.3f}")
    ax.axvline(med1,color=RED,ls="--",lw=1,label=f"At-risk median={med1:.3f}")
    ax.set(xlabel="Autoencoder reconstruction MSE",ylabel="Held-out students",
           title="Held-out reconstruction-error distributions")
    ax.legend(fontsize=7)
    _save(fig,"reconstruction_error.png")


def _figure_clients(result):
    clients=result["client_ids"]
    y=result["y_train"]
    sizes=[len(c) for c in clients]
    neg=[int(np.sum(y[c]==0)) for c in clients]
    pos=[int(np.sum(y[c]==1)) for c in clients]
    x=np.arange(len(sizes))
    fig,ax=plt.subplots(figsize=(5.4,3.3))
    ax.bar(x,neg,label="On-track",color=BLUE)
    ax.bar(x,pos,bottom=neg,label="At-risk",color=RED)
    ax.set_xticks(x); ax.set_xticklabels([f"Client {i+1}" for i in x])
    ax.set_ylabel("Training students")
    ax.set_title("Simulated IID client shards (not actual departments)")
    ax.legend(fontsize=8)
    _save(fig,"client_distribution.png")


def generate_all_figures(X_raw, y_cat, cgpa, train_idx, test_idx,
                         fed, central, no_personality, fed_metrics,
                         ablation, dp_rows):
    raw = pd.read_csv(cfg.DATASET_FILE)
    raw.columns=[str(c).strip() for c in raw.columns]
    _figure_categories(y_cat)
    _figure_cgpa(cgpa,y_cat)
    _figure_demographics(raw)
    _figure_feature_importance(fed)
    _figure_correlation(raw)
    _figure_personality(raw,y_cat)
    _figure_framework()
    _figure_convergence(fed.get("history"))
    _figure_confusion(fed)
    _figure_model_comparison(fed_metrics)
    _figure_roc(fed)
    _figure_far(fed_metrics)
    _figure_ablation(ablation)
    _figure_dp(dp_rows)
    _figure_reconstruction(fed,X_raw,train_idx,test_idx)
    _figure_clients(fed)
