"""Leakage-controlled experiments for the SAPEx-D at-risk task.

Primary evaluation uses one stratified 80/20 split (seed 42). Feature scaling
and selection are fitted within each split. Meta-learning uses out-of-fold
(OOF) detector scores from 5-fold cross-fitting on the training partition;
the held-out test partition is never used for feature selection, calibration,
or meta-learner fitting. This script also computes paired bootstrap intervals,
McNemar tests, detector/feature ablations, and a separately scoped DP-AE
sensitivity analysis.

Run: python main.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config as cfg
import preprocess
import feature_selection
import federated
import evaluate
from models import (Autoencoder, build_isolation_forest,
                    build_random_forest, stack_ensemble)


MODEL_NAMES = ["Autoencoder", "Isolation Forest", "Random Forest", "FedPRISM"]
SCORE_NAMES = ["Autoencoder", "Isolation Forest", "Random Forest"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "false_alarm_rate", "roc_auc"]


def _fit_federated(X, y, seed, epsilon=None, collect_history=False):
    client_ids = federated.make_clients(len(X), seed=seed)
    clients_X = [X[c] for c in client_ids]
    clients_y = [y[c] for c in client_ids]
    clients_X_normal = [xc[yc == cfg.NORMAL_LABEL]
                        for xc, yc in zip(clients_X, clients_y)]

    if collect_history:
        ae, history = federated.fedavg_autoencoder(
            clients_X_normal, X.shape[1], epsilon=epsilon, seed=seed,
            return_history=True)
    else:
        ae = federated.fedavg_autoencoder(
            clients_X_normal, X.shape[1], epsilon=epsilon, seed=seed)
        history = None
    iso = federated.federated_isolation_forest(clients_X, seed=seed)
    rf = federated.federated_random_forest(clients_X, clients_y, seed=seed)
    return {"ae": ae, "iso": iso, "rf": rf, "history": history,
            "client_ids": client_ids}


def _fit_central(X, y, seed):
    ae = Autoencoder(X.shape[1], seed=seed).fit(X[y == cfg.NORMAL_LABEL])
    iso = build_isolation_forest(seed=seed).fit(X)
    rf = build_random_forest(seed=seed).fit(X, y)
    return {"ae": ae, "iso": iso, "rf": rf, "history": None}


def _scores(models, X, is_federated):
    ae_score = models["ae"].anomaly_scores(X)
    if is_federated:
        iso_score = -np.mean([m.decision_function(X) for m in models["iso"]], axis=0)
        rf_score = np.mean([m.predict_proba(X)[:, 1] for m in models["rf"]], axis=0)
    else:
        iso_score = -models["iso"].decision_function(X)
        rf_score = models["rf"].predict_proba(X)[:, 1]
    return np.column_stack([ae_score, iso_score, rf_score])


def _scaled_selected(X_fit_raw, X_eval_raw, cgpa_fit, include_personality):
    X_fit, X_eval = preprocess.fit_scale(X_fit_raw, X_eval_raw)
    X_fit_sel, f_scores = feature_selection.select_features(
        X_fit, cgpa_fit, include_personality=include_personality)
    features = list(X_fit_sel.columns)
    return (X_fit_sel.to_numpy(dtype=float),
            X_eval.loc[:, features].to_numpy(dtype=float),
            features, f_scores.loc[features])


def _run_protocol(X_raw, y_bin, cgpa, train_idx, test_idx,
                  include_personality=True, federated_model=True,
                  epsilon=None, seed=cfg.RANDOM_SEED, collect_history=False):
    """Cross-fit train scores, fit base/meta models, and evaluate untouched test."""
    X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_raw.iloc[test_idx].reset_index(drop=True)
    y_train = np.asarray(y_bin)[train_idx]
    y_test = np.asarray(y_bin)[test_idx]
    cgpa_train = np.asarray(cgpa)[train_idx]

    cv = StratifiedKFold(n_splits=cfg.META_FOLDS, shuffle=True,
                         random_state=seed)
    oof = np.full((len(train_idx), 3), np.nan, dtype=float)
    oof_features = []
    for fold, (fit_ids, val_ids) in enumerate(cv.split(np.zeros(len(y_train)), y_train)):
        X_fit, X_val, feats, _ = _scaled_selected(
            X_train_raw.iloc[fit_ids], X_train_raw.iloc[val_ids],
            cgpa_train[fit_ids], include_personality)
        y_fit = y_train[fit_ids]
        model_seed = seed + 100 * (fold + 1)
        if federated_model:
            models = _fit_federated(X_fit, y_fit, model_seed,
                                    epsilon=epsilon, collect_history=False)
        else:
            models = _fit_central(X_fit, y_fit, model_seed)
        oof[val_ids] = _scores(models, X_val, federated_model)
        oof_features.append({"fold": fold + 1, "feature_count": len(feats),
                             "features": feats})

    if np.isnan(oof).any():
        raise RuntimeError("OOF scores are incomplete")

    X_fit, X_test, features, f_scores = _scaled_selected(
        X_train_raw, X_test_raw, cgpa_train, include_personality)
    if federated_model:
        final_models = _fit_federated(
            X_fit, y_train, seed, epsilon=epsilon,
            collect_history=collect_history)
    else:
        final_models = _fit_central(X_fit, y_train, seed)
    test_scores = _scores(final_models, X_test, federated_model)

    thresholds = [evaluate.calibrate_threshold(y_train, oof[:, j])
                  for j in range(3)]
    predictions = {
        name: (test_scores[:, j] >= thresholds[j]).astype(int)
        for j, name in enumerate(SCORE_NAMES)
    }

    meta = stack_ensemble(oof, y_train, seed=seed)
    ensemble_scores = meta.predict_proba(test_scores)[:, 1]
    predictions["FedPRISM"] = meta.predict(test_scores)

    # Equal-weight standardized score average, with its decision threshold
    # selected from OOF predictions only (a non-learned ensemble ablation).
    zscaler = StandardScaler().fit(oof)
    oof_equal = zscaler.transform(oof).mean(axis=1)
    test_equal = zscaler.transform(test_scores).mean(axis=1)
    equal_threshold = evaluate.calibrate_threshold(y_train, oof_equal)
    predictions["Equal-weight blend"] = (test_equal >= equal_threshold).astype(int)

    # Drop-one-detector ablations; every meta-learner is trained on OOF scores.
    detector_ablations = {}
    for drop_i, drop_name in enumerate(SCORE_NAMES):
        keep = [i for i in range(3) if i != drop_i]
        model_drop = stack_ensemble(oof[:, keep], y_train, seed=seed)
        name = f"FedPRISM without {drop_name}"
        sc = model_drop.predict_proba(test_scores[:, keep])[:, 1]
        detector_ablations[name] = {"scores": sc,
                                    "predictions": model_drop.predict(test_scores[:, keep])}

    all_scores = {name: test_scores[:, j] for j, name in enumerate(SCORE_NAMES)}
    all_scores["FedPRISM"] = ensemble_scores
    all_scores["Equal-weight blend"] = test_equal
    all_predictions = dict(predictions)
    for name, vals in detector_ablations.items():
        all_scores[name] = vals["scores"]
        all_predictions[name] = vals["predictions"]

    return {
        "y_train": y_train, "y_test": y_test,
        "oof_scores": oof, "test_scores": test_scores,
        "thresholds": thresholds, "predictions": all_predictions,
        "scores": all_scores, "features": features,
        "feature_scores": f_scores, "models": final_models,
        "meta": meta, "history": final_models.get("history"),
        "client_ids": final_models.get("client_ids"),
        "oof_features": oof_features,
        "is_federated": federated_model,
    }


def _metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _stratified_bootstrap_ci(y, pred, score, n_boot=cfg.BOOTSTRAP_REPS,
                             seed=cfg.RANDOM_SEED):
    """Percentile CIs from a class-stratified bootstrap of held-out students."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    score = np.asarray(score)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in METRIC_NAMES}
    for _ in range(n_boot):
        ids = np.concatenate([
            rng.choice(neg, len(neg), replace=True),
            rng.choice(pos, len(pos), replace=True),
        ])
        m = _metrics(y[ids], pred[ids], score[ids])
        for k in METRIC_NAMES:
            samples[k].append(m[k])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in samples.items()}


def _paired_bootstrap_diff(y, pred_a, score_a, pred_b, score_b,
                           n_boot=cfg.BOOTSTRAP_REPS, seed=cfg.RANDOM_SEED):
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "f1", "false_alarm_rate", "roc_auc"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        ids = np.concatenate([rng.choice(neg, len(neg), replace=True),
                              rng.choice(pos, len(pos), replace=True)])
        ma = _metrics(y[ids], np.asarray(pred_a)[ids], np.asarray(score_a)[ids])
        mb = _metrics(y[ids], np.asarray(pred_b)[ids], np.asarray(score_b)[ids])
        for k in keys:
            samples[k].append(ma[k] - mb[k])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in samples.items()}


def _mcnemar_exact(y, pred_a, pred_b):
    correct_a = np.asarray(pred_a) == np.asarray(y)
    correct_b = np.asarray(pred_b) == np.asarray(y)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    p = float(binomtest(min(a_only, b_only), n, 0.5,
                        alternative="two-sided").pvalue) if n else 1.0
    return {"fedprism_only_correct": a_only,
            "comparator_only_correct": b_only,
            "discordant": n, "mcnemar_exact_p": p}


def _summarize_predictions(result, seed=cfg.RANDOM_SEED):
    y = result["y_test"]
    out = {}
    for name, pred in result["predictions"].items():
        score = result["scores"][name]
        m = _metrics(y, pred, score)
        ci = _stratified_bootstrap_ci(y, pred, score, seed=seed)
        out[name] = {**m, "ci95": ci}
    return out


def _fit_central_prediction_ablations(result, y_train, y_test):
    """Fit meta ablations from OOF scores and evaluate same test observations."""
    oof, test = result["oof_scores"], result["test_scores"]
    out = {}
    for drop_i, drop_name in enumerate(SCORE_NAMES):
        keep = [i for i in range(3) if i != drop_i]
        meta = stack_ensemble(oof[:, keep], y_train)
        sc = meta.predict_proba(test[:, keep])[:, 1]
        out[f"FedPRISM without {drop_name}"] = {
            "scores": sc, "predictions": meta.predict(test[:, keep])}
    return out


def _dp_autoencoder_sweep(result, y_train, y_test, seed=cfg.RANDOM_SEED):
    """Evaluate AE-only test AUC under client-level DP; no end-to-end claim."""
    Xtr = result["X_train_final"]
    Xte = result["X_test_final"]
    client_ids = federated.make_clients(len(Xtr), seed=seed)
    client_y = [y_train[c] for c in client_ids]
    clients_normal = [Xtr[c][yc == cfg.NORMAL_LABEL]
                      for c, yc in zip(client_ids, client_y)]
    n_clients = len(client_ids)
    sensitivity = 2.0 * cfg.DP_CLIP_NORM / n_clients
    rows = []

    # The no-noise value is explicitly a non-private reference point.
    settings = [(None, "non-private")] + [(eps, str(eps)) for eps in cfg.DP_EPSILONS]
    for i, (eps, label) in enumerate(settings):
        ae = federated.fedavg_autoencoder(
            clients_normal, Xtr.shape[1], epsilon=eps,
            delta=cfg.DP_DELTA, clip_norm=cfg.DP_CLIP_NORM,
            seed=seed + 9000 + i)
        s = ae.anomaly_scores(Xte)
        point_auc = float(roc_auc_score(y_test, s))
        # Stratified bootstrap CI for AE AUC on the same untouched test set.
        pred = (s >= np.median(ae.anomaly_scores(Xtr)))
        ci = _stratified_bootstrap_ci(y_test, pred.astype(int), s,
                                      seed=seed + 100 + i)["roc_auc"]
        if eps is None:
            noise_mult, noise_std, achieved = 0.0, 0.0, None
        else:
            noise_mult = federated.noise_multiplier_for_epsilon(
                eps, cfg.DP_DELTA, cfg.FED_ROUNDS)
            noise_std = noise_mult * sensitivity
            achieved = federated.rdp_epsilon(
                noise_mult, cfg.DP_DELTA, cfg.FED_ROUNDS)
        rows.append({"epsilon_total": label,
                     "delta_total": cfg.DP_DELTA if eps is not None else None,
                     "rounds": cfg.FED_ROUNDS,
                     "client_count": n_clients,
                     "clip_norm": cfg.DP_CLIP_NORM,
                     "noise_multiplier": float(noise_mult),
                     "noise_std_aggregate": float(noise_std),
                     "accounted_epsilon": achieved,
                     "ae_test_roc_auc": point_auc,
                     "ae_auc_ci95": ci})
    return rows


def _class_counts_by_client(y_train, seed=cfg.RANDOM_SEED):
    ids = federated.make_clients(len(y_train), seed=seed)
    return [{"client": i + 1, "n": int(len(c)),
             "on_track": int(np.sum(np.asarray(y_train)[c] == 0)),
             "at_risk": int(np.sum(np.asarray(y_train)[c] == 1))}
            for i, c in enumerate(ids)]


def _round_dict(d):
    if isinstance(d, dict):
        return {k: _round_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_round_dict(x) for x in d]
    if isinstance(d, float) and np.isfinite(d):
        return round(d, 5)
    return d


def run():
    cfg.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    X_raw, y_cat, y_bin, cgpa = preprocess.prepare()
    indices = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(
        indices, test_size=cfg.TEST_SPLIT, stratify=y_bin,
        random_state=cfg.RANDOM_SEED)

    # Main result: federated simulation without noise; explicitly not an
    # end-to-end DP result. DP is assessed for the AE module separately below.
    print("[1/6] Cross-fitting federated 35-feature pipeline ...", flush=True)
    fed = _run_protocol(X_raw, y_bin, cgpa, train_idx, test_idx,
                        include_personality=True, federated_model=True,
                        epsilon=None, collect_history=True)
    fed["X_train_final"] = preprocess.fit_scale(
        X_raw.iloc[train_idx].reset_index(drop=True),
        X_raw.iloc[test_idx].reset_index(drop=True))[0].loc[:, fed["features"]].to_numpy(float)
    fed["X_test_final"] = preprocess.fit_scale(
        X_raw.iloc[train_idx].reset_index(drop=True),
        X_raw.iloc[test_idx].reset_index(drop=True))[1].loc[:, fed["features"]].to_numpy(float)

    # Centralized comparator trained/evaluated under the same leakage controls.
    print("[2/6] Cross-fitting centralized comparator ...", flush=True)
    central = _run_protocol(X_raw, y_bin, cgpa, train_idx, test_idx,
                            include_personality=True, federated_model=False,
                            epsilon=None, collect_history=False)

    # Personality ablation: repeat cross-fitting with only 30 non-personality
    # F-ANOVA features; identical outer split, folds, model configurations.
    print("[3/6] Cross-fitting 30-feature personality ablation ...", flush=True)
    no_personality = _run_protocol(
        X_raw, y_bin, cgpa, train_idx, test_idx,
        include_personality=False, federated_model=True,
        epsilon=None, collect_history=False)

    fed_metrics = _summarize_predictions(fed)
    central_metrics = _summarize_predictions(central, seed=cfg.RANDOM_SEED + 5)
    no_personality_metrics = _summarize_predictions(
        no_personality, seed=cfg.RANDOM_SEED + 10)

    # Add controlled model/component ablations using the cross-fitted scores.
    ablation = {}
    for name in ["Autoencoder", "Isolation Forest", "Random Forest",
                 "FedPRISM", "Equal-weight blend",
                 "FedPRISM without Autoencoder",
                 "FedPRISM without Isolation Forest",
                 "FedPRISM without Random Forest"]:
        ablation[name] = fed_metrics[name]
    ablation["FedPRISM without Big Five (30 features)"] = \
        no_personality_metrics["FedPRISM"]

    yte = fed["y_test"]
    comparisons = {}
    for comparator in ["Random Forest", "Autoencoder", "Isolation Forest"]:
        comparisons[f"FedPRISM vs {comparator}"] = {
            "paired_bootstrap_difference_fedprism_minus_comparator":
                _paired_bootstrap_diff(
                    yte, fed["predictions"]["FedPRISM"],
                    fed["scores"]["FedPRISM"],
                    fed["predictions"][comparator], fed["scores"][comparator],
                    seed=cfg.RANDOM_SEED + 20),
            **_mcnemar_exact(yte, fed["predictions"]["FedPRISM"],
                             fed["predictions"][comparator]),
        }
    comparisons["Federated simulation vs centralized comparator"] = {
        "paired_bootstrap_difference_fedprism_minus_comparator":
            _paired_bootstrap_diff(
                yte, fed["predictions"]["FedPRISM"],
                fed["scores"]["FedPRISM"],
                central["predictions"]["FedPRISM"],
                central["scores"]["FedPRISM"], seed=cfg.RANDOM_SEED + 21),
        **_mcnemar_exact(yte, fed["predictions"]["FedPRISM"],
                         central["predictions"]["FedPRISM"]),
    }
    comparisons["FedPRISM vs no Big Five"] = {
        "paired_bootstrap_difference_fedprism_minus_comparator":
            _paired_bootstrap_diff(
                yte, fed["predictions"]["FedPRISM"],
                fed["scores"]["FedPRISM"],
                no_personality["predictions"]["FedPRISM"],
                no_personality["scores"]["FedPRISM"],
                seed=cfg.RANDOM_SEED + 22),
        **_mcnemar_exact(yte, fed["predictions"]["FedPRISM"],
                         no_personality["predictions"]["FedPRISM"]),
    }

    print("[4/6] Calibrating DP autoencoder sensitivity sweep ...", flush=True)
    dp_rows = _dp_autoencoder_sweep(fed, fed["y_train"], fed["y_test"])
    clients = _class_counts_by_client(fed["y_train"])
    print("[5/6] Saving tables and figures ...", flush=True)

    # Save human-readable experiment outputs.
    pred_df = pd.DataFrame({"y_true": yte})
    for name, pred in fed["predictions"].items():
        pred_df[f"pred__{name}"] = pred
    for name, score in fed["scores"].items():
        pred_df[f"score__{name}"] = score
    pred_df["central_pred__FedPRISM"] = central["predictions"]["FedPRISM"]
    pred_df["central_score__FedPRISM"] = central["scores"]["FedPRISM"]
    pred_df["no_bigfive_pred__FedPRISM"] = no_personality["predictions"]["FedPRISM"]
    pred_df["no_bigfive_score__FedPRISM"] = no_personality["scores"]["FedPRISM"]
    pred_df.to_csv(cfg.RESULTS_DIR / "test_predictions.csv", index=False)

    pd.DataFrame([{ "model": name, **{k: m[k] for k in METRIC_NAMES},
                    **{f"{k}_lo": m["ci95"][k][0] for k in METRIC_NAMES},
                    **{f"{k}_hi": m["ci95"][k][1] for k in METRIC_NAMES}}
                  for name, m in fed_metrics.items()]).to_csv(
        cfg.RESULTS_DIR / "heldout_metrics.csv", index=False)
    pd.DataFrame({name: {k: fed_metrics[name][k] for k in METRIC_NAMES}
                  for name in MODEL_NAMES}).T.to_csv(
        cfg.RESULTS_DIR / "model_comparison.csv", index_label="model")
    pd.DataFrame([{ "ablation": name,
                    **{k: m[k] for k in ["accuracy", "f1", "false_alarm_rate", "roc_auc"]},
                    "accuracy_lo": m["ci95"]["accuracy"][0],
                    "accuracy_hi": m["ci95"]["accuracy"][1],
                    "f1_lo": m["ci95"]["f1"][0],
                    "f1_hi": m["ci95"]["f1"][1]}
                  for name, m in ablation.items()]).to_csv(
        cfg.RESULTS_DIR / "ablation_metrics.csv", index=False)
    pd.DataFrame(dp_rows).to_csv(cfg.RESULTS_DIR / "dp_ae_sweep.csv", index=False)
    pd.DataFrame(clients).to_csv(cfg.RESULTS_DIR / "client_distribution.csv", index=False)

    payload = {
        "dataset": {"n": int(len(X_raw)), "n_raw_features": int(X_raw.shape[1]),
                    "categories": {str(k): int(v) for k, v in y_cat.value_counts().items()}},
        "split": {"seed": cfg.RANDOM_SEED, "train_size": len(train_idx),
                  "test_size": len(test_idx), "test_positive": int(np.sum(yte == 1)),
                  "test_negative": int(np.sum(yte == 0)),
                  "meta_crossfit_folds": cfg.META_FOLDS},
        "feature_dimensions": {"raw_predictors": int(X_raw.shape[1]),
                               "selected_with_personality": int(len(fed["features"])),
                               "selected_without_personality": int(len(no_personality["features"]))},
        "client_distribution": clients,
        "federated_simulation": fed_metrics,
        "centralized_comparator": central_metrics,
        "FedPRISM without Big Five (30 features)": no_personality_metrics["FedPRISM"],
        "component_ablations": ablation,
        "paired_comparisons": comparisons,
        "dp_autoencoder_only_sweep": dp_rows,
        "privacy_scope": {
            "guarantee": "client-level approximate DP for released FedAvg autoencoder parameters only when epsilon is specified",
            "delta_total": cfg.DP_DELTA,
            "clip_norm": cfg.DP_CLIP_NORM,
            "rounds": cfg.FED_ROUNDS,
            "accountant": "RDP Gaussian accountant, basic composition across rounds; no subsampling amplification",
            "not_covered": ["Isolation Forest", "Random Forest", "stacked meta-learner", "recommendation outputs", "untrusted aggregation-server view"],
            "secure_aggregation": "not implemented; in-process trusted-coordinator simulation",
        },
        "feature_selection": {"selected_features": fed["features"],
                              "f_scores": {k: float(v) for k, v in fed["feature_scores"].items()}},
        "thresholds_from_oof": {k: float(v) for k, v in zip(SCORE_NAMES, fed["thresholds"])},
    }
    evaluate.save_json(_round_dict(payload), cfg.RESULTS_DIR / "metrics.json")

    # Save the selected-feature lists and FedAvg convergence history.
    with open(cfg.RESULTS_DIR / "selected_features.txt", "w") as f:
        f.write("Selected features (35: top 30 non-personality F-score + 5 Big Five)\n")
        f.write("\n".join(fed["features"]))
    if fed["history"]:
        pd.DataFrame({"round": np.arange(1, len(fed["history"]) + 1),
                      "training_reconstruction_mse": fed["history"]}).to_csv(
            cfg.RESULTS_DIR / "fedavg_history.csv", index=False)

    # Pass the correct outer-split arrays to the plotting module.
    from analysis_figures import generate_all_figures
    generate_all_figures(X_raw, y_cat, cgpa, train_idx, test_idx,
                         fed, central, no_personality, fed_metrics,
                         ablation, dp_rows)

    print(json.dumps({"federated": {k: {m: round(v[m], 4) for m in
              ["accuracy", "precision", "recall", "f1", "false_alarm_rate", "roc_auc"]}
              for k, v in fed_metrics.items() if k in MODEL_NAMES},
              "centralized_accuracy": central_metrics["FedPRISM"]["accuracy"],
              "client_distribution": clients,
              "selected_features": len(fed["features"])}, indent=2))
    return payload
