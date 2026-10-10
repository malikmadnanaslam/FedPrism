"""Rule-based guidance prototype for at-risk academic screening.

The function compares a student's *modifiable academic behaviours* with the
training-set Good-band reference and the autoencoder reconstruction profile.
Its outputs are candidate prompts for human review, not validated treatments,
causal prescriptions, or evidence of improved outcomes. Big-Five variables,
demographics, and immutable context are deliberately excluded from advice.
"""

import numpy as np
import pandas as pd


FEATURE_ADVICE = {
    "Weekly Study Hours":
        "discuss a realistic, structured weekly self-study plan",
    "Reading Frequency(non-Scientific books/ Journals)":
        "consider a regular reading routine using accessible material",
    "Reading Frequency(Scientific Books / Journals)":
        "consider using relevant course readings and journals",
    "Your Attendance in Seminars/ Conference  Related to the Department":
        "explore relevant departmental seminars or academic events",
    "Attendance To Classes":
        "review attendance barriers and available support with an adviser",
    "Taking Notes in Classes":
        "try a note-taking strategy suited to the course",
    "Listening in Classes":
        "discuss practical ways to stay engaged during lectures",
    "Discussion improves my interest & success in the course":
        "consider participating in class discussion or peer study",
    "How Do you rate your interest in your major course":
        "discuss course interests, options, and academic support with an adviser",
    "Scholar-Ship Type":
        "ask an adviser about relevant financial-aid or scholarship options",
    "Any Artistic or Sports Activity?":
        "consider a suitable extracurricular activity if desired and accessible",
}

ACTIONABLE_FEATURES = list(FEATURE_ADVICE.keys())


def explain(feature):
    return FEATURE_ADVICE.get(feature, "discuss this learning behaviour with an adviser")


def recommend(student, good_ref, features, ae, percentile=70):
    """Generate candidate behaviour-focused prompts for human review.

    ``good_ref`` must be constructed from the training partition only. The
    returned prompts are neither causal explanations nor interventions.
    """
    x = np.asarray(student, dtype=float).reshape(1, -1)
    recon = ae.reconstruct(x)
    per_feature_err = np.abs(x - recon).ravel()
    threshold = np.percentile(per_feature_err, percentile)
    if isinstance(good_ref, pd.DataFrame):
        good_avg = good_ref.loc[:, features].mean().values
    else:
        good_avg = np.asarray(good_ref).mean(axis=0)

    recs = []
    for i, feature in enumerate(features):
        if feature not in ACTIONABLE_FEATURES:
            continue
        gap = float(good_avg[i] - x.ravel()[i])
        high_error = bool(per_feature_err[i] >= threshold)
        recs.append({
            "feature": feature,
            "student_value": round(float(x.ravel()[i]), 3),
            "reference_mean": round(float(good_avg[i]), 3),
            "gap": round(gap, 3),
            "reconstruction_error": round(float(per_feature_err[i]), 4),
            "priority_for_review": bool(high_error or gap > 0),
            "candidate_prompt": explain(feature),
        })
    recs.sort(key=lambda item: (not item["priority_for_review"], -item["gap"]))
    return recs


def summarize(recs, top_n=10):
    """Return a brief, non-prescriptive list for an adviser to review."""
    return [f"[REVIEW] {r['feature']}: {r['candidate_prompt']}"
            for r in recs[:top_n]]
