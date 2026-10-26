"""Additive explanation of one prediction from the frozen logistic regression (Phase 6 methodology, notebook 07 part C).

log-odds = intercept + sum_j coefficient_j * transformed_value_j. A raw feature's contribution is the sum of the terms of its transformed columns.
The baseline (intercept) is the reference profile: every standardised score at the training mean and every one-hot variable at its reference level.
Contributions are therefore relative to that baseline. They describe how the model's score is built; they are associations learned from
observational data, not causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction import FEATURES, LABELS, CONTINUOUS, COUNT, BINARY, to_frame


@dataclass(frozen=True)
class Explanation:
    intercept: float
    contributions: pd.DataFrame      # one row per raw feature: feature, label, value, contribution, odds_multiplier, baseline_text, detail
    logit_reconstructed: float       # intercept + sum(contributions)


def _base_of(transformed_name: str) -> str:
    hits = [c for c in FEATURES if transformed_name == c or transformed_name.startswith(c + "_")]
    return max(hits, key=len)        # longest raw name wins (same rule as the Phase 6 notebook)


def explain(pipeline, clean: dict) -> Explanation:
    pre, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    names = list(pre.get_feature_names_out())
    xt = pre.transform(to_frame(clean))
    x = np.asarray(xt, dtype=float)[0]
    terms = x * model.coef_[0]
    per_feature = {f: 0.0 for f in FEATURES}
    for nm, t in zip(names, terms):
        per_feature[_base_of(nm)] += float(t)

    scaler = pre.named_transformers_["num"]
    means = dict(zip(CONTINUOUS, scaler.mean_))
    drops = dict(zip(COUNT, pre.named_transformers_["cnt"].drop))
    rows = []
    for f in FEATURES:
        v = clean[f]
        if f in CONTINUOUS:
            base = f"training average {means[f]:.1f}"
            detail = f"{v:g} (training average {means[f]:.1f})"
        elif f in COUNT:
            base = f"reference level {drops[f]}"
            detail = f"{v} (reference level {drops[f]})"
        else:
            base = "reference level No"
            detail = f"{v} (reference level No)"
        c = per_feature[f]
        rows.append({"feature": f, "label": LABELS[f], "value": v, "contribution": c, "odds_multiplier": float(np.exp(c)), "baseline_text": base, "detail": detail})
    df = pd.DataFrame(rows)
    intercept = float(model.intercept_[0])
    return Explanation(intercept=intercept, contributions=df, logit_reconstructed=intercept + float(df.contribution.sum()))
