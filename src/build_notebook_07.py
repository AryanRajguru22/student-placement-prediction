"""Builds notebooks/07_threshold_explainability_ablation.ipynb (Phase 6).
Final conclusions are read from src/phase6_findings.md so they can be written against the executed numbers (two-pass build)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 07 - Threshold Analysis, Explainability and Feature-Group Ablation (Phase 6)
**Provisional model:** logistic regression with categorical counts (`LR_B`, L2, C = 1), raw (uncalibrated) probabilities - exactly the Phase 4 artifact, as confirmed by Phase 5.

| Part | Question |
|---|---|
| A | How do accuracy, precision, recall, F1 and specificity change with the decision threshold, and what operating threshold is defensible? |
| B | Which features is the model's predicted odds associated with (coefficients, permutation importance)? |
| C | How can individual validation predictions be explained? |
| D | How much predictive information comes from each feature group (ablation)? |
| E | Where does the model make confident mistakes, and what does that say about a performance ceiling? |

## Data policy (enforced in code)
| Split | Phase 6 use |
|---|---|
| TRAIN (`train.csv`, 6,938) | model/pipeline fitting, 5-fold CV for the ablation, coefficient confidence intervals |
| Phase 5 OOF probabilities (`06_oof_predictions.csv`) | honest training-side threshold analysis |
| VALIDATION (`validation.csv`, 1,487) | validation-side threshold analysis, permutation importance, individual explanations, ablation confirmation, error analysis |
| TEST (`test.csv`) | **off limits.** An audit hook *raises an error the moment any process-level `open()` touches a file named `test.csv`* (raw or processed); the notebook ends by printing every data file opened and asserting the test set was never accessed. |

*Disclosure:* the validation split already served model development (Phase 4 selection). A threshold chosen on validation, and then scored on validation, is therefore **optimistic**; the OOF analysis is the independent check. Nothing here is a causal statement: every explanation describes the *fitted model's associations* in observational (possibly rule-generated) data.
""")

code(r"""
import warnings
warnings.simplefilter("always")
import os, sys, json, time
from pathlib import Path

# ---- TEST-SET AUDIT HOOK: record every file open; RAISE immediately if a file named test.csv is opened ----
_OPENED = []
def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]); _OPENED.append(p)
        if os.path.basename(p).lower() == "test.csv":
            raise PermissionError(f"PHASE 6 VIOLATION: attempt to open {p} (the test set is off limits)")
sys.addaudithook(_audit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, sklearn
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models" / "phase4"
MET, FIG = ROOT / "results" / "metrics" / "modeling", ROOT / "results" / "figures" / "modeling"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

SEED, K, B_BOOT = 42, 5, 1000
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 110)
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12,
                                      "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
PLACED_C, NOT_C, NEUTRAL = "#2166ac", "#e08214", "#6b7785"
LR_C = "#0072B2"
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); plt.show()
print("scikit-learn", sklearn.__version__, "| seed", SEED)
""")

md("## 1. Load TRAIN and VALIDATION only; load the fitted Phase 4 `LR_B`; reproduction checks")
code(r"""
TARGET = "PlacementStatus"
FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
CONT = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]       # Phase 4 column order
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
BIN = ["ExtracurricularActivities", "PlacementTraining"]
train = pd.read_csv(PROC / "train.csv"); val = pd.read_csv(PROC / "validation.csv")
for d in (train, val):
    assert d.isna().sum().sum() == 0 and set(d[TARGET]) == {"Placed", "NotPlaced"}
assert not (set(map(tuple, train[FEATURES].values)) & set(map(tuple, val[FEATURES].values)))
X_tr, X_va = train[FEATURES].copy(), val[FEATURES].copy()
y_tr = (train[TARGET] == "Placed").astype(int).to_numpy(); y_va = (val[TARGET] == "Placed").astype(int).to_numpy()
for c in COUNT: X_tr[c] = X_tr[c].astype(int); X_va[c] = X_va[c].astype(int)
BASE_RATE_TR, BASE_RATE_VA = y_tr.mean(), y_va.mean()
print(f"train {X_tr.shape} (Placed {BASE_RATE_TR:.4f}) | validation {X_va.shape} (Placed {BASE_RATE_VA:.4f})")

with open(MODELS / "model_configs.json", encoding="utf-8") as fh: P4 = json.load(fh)
REFS = {k: int(v) for k, v in P4["count_reference_levels_(LR)"].items()}          # fixed Phase 4 reference levels (train mode)
assert REFS == {c: int(X_tr[c].mode().iloc[0]) for c in COUNT}
LR = joblib.load(MODELS / "LR_B.joblib")                                           # already fitted on the full training set in Phase 4
pre, lrm = LR.named_steps["prep"], LR.named_steps["model"]
assert LR.get_params()["model__C"] == 1.0 and lrm.classes_.tolist() == [0, 1]

# (1) the loaded model reproduces the saved Phase 4 validation probabilities
p4 = pd.read_csv(MET / "05_validation_predictions.csv"); assert (p4.y_true.to_numpy() == y_va).all()
p_va = LR.predict_proba(X_va)[:, 1]; assert np.allclose(p_va, p4["LR_B"].to_numpy(), atol=1e-10)
# (2) Phase 5 OOF file: same row order/target as train.csv and the same fold assignment as StratifiedKFold(5, shuffle, 42)
oof = pd.read_csv(MET / "06_oof_predictions.csv")
assert (oof.train_row_id.to_numpy() == np.arange(len(train))).all() and (oof.actual_target.to_numpy() == y_tr).all()
outer = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(X_tr, y_tr))
fold_of = np.empty(len(y_tr), int)
for k, (_, te) in enumerate(outer, 1): fold_of[te] = k
assert (oof.fold.to_numpy() == fold_of).all()
p_oof = oof["logistic_probability"].to_numpy()
# (3) coefficients of the loaded object equal the Phase 4 coefficient table
c4 = pd.read_csv(MET / "05_logistic_coefficients.csv"); c4 = c4[(c4.model_id == "LR_B") & (c4.feature != "(intercept)")]
assert np.allclose(np.sort(c4.coefficient.to_numpy()), np.sort(lrm.coef_[0]), atol=1e-10)
print("Reproduction checks passed: Phase 4 validation probabilities, Phase 5 OOF rows/folds, Phase 4 coefficients.")
print("Count reference levels:", REFS, "| binary reference: 'No'")
""")

md(r"""
## 2. Metric helpers
Classification rule everywhere: **predict Placed when `p >= threshold`** (as in Phases 4-5). The probability remains the primary output; the threshold only turns it into a yes/no decision.
""")
code(r"""
def confusion(y, p, t):
    pred = p >= t; tp = int((pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return tp, tn, fp, fn

def thr_row(y, p, t, dataset):
    tp, tn, fp, fn = confusion(y, p, t); n = len(y); P, N = tp + fn, tn + fp
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / P; spec = tn / N
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"dataset": dataset, "threshold": t, "n": n, "accuracy": (tp + tn) / n, "precision": prec, "recall_sensitivity": rec, "f1": f1, "specificity": spec,
            "false_positive_rate": fp / N, "false_negative_rate": fn / P, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "predicted_placement_rate": (tp + fp) / n,
            "observed_placement_rate": P / n}

def f1_boot(y, pred, IDX):                 # vectorised F1 over bootstrap resamples (IDX: B x n indices)
    yb, pb = y[IDX], pred[IDX]; tp = (yb & pb).sum(1); fp = (~yb.astype(bool) & pb).sum(1); fn = (yb.astype(bool) & ~pb).sum(1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

def metrics_all(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p), "f1": f1_score(y, pred, zero_division=0),
            "accuracy": accuracy_score(y, pred)}
""")

md(r"""
# PART A - Decision-threshold analysis
**Grid:** 0.20 to 0.80 in steps of 0.05 (13 thresholds). The model is **not** refitted: only the cut-off changes. OOF (training-side) and validation results are reported **separately** and never pooled.

**Pre-declared operating criterion (fixed before looking at the results).**
1. *Selected operating threshold under the F1-oriented criterion* = the grid threshold with the highest **validation** F1. Ties (within 1e-9) go to the threshold closest to 0.50.
2. *Practical-sensibility check* (a check, not a constraint): at the selected threshold, validation precision >= 0.60 and recall >= 0.60, and the predicted placement rate within +/-10 percentage points of the observed rate.
3. *Is the default 0.50 practically as good?* A paired bootstrap (1,000 resamples) of F1(selected) - F1(0.50) on validation, and the five per-fold differences on OOF. If the bootstrap interval includes 0, 0.50 is called practically tied with the selected threshold.
4. *Independent check:* the F1-optimal threshold on OOF predictions is reported alongside, but is **not** used for selection.

This is **not** "the best threshold": F1 weights precision and recall equally and ignores the costs of the two error types, which this project has not defined. A different cost trade-off would justify a different threshold.
""")
code(r"""
THR = np.round(np.linspace(0.20, 0.80, 13), 2)
tab_oof = pd.DataFrame([thr_row(y_tr, p_oof, t, "train_OOF") for t in THR])
tab_val = pd.DataFrame([thr_row(y_va, p_va, t, "validation") for t in THR])

best_f1 = tab_val.f1.max(); cand = tab_val[np.isclose(tab_val.f1, best_f1, atol=1e-9)]
T_SEL = float(cand.iloc[(cand.threshold - 0.5).abs().argmin()].threshold)               # tie-break: closest to 0.50
sel_val = tab_val[tab_val.threshold == T_SEL].iloc[0]
sensible = bool(sel_val.precision >= 0.60 and sel_val.recall_sensitivity >= 0.60 and abs(sel_val.predicted_placement_rate - sel_val.observed_placement_rate) <= 0.10)
T_OOF = float(tab_oof.loc[tab_oof.f1.idxmax(), "threshold"])
print(f"Selected operating threshold (max validation F1): {T_SEL:.2f}   | validation F1 {best_f1:.4f}   | passes sensibility check: {sensible}")
print(f"F1-optimal threshold on OOF (independent check, not used for selection): {T_OOF:.2f}")

tab = pd.concat([tab_oof, tab_val], ignore_index=True)
tab["is_selected_threshold"] = np.isclose(tab.threshold, T_SEL); tab["is_default_0.50"] = np.isclose(tab.threshold, 0.5)
tab.to_csv(MET / "07_threshold_analysis.csv", index=False)
show = ["threshold", "accuracy", "precision", "recall_sensitivity", "f1", "specificity", "false_positive_rate", "false_negative_rate", "TP", "TN", "FP", "FN", "predicted_placement_rate"]
print("\nVALIDATION (n = 1,487; observed placement rate %.3f):" % BASE_RATE_VA); display(tab_val[show].round(3).set_index("threshold"))
print("TRAIN OOF (n = 6,938; observed placement rate %.3f):" % BASE_RATE_TR); display(tab_oof[show].round(3).set_index("threshold"))
""")
code(r"""
# Stability of the selection: plateau, neighbours, bootstrap vs 0.50 (validation) and per-fold differences (OOF)
rng = np.random.default_rng(SEED); IDXV = rng.integers(0, len(y_va), (B_BOOT, len(y_va)))
f1_sel_b, f1_def_b = f1_boot(y_va, p_va >= T_SEL, IDXV), f1_boot(y_va, p_va >= 0.5, IDXV)
d = f1_sel_b - f1_def_b; ci = np.percentile(d, [2.5, 97.5])
f1_val = tab_val.set_index("threshold").f1; f1_oof = tab_oof.set_index("threshold").f1
plateau = tab_val[tab_val.f1 >= best_f1 - 0.005].threshold.tolist()
fold_rows = []
for k in range(1, K + 1):
    m = fold_of == k
    for t in sorted({0.5, T_SEL}): fold_rows.append({"fold": k, "threshold": t, "f1": thr_row(y_tr[m], p_oof[m], t, "oof_fold")["f1"]})
fr = pd.DataFrame(fold_rows).pivot(index="fold", columns="threshold", values="f1")
fold_diff = (fr[T_SEL] - fr[0.5]) if T_SEL != 0.5 else (fr[0.5] * 0)
summary = {"criterion": "grid threshold (0.20-0.80, step 0.05) with the highest VALIDATION F1; ties -> closest to 0.50", "selected_threshold": T_SEL,
           "validation_f1_selected": float(f1_val[T_SEL]), "validation_f1_at_0.50": float(f1_val[0.5]), "bootstrap_dF1_selected_minus_0.50": float(d.mean()),
           "bootstrap_dF1_ci95": [float(ci[0]), float(ci[1])], "default_0.50_practically_tied": bool(ci[0] <= 0 <= ci[1]),
           "plateau_thresholds_within_0.005_F1": plateau, "f1_neighbours_validation": {str(t): float(f1_val[t]) for t in THR if abs(t - T_SEL) <= 0.051},
           "sensibility_check_passed": sensible, "oof_f1_optimal_threshold": T_OOF, "oof_f1_at_selected": float(f1_oof[T_SEL]), "oof_f1_at_0.50": float(f1_oof[0.5]),
           "oof_fold_f1_difference_selected_minus_0.50": [float(x) for x in fold_diff], "oof_folds_selected_better": int((fold_diff > 0).sum())}
with open(MET / "07_threshold_selection.json", "w", encoding="utf-8") as fh: json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
for ax, (tb, ttl) in zip(axes, [(tab_oof, "Training out-of-fold (n = 6,938)"), (tab_val, "Validation (n = 1,487)")]):
    for col, c, lab in [("precision", "#D55E00", "Precision"), ("recall_sensitivity", "#0072B2", "Recall (sensitivity)"), ("f1", "#1d2733", "F1"), ("specificity", "#009E73", "Specificity")]:
        ax.plot(tb.threshold, tb[col], "o-", color=c, lw=1.9 if col == "f1" else 1.5, ms=4, label=lab)
    ax.axvline(0.5, color=NEUTRAL, ls=":", lw=1.3, label="Default 0.50")
    ax.axvline(T_SEL, color="#CC79A7", ls="--", lw=1.6, label=f"Selected {T_SEL:.2f} (max validation F1)")
    ax.set_title(ttl); ax.set_xlabel("Decision threshold (predict Placed if p >= threshold)"); ax.set_xlim(0.18, 0.82); ax.set_ylim(0.3, 1.02)
axes[0].set_ylabel("Metric value"); axes[0].legend(frameon=False, fontsize=8.8, loc="lower center")
fig.suptitle("Threshold trade-off for logistic regression B (the model is not refitted; only the cut-off changes)", y=1.02, fontsize=13, fontweight="semibold")
savefig(fig, "phase6_threshold_metrics")

reps = sorted({0.30, 0.50, T_SEL, 0.70}); fig, axes = plt.subplots(2, len(reps), figsize=(3.6 * len(reps), 7.2))
for r, (y_, p_, nm) in enumerate([(y_va, p_va, "Validation"), (y_tr, p_oof, "Train OOF")]):
    for j, t in enumerate(reps):
        tp, tn, fp, fn = confusion(y_, p_, t); cm = np.array([[tn, fp], [fn, tp]]); ax = axes[r, j]
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm): ax.text(b, a, f"{v:,}\n({v / cm[a].sum():.0%})", ha="center", va="center", fontsize=10, color="white" if v / cm[a].sum() > 0.55 else "#1d2733")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred NotPlaced", "Pred Placed"], fontsize=8.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["NotPlaced", "Placed"], fontsize=8.5)
        tag = " (selected)" if np.isclose(t, T_SEL) else (" (default)" if np.isclose(t, 0.5) else ""); ax.set_title(f"{nm}: threshold {t:.2f}{tag}", fontsize=10.5); ax.grid(False)
fig.suptitle("Confusion matrices (cell % = share of the actual class)", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "phase6_confusion_matrices")
""")

md(r"""
# PART B - Logistic regression explainability
## B1. Coefficients and odds ratios
The coefficients below are those of the **fitted Phase 4 pipeline** (L2-penalised, C = 1). Continuous features were standardised with the training mean/SD, so their odds ratio is **per one training standard deviation**; count and binary features are one-hot, so their odds ratio is **relative to the stated reference category**. Odds ratios of different kinds are therefore *not directly comparable in size*.

**Confidence intervals.** The penalised fit has no exact standard errors. The 95% intervals shown are **Wald intervals from an unpenalised logistic regression (maximum likelihood) on exactly the same transformed training design**; with n = 6,938 and 15 parameters the L2 penalty at C = 1 shifts coefficients very little (the maximum absolute difference is reported). They are conditional on the chosen representation and reference levels, ignore selection of that representation, and treat training rows as independent. Their purpose is interpretability, not hypothesis testing.
""")
code(r"""
names = list(pre.get_feature_names_out()); coef = lrm.coef_[0].copy(); intercept = float(lrm.intercept_[0])
scaler = pre.named_transformers_["num"]; sd_of = dict(zip(CONT, scaler.scale_)); mean_of = dict(zip(CONT, scaler.mean_))     # scale_ is the ddof=0 SD actually used
Xt_tr = pre.transform(X_tr); assert list(Xt_tr.columns) == names

def base_of(nm):
    hits = [c for c in FEATURES if nm == c or nm.startswith(c + "_")]; return max(hits, key=len)
rows = []
for nm, b in zip(names, coef):
    base = base_of(nm)
    if base in CONT:
        rep, ref, unit, n_lvl, sd = "standardised continuous", "- (train mean)", "per 1 training SD", np.nan, sd_of[base]; orig = float(np.exp(b / sd))
    else:
        lvl = nm[len(base) + 1:]; rep = "one-hot count level" if base in COUNT else "binary"; ref = str(REFS[base]) if base in COUNT else "No"
        unit = f"{base} = {lvl} vs {ref}"; n_lvl = int((Xt_tr[nm] == 1).sum()); sd = np.nan; orig = np.nan
    rows.append({"feature": nm, "base_feature": base, "representation": rep, "reference_category": ref, "effect_unit": unit, "train_sd": sd, "n_train_with_level": n_lvl,
                 "coefficient_log_odds": b, "odds_ratio": float(np.exp(b)), "odds_ratio_per_original_unit": orig,
                 "direction": "associated with HIGHER predicted placement odds" if b > 0 else "associated with LOWER predicted placement odds"})
coefs = pd.DataFrame(rows)

# unpenalised MLE on the identical design -> Wald 95% intervals
mle = sm.Logit(y_tr, sm.add_constant(Xt_tr, has_constant="add")).fit(disp=0)
assert mle.mle_retvals["converged"]
mci = mle.conf_int(); mci.columns = ["lo", "hi"]
coefs["mle_coefficient"] = [float(mle.params[n]) for n in names]
coefs["ci95_lo_log_odds"] = [float(mci.loc[n, "lo"]) for n in names]; coefs["ci95_hi_log_odds"] = [float(mci.loc[n, "hi"]) for n in names]
coefs["odds_ratio_ci95_lo"] = np.exp(coefs.ci95_lo_log_odds); coefs["odds_ratio_ci95_hi"] = np.exp(coefs.ci95_hi_log_odds)
coefs["ci_includes_OR_1"] = (coefs.odds_ratio_ci95_lo <= 1) & (coefs.odds_ratio_ci95_hi >= 1)
coefs["small_level_flag"] = coefs.n_train_with_level < 50
coefs["abs_coefficient"] = coefs.coefficient_log_odds.abs()
coefs = coefs.sort_values("abs_coefficient", ascending=False).reset_index(drop=True); coefs["rank_by_abs_coefficient"] = np.arange(1, len(coefs) + 1)
maxdiff = float((coefs.coefficient_log_odds - coefs.mle_coefficient).abs().max())
coefs.drop(columns="abs_coefficient").to_csv(MET / "07_logistic_coefficients.csv", index=False)
print(f"Intercept (reference profile at the training mean, all one-hot levels at reference): {intercept:.3f} (log-odds)")
maxdiff_ex = float((coefs.coefficient_log_odds - coefs.mle_coefficient).abs()[coefs.feature != "Projects_0"].max())
print(f"Max |penalised - unpenalised| coefficient difference: {maxdiff:.4f} log-odds overall, but {maxdiff_ex:.4f} excluding Projects_0 (n = 19 training students, shrunk by the L2 penalty).")
print("-> for every level with a reasonable sample the unpenalised intervals are a close guide to the penalised fit; for Projects_0 the interval is wide and the penalised estimate is pulled toward 0.")
display(coefs[["rank_by_abs_coefficient", "feature", "effect_unit", "coefficient_log_odds", "odds_ratio", "odds_ratio_ci95_lo", "odds_ratio_ci95_hi", "odds_ratio_per_original_unit", "n_train_with_level", "small_level_flag"]].round(3).set_index("rank_by_abs_coefficient"))
""")
code(r"""
fig, ax = plt.subplots(figsize=(10.4, 7.2)); d_ = coefs.iloc[::-1].reset_index(drop=True); ypos = np.arange(len(d_))
colmap = {"standardised continuous": "#0072B2", "one-hot count level": "#D55E00", "binary": "#009E73"}
ax.set_xscale("log"); ax.set_xlim(0.45, 16); XT = 10.5            # value labels sit in a fixed right-hand column
for i, r in d_.iterrows():
    c = colmap[r.representation]
    ax.plot([r.odds_ratio_ci95_lo, r.odds_ratio_ci95_hi], [i, i], color=c, lw=2.2, alpha=0.55, solid_capstyle="round")
    ax.scatter(r.odds_ratio, i, color=c, s=48, zorder=3, edgecolor="white", linewidth=0.7, marker="o" if not r.small_level_flag else "X")
    ax.text(XT, i, f"OR {r.odds_ratio:.2f}  [{r.odds_ratio_ci95_lo:.2f}, {r.odds_ratio_ci95_hi:.2f}]" + ("  n=%d, unreliable" % r.n_train_with_level if r.small_level_flag else ""), va="center", fontsize=8.2, color="#1d2733")
ylabels = [(f"{r.base_feature} (per 1 SD)" if r.representation == "standardised continuous" else r.effect_unit) for _, r in d_.iterrows()]
ax.axvline(1, color="#1d2733", lw=1); ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8.8)
ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
ax.set_xlabel("Odds ratio (log scale); dot = penalised model, line = 95% Wald interval of the unpenalised fit")
ax.set_title("Logistic regression B: associations with predicted placement odds (not causal effects)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=k) for k, c in colmap.items()], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=9); ax.grid(axis="y", visible=False)
savefig(fig, "phase6_logistic_coefficients")
""")

md(r"""
## B2. Permutation importance (validation set)
Each of the **10 raw input columns** is shuffled (the three count variables are permuted as whole columns, not level by level) in the *full pipeline*, 20 repeats, `random_state = 42`, and the drop in performance is recorded. Scores: **ROC-AUC** (primary; importance = AUC lost) and **log loss** (secondary; importance = log loss *added*, so positive = worse after shuffling). Evaluated on validation, never train. Permutation importance measures how much the **fitted model relies on a column for prediction**; it does not establish causation, and with correlated predictors (e.g. aptitude / HSC marks, r about 0.56) credit is shared, so single values are not exclusive contributions.
""")
code(r"""
t0 = time.time()
pi = permutation_importance(LR, X_va, y_va, scoring={"roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss"}, n_repeats=20, random_state=SEED, n_jobs=1)
perm = pd.DataFrame({"feature": FEATURES,
                     "auc_importance_mean": pi["roc_auc"]["importances_mean"], "auc_importance_sd": pi["roc_auc"]["importances_std"],
                     "logloss_importance_mean": pi["neg_log_loss"]["importances_mean"], "logloss_importance_sd": pi["neg_log_loss"]["importances_std"]})
perm["rank_by_auc"] = perm.auc_importance_mean.rank(ascending=False, method="min").astype(int)
perm["rank_by_logloss"] = perm.logloss_importance_mean.rank(ascending=False, method="min").astype(int)
perm["n_repeats"] = 20; perm["baseline_val_auc"] = roc_auc_score(y_va, p_va); perm["baseline_val_log_loss"] = log_loss(y_va, p_va)
perm = perm.sort_values("auc_importance_mean", ascending=False).reset_index(drop=True); perm.to_csv(MET / "07_permutation_importance.csv", index=False)
print(f"done in {time.time() - t0:.0f}s | baseline validation AUC {perm.baseline_val_auc[0]:.4f}, log loss {perm.baseline_val_log_loss[0]:.4f}")
display(perm.round(4).set_index("feature"))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True); d_ = perm.iloc[::-1]
for ax, (mc, sc, ttl, xl) in zip(axes, [("auc_importance_mean", "auc_importance_sd", "ROC-AUC lost when the column is shuffled", "Decrease in validation ROC-AUC"),
                                        ("logloss_importance_mean", "logloss_importance_sd", "Log loss added when the column is shuffled", "Increase in validation log loss")]):
    ax.barh(d_.feature, d_[mc], xerr=d_[sc], color=LR_C, alpha=0.85, ecolor="#1d2733", capsize=3); ax.set_xlabel(xl); ax.set_title(ttl); ax.grid(axis="y", visible=False)
fig.suptitle("Permutation importance of logistic regression B on validation (mean +/- SD over 20 repeats): predictive reliance, not causation", y=1.02, fontsize=12.5, fontweight="semibold")
savefig(fig, "phase6_permutation_importance")
""")

md(r"""
## B3. Global comparison: coefficients vs permutation importance
The two methods answer different questions, and the coefficient table mixes units (per SD, versus a level contrast). To put coefficients on a common footing per raw feature, each feature's **contribution to the linear predictor** is computed for every validation student, `sum of (coefficient x transformed value)` over the feature's columns, and its **standard deviation across students** is reported. For a standardised feature this is |coefficient| (times its z-score SD, about 1); for a one-hot feature it accounts for how common each level is, so a rare level (e.g. Projects = 0, 19 training students) cannot dominate. This is a descriptive summary of the fitted model, not a causal ranking.
""")
code(r"""
Xt_va = pre.transform(X_va); assert np.allclose(Xt_va.to_numpy() @ coef + intercept, LR.decision_function(X_va), atol=1e-9)
contrib_cols = pd.DataFrame(Xt_va.to_numpy() * coef, columns=names, index=X_va.index)
contrib_va = pd.DataFrame({f: contrib_cols[[n for n in names if base_of(n) == f]].sum(axis=1) for f in FEATURES})
assert np.allclose(contrib_va.sum(axis=1) + intercept, LR.decision_function(X_va), atol=1e-9)
gc = perm.set_index("feature")[["auc_importance_mean", "auc_importance_sd", "rank_by_auc", "logloss_importance_mean", "rank_by_logloss"]].copy()
gc["contribution_sd_validation"] = contrib_va.std(ddof=0)[gc.index]; gc["rank_by_contribution_sd"] = gc.contribution_sd_validation.rank(ascending=False, method="min").astype(int)
def coef_summary(f):
    r = coefs[coefs.base_feature == f]
    if f in CONT: return f"{r.coefficient_log_odds.iloc[0]:+.2f} log-odds per 1 SD (OR {r.odds_ratio.iloc[0]:.2f})"
    return "; ".join(f"{x.effect_unit.split(' = ')[1]}: OR {x.odds_ratio:.2f}" for x in r.sort_values("feature").itertuples())
gc["coefficient_summary"] = [coef_summary(f) for f in gc.index]
gc = gc.sort_values("rank_by_auc"); gc.reset_index().to_csv(MET / "07_global_explainability_comparison.csv", index=False)
rho = stats.spearmanr(gc.rank_by_contribution_sd, gc.rank_by_auc)
print(f"Spearman correlation between the two rankings (10 features; only a rough guide): rho = {rho.statistic:.2f}")
display(gc[["rank_by_auc", "auc_importance_mean", "rank_by_logloss", "rank_by_contribution_sd", "contribution_sd_validation", "coefficient_summary"]].round(4))
""")

md(r"""
# PART C - Individual validation-student explanations
For logistic regression the log-odds are exactly additive: `log-odds = intercept + sum_j coefficient_j x transformed_value_j`. A feature's **contribution** is the sum of its terms; it is measured **relative to a baseline profile**: the *training mean* for standardised features and the *reference level* for one-hot features. A positive contribution means the value **raises the model's predicted log-odds** of placement relative to that baseline; it does not mean the value causes placement.

**Six cases, chosen by fixed deterministic rules** (ties -> lowest row index; all six are distinct validation students):
1. *Very high probability* - correctly classified Placed student closest to the 95th percentile of validation probabilities.
2. *Very low probability* - correctly classified NotPlaced student closest to the 5th percentile.
3. *Decision boundary* - closest to p = 0.50.
4. *False positive* - the **most confident** false positive at 0.50 (highest p among NotPlaced). Deliberately the extreme error, not a typical one.
5. *False negative* - the **most confident** false negative at 0.50 (lowest p among Placed).
6. *Middle / uncertain* - closest to the median validation probability.
""")
code(r"""
used = set(); pick = {}
def choose(label, mask, target=None, extreme=None):
    cand = np.where(mask)[0]; cand = [i for i in cand if i not in used]
    if extreme == "max": i = max(cand, key=lambda j: (p_va[j], -j))
    elif extreme == "min": i = min(cand, key=lambda j: (p_va[j], j))
    else: i = min(cand, key=lambda j: (abs(p_va[j] - target), j))
    used.add(i); pick[label] = int(i)
q95, q05, med = np.quantile(p_va, 0.95), np.quantile(p_va, 0.05), np.median(p_va)
choose("1 very high probability", (y_va == 1) & (p_va >= 0.5), target=q95)
choose("2 very low probability", (y_va == 0) & (p_va < 0.5), target=q05)
choose("3 decision boundary", np.ones(len(y_va), bool), target=0.5)
choose("4 false positive (most confident)", (y_va == 0) & (p_va >= 0.5), extreme="max")
choose("5 false negative (most confident)", (y_va == 1) & (p_va < 0.5), extreme="min")
choose("6 middle / uncertain", np.ones(len(y_va), bool), target=med)
assert len(set(pick.values())) == 6
rules = {"1 very high probability": f"correct Placed closest to 95th percentile of p ({q95:.3f})", "2 very low probability": f"correct NotPlaced closest to 5th percentile of p ({q05:.3f})",
         "3 decision boundary": "closest to p = 0.50", "4 false positive (most confident)": "highest p among NotPlaced (p >= 0.50)",
         "5 false negative (most confident)": "lowest p among Placed (p < 0.50)", "6 middle / uncertain": f"closest to median validation p ({med:.3f})"}
long_rows, sum_rows = [], []
for lab, i in pick.items():
    p_i, y_i = p_va[i], y_va[i]; cv = contrib_va.iloc[i]; assert abs(intercept + cv.sum() - LR.decision_function(X_va.iloc[[i]])[0]) < 1e-9
    for f in FEATURES:
        long_rows.append({"case": lab, "val_row_id": i, "feature": f, "input_value": X_va.iloc[i][f], "contribution_log_odds": float(cv[f]),
                          "effect_on_predicted_log_odds": "raises" if cv[f] > 0.05 else ("lowers" if cv[f] < -0.05 else "negligible (|c| <= 0.05)")})
    long_rows.append({"case": lab, "val_row_id": i, "feature": "(intercept: reference profile)", "input_value": np.nan, "contribution_log_odds": intercept, "effect_on_predicted_log_odds": "baseline"})
    pos = cv[cv > 0.05].sort_values(ascending=False).head(3); neg = cv[cv < -0.05].sort_values().head(3)
    fmt = lambda s: "; ".join(f"{f} = {X_va.iloc[i][f]:g} ({c:+.2f})" if not isinstance(X_va.iloc[i][f], str) else f"{f} = {X_va.iloc[i][f]} ({c:+.2f})" for f, c in s.items()) or "none"
    sum_rows.append({"case": lab, "selection_rule": rules[lab], "val_row_id": i, "actual": "Placed" if y_i else "NotPlaced", "predicted_probability": p_i, "total_log_odds": intercept + cv.sum(),
                     "class_at_0.50": "Placed" if p_i >= 0.5 else "NotPlaced", f"class_at_selected_{T_SEL:.2f}": "Placed" if p_i >= T_SEL else "NotPlaced",
                     "major_positive_contributions": fmt(pos), "major_negative_contributions": fmt(neg),
                     **{f"input_{f}": X_va.iloc[i][f] for f in FEATURES}})
expl_long, expl_sum = pd.DataFrame(long_rows), pd.DataFrame(sum_rows)
expl_long.to_csv(MET / "07_individual_explanations.csv", index=False); expl_sum.to_csv(MET / "07_individual_explanation_summary.csv", index=False)
print("Saved long-format contributions (07_individual_explanations.csv) and one-row-per-case summary.")
pd.set_option("display.max_colwidth", 140)
display(expl_sum[["case", "actual", "predicted_probability", "class_at_0.50", f"class_at_selected_{T_SEL:.2f}", "major_positive_contributions", "major_negative_contributions"]].round(3).set_index("case"))
display(expl_sum[["case"] + [f"input_{f}" for f in FEATURES]].set_index("case"))
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True)
lim = np.abs(contrib_va.to_numpy()).max() * 1.05
for ax, (lab, i) in zip(axes.ravel(), pick.items()):
    cv = contrib_va.iloc[i].sort_values(); ax.barh(cv.index, cv.values, color=np.where(cv.values >= 0, PLACED_C, NOT_C), alpha=0.9)
    ax.axvline(0, color="#1d2733", lw=1); ax.set_xlim(-lim, lim); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"{lab}\np = {p_va[i]:.2f}, actual {'Placed' if y_va[i] else 'NotPlaced'}", fontsize=10)
for ax in axes[1]: ax.set_xlabel("Contribution to predicted log-odds\n(vs training mean / reference level)")
fig.suptitle("Why the model gave these six validation students their probabilities (blue raises, orange lowers the predicted log-odds; association, not cause)", y=1.01, fontsize=12.5, fontweight="semibold")
savefig(fig, "phase6_individual_explanations")
""")

md(r"""
# PART D - Feature-group ablation (logistic regression)
Five logistic regressions, each with the **same preprocessing philosophy and regularisation as `LR_B`** (L2, C = 1; continuous scores standardised; counts one-hot with the fixed Phase 4 reference levels 1 / 3 / 0; binary variables one-hot with reference `No`; transformers are simply omitted for groups that lack that kind of feature). IDs are `ABL_A`...`ABL_E` to avoid confusion with the Phase 4 *Representations* A and B (`LR_B` is Representation B).

| ID | Group | Features |
|---|---|---|
| ABL_A | Academic only | CGPA, SSC_Marks, HSC_Marks |
| ABL_B | Academic + Aptitude | + AptitudeTestScore |
| ABL_C | Academic + Technical | + Internships, Projects, Workshops/Certifications |
| ABL_D | Employability only | SoftSkillsRating, ExtracurricularActivities, PlacementTraining |
| ABL_E | Full model | all 10 predictors (must reproduce `LR_B`) |

**Evaluation.** 5-fold stratified CV on TRAIN (the same folds as Phase 5), then fit on all of TRAIN and evaluate once on VALIDATION. F1 uses the default 0.50 threshold (no per-model threshold search). Validation 95% intervals and paired differences use 1,000 paired bootstrap resamples. **Caveats:** the groups overlap conceptually and are correlated, and a group's gain depends on what is already in the model (employability's gain is measured only on top of everything else, C to E; ABL_D gives its stand-alone value). Differences describe predictive improvement, not effects.
""")
code(r"""
GROUPS = {"ABL_A": ("Academic only", ["CGPA", "SSC_Marks", "HSC_Marks"]),
          "ABL_B": ("Academic + Aptitude", ["CGPA", "SSC_Marks", "HSC_Marks", "AptitudeTestScore"]),
          "ABL_C": ("Academic + Technical", ["CGPA", "SSC_Marks", "HSC_Marks", "AptitudeTestScore", "Internships", "Projects", "Workshops/Certifications"]),
          "ABL_D": ("Employability only", ["SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining"]),
          "ABL_E": ("Full model", FEATURES)}
def make_lr(cols):
    cont, cnt, bn = [c for c in CONT if c in cols], [c for c in COUNT if c in cols], [c for c in BIN if c in cols]
    parts = []
    if cont: parts.append(("num", StandardScaler(), cont))
    if cnt: parts.append(("cnt", OneHotEncoder(drop=[REFS[c] for c in cnt], sparse_output=False, handle_unknown="error"), cnt))
    if bn: parts.append(("bin", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error"), bn))
    assert sorted(cont + cnt + bn) == sorted(cols)
    return Pipeline([("prep", ColumnTransformer(parts, verbose_feature_names_out=False).set_output(transform="pandas")),
                     ("model", LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=5000, random_state=SEED))])
for g, (_, cols) in GROUPS.items(): print(g, len(cols), cols)

# the full model must reproduce Phase 4 / Phase 5 exactly -> proves "same preprocessing"
full = make_lr(FEATURES).fit(X_tr, y_tr)
assert np.allclose(full.predict_proba(X_va)[:, 1], p_va, atol=1e-9), "ABL_E does not reproduce the Phase 4 LR_B validation probabilities"
assert np.allclose(full.named_steps["model"].coef_, lrm.coef_, atol=1e-8)
print("ABL_E reproduces the Phase 4 LR_B validation probabilities and coefficients.")
""")
code(r"""
t0 = time.time(); fold_rows, oof_abl, val_p = [], {g: np.full(len(y_tr), np.nan) for g in GROUPS}, {}
for g, (lab, cols) in GROUPS.items():
    for k, (tr, te) in enumerate(outer, 1):
        pipe = make_lr(cols).fit(X_tr.iloc[tr][cols], y_tr[tr]); p = pipe.predict_proba(X_tr.iloc[te][cols])[:, 1]; oof_abl[g][te] = p
        fold_rows.append({"group_id": g, "group": lab, "fold": k, **metrics_all(p, y_tr[te])})
    val_p[g] = make_lr(cols).fit(X_tr[cols], y_tr).predict_proba(X_va[cols])[:, 1]
assert all(not np.isnan(oof_abl[g]).any() for g in GROUPS)
assert np.allclose(oof_abl["ABL_E"], p_oof, atol=1e-8), "ABL_E OOF does not reproduce Phase 5 logistic OOF"
print(f"ablation CV + validation fits done in {time.time() - t0:.0f}s; ABL_E OOF reproduces the Phase 5 logistic OOF probabilities.")
fr_ab = pd.DataFrame(fold_rows); fr_ab.to_csv(MET / "07_feature_ablation_fold_results.csv", index=False)

# validation bootstrap (paired: same resamples for every group)
MET_NAMES = ["roc_auc", "log_loss", "brier", "f1"]
def boot_metrics(p):
    out = {k: np.empty(B_BOOT) for k in MET_NAMES}; ll = -(y_va * np.log(np.clip(p, 1e-15, 1 - 1e-15)) + (1 - y_va) * np.log(1 - np.clip(p, 1e-15, 1 - 1e-15))); br = (p - y_va) ** 2
    out["log_loss"], out["brier"] = ll[IDXV].mean(1), br[IDXV].mean(1); out["f1"] = f1_boot(y_va, p >= 0.5, IDXV)
    out["roc_auc"] = np.array([roc_auc_score(y_va[ix], p[ix]) for ix in IDXV]); return out
BOOT = {g: boot_metrics(val_p[g]) for g in GROUPS}
rows = []
for g, (lab, cols) in GROUPS.items():
    sub = fr_ab[fr_ab.group_id == g]; vm = metrics_all(val_p[g], y_va)
    r = {"group_id": g, "group": lab, "n_features": len(cols), "features": ", ".join(cols)}
    for m in MET_NAMES:
        r[f"cv_{m}_mean"], r[f"cv_{m}_sd"] = sub[m].mean(), sub[m].std(); r[f"val_{m}"] = vm[m]
        r[f"val_{m}_ci_lo"], r[f"val_{m}_ci_hi"] = np.percentile(BOOT[g][m], [2.5, 97.5])
    r["val_accuracy"] = vm["accuracy"]; rows.append(r)
abl = pd.DataFrame(rows); abl.to_csv(MET / "07_feature_ablation.csv", index=False)
ref = metrics_all(np.full(len(y_va), BASE_RATE_TR), y_va)
print(f"Reference (always predict the train placement rate): validation AUC {ref['roc_auc']:.3f}, log loss {ref['log_loss']:.3f}, Brier {ref['brier']:.3f}")
cols_show = ["group", "n_features"] + [f"cv_{m}_mean" for m in MET_NAMES[:3]] + [f"cv_{m}_sd" for m in MET_NAMES[:3]] + ["cv_f1_mean"] + [f"val_{m}" for m in MET_NAMES]
display(abl.set_index("group_id")[cols_show].round(4))
""")
code(r"""
# Increments: validation paired-bootstrap differences and paired CV-fold differences
PAIRS = [("ABL_B", "ABL_A", "adding aptitude to academic"), ("ABL_C", "ABL_B", "adding technical features (counts) to academic + aptitude"),
         ("ABL_E", "ABL_C", "adding employability to academic + technical"), ("ABL_E", "ABL_A", "full vs academic only"), ("ABL_E", "ABL_D", "full vs employability only"),
         ("ABL_C", "ABL_A", "academic + aptitude + technical vs academic only")]
inc = []
for big, small, desc in PAIRS:
    row = {"comparison": f"{big} minus {small}", "description": desc}
    for m in ["roc_auc", "log_loss", "brier"]:
        dv = BOOT[big][m] - BOOT[small][m]; fv = fr_ab[fr_ab.group_id == big].sort_values("fold")[m].to_numpy() - fr_ab[fr_ab.group_id == small].sort_values("fold")[m].to_numpy()
        row[f"val_d_{m}"] = float(abl.set_index("group_id").loc[big, f"val_{m}"] - abl.set_index("group_id").loc[small, f"val_{m}"])
        row[f"val_d_{m}_ci_lo"], row[f"val_d_{m}_ci_hi"] = np.percentile(dv, [2.5, 97.5]); row[f"cv_d_{m}_mean"], row[f"cv_d_{m}_sd"] = fv.mean(), fv.std(ddof=1)
        row[f"cv_folds_improved_{m}"] = int((fv > 0).sum() if m == "roc_auc" else (fv < 0).sum())
    inc.append(row)
inc = pd.DataFrame(inc); inc.to_csv(MET / "07_feature_ablation_increments.csv", index=False)
display(inc.set_index("comparison")[["val_d_roc_auc", "val_d_roc_auc_ci_lo", "val_d_roc_auc_ci_hi", "cv_d_roc_auc_mean", "cv_folds_improved_roc_auc", "val_d_log_loss", "val_d_log_loss_ci_lo", "val_d_log_loss_ci_hi", "cv_d_log_loss_mean", "cv_folds_improved_log_loss", "val_d_brier"]].round(4))
print("(AUC: positive = bigger model better; log loss / Brier: negative = bigger model better. Folds improved = of 5.)")
""")
code(r"""
order = list(GROUPS); xs = np.arange(len(order)); lab = [f"{g}\n{GROUPS[g][0]}" for g in order]
def abl_panel(ax, m, ttl, better):
    cvm, cvs = abl[f"cv_{m}_mean"].to_numpy(), abl[f"cv_{m}_sd"].to_numpy(); v = abl[f"val_{m}"].to_numpy()
    lo, hi = abl[f"val_{m}_ci_lo"].to_numpy(), abl[f"val_{m}_ci_hi"].to_numpy()
    ax.errorbar(xs - 0.12, cvm, yerr=cvs, fmt="o", color=NEUTRAL, capsize=4, ms=6, label="5-fold CV on train (mean +/- SD)")
    ax.errorbar(xs + 0.12, v, yerr=[v - lo, hi - v], fmt="D", color=LR_C, capsize=4, ms=6, label="Validation (95% bootstrap CI)")
    for x, a in zip(xs, v): ax.text(x + 0.16, a, f"{a:.3f}", fontsize=8.5, va="center", ha="left", color=LR_C)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8.5); ax.set_title(f"{ttl} ({better})"); ax.set_xlim(-0.5, len(order) - 0.3)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2)); abl_panel(axes[0], "roc_auc", "ROC-AUC", "higher is better"); abl_panel(axes[1], "log_loss", "Log loss", "lower is better")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Feature-group ablation: logistic regression with the same preprocessing and regularisation (association, not causal contribution)", y=1.02, fontsize=12.5, fontweight="semibold")
savefig(fig, "phase6_feature_ablation")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.0)); abl_panel(axes[0], "brier", "Brier score", "lower is better"); abl_panel(axes[1], "f1", "F1 at threshold 0.50", "higher is better")
savefig(fig, "phase6_feature_ablation_brier_f1")
""")

md(r"""
# PART E - Error analysis (validation, `LR_B`)
Primary classification threshold **0.50**; **high-confidence** is pre-declared as `p >= 0.80` (predicts Placed) or `p <= 0.20` (predicts NotPlaced), matching the Phase 5 bands. Groups: false positives / false negatives (any confidence), correct high-confidence predictions, and the instructive **incorrect high-confidence** groups (HC false positives: predicted >= 0.80 but NotPlaced; HC false negatives: predicted <= 0.20 but Placed). Group sizes are small for the high-confidence errors, so feature patterns are reported descriptively with standardised differences and **no significance tests**.
""")
code(r"""
va = X_va.copy(); va["p"] = p_va; va["y"] = y_va
masks = {"TP (predicted Placed, Placed)": (p_va >= .5) & (y_va == 1), "TN (predicted NotPlaced, NotPlaced)": (p_va < .5) & (y_va == 0),
         "FP (predicted Placed, NotPlaced)": (p_va >= .5) & (y_va == 0), "FN (predicted NotPlaced, Placed)": (p_va < .5) & (y_va == 1),
         "HC_TP: p >= 0.80, Placed": (p_va >= .8) & (y_va == 1), "HC_TN: p <= 0.20, NotPlaced": (p_va <= .2) & (y_va == 0),
         "HC_FP: p >= 0.80, NotPlaced": (p_va >= .8) & (y_va == 0), "HC_FN: p <= 0.20, Placed": (p_va <= .2) & (y_va == 1)}
masks["HC_correct (HC_TP + HC_TN)"] = masks["HC_TP: p >= 0.80, Placed"] | masks["HC_TN: p <= 0.20, NotPlaced"]
masks["HC_incorrect (HC_FP + HC_FN)"] = masks["HC_FP: p >= 0.80, NotPlaced"] | masks["HC_FN: p <= 0.20, Placed"]
n_err, N = int(masks["FP (predicted Placed, NotPlaced)"].sum() + masks["FN (predicted NotPlaced, Placed)"].sum()), len(va)
rows = []
for g, m in masks.items():
    s = va[m]; r = {"group": g, "n": int(m.sum()), "pct_of_validation": 100 * m.mean(), "pct_of_all_errors": 100 * m.sum() / n_err if g.startswith(("FP", "FN", "HC_FP", "HC_FN", "HC_incorrect")) else np.nan,
                    "p_min": s.p.min(), "p_q25": s.p.quantile(.25), "p_median": s.p.median(), "p_q75": s.p.quantile(.75), "p_max": s.p.max()}
    for f in CONT + COUNT: r[f"mean_{f}"] = s[f].mean()
    for f in BIN: r[f"pct_{f}_Yes"] = 100 * (s[f] == "Yes").mean()
    rows.append(r)
err = pd.DataFrame(rows); err.to_csv(MET / "07_error_analysis.csv", index=False)
display(err.set_index("group").round(2).T)
""")
code(r"""
# Standardised differences between each error group and its correctly classified comparator on the same predicted side (numeric encoding of all 10 features)
num = va[FEATURES].copy()
for f in BIN: num[f] = (num[f] == "Yes").astype(int)
def smd(a, b):
    sa, sb = a.std(ddof=1), b.std(ddof=1); sp = np.sqrt((sa ** 2 + sb ** 2) / 2); return (a.mean() - b.mean()) / sp if sp > 0 else np.nan
CMP = [("HC_FP vs HC_TP", "HC_FP: p >= 0.80, NotPlaced", "HC_TP: p >= 0.80, Placed"), ("HC_FN vs HC_TN", "HC_FN: p <= 0.20, Placed", "HC_TN: p <= 0.20, NotPlaced"),
       ("FP vs TP", "FP (predicted Placed, NotPlaced)", "TP (predicted Placed, Placed)"), ("FN vs TN", "FN (predicted NotPlaced, Placed)", "TN (predicted NotPlaced, NotPlaced)")]
srows = []
for nm, a, b in CMP:
    for f in FEATURES:
        srows.append({"comparison": nm, "n_error_group": int(masks[a].sum()), "n_comparator": int(masks[b].sum()), "feature": f,
                      "mean_error_group": num.loc[masks[a], f].mean(), "mean_comparator": num.loc[masks[b], f].mean(), "standardised_difference": smd(num.loc[masks[a], f], num.loc[masks[b], f])})
smd_df = pd.DataFrame(srows); smd_df.to_csv(MET / "07_error_feature_comparison.csv", index=False)
piv = smd_df.pivot(index="feature", columns="comparison", values="standardised_difference").reindex(FEATURES)[[c[0] for c in CMP]]
print("Standardised mean difference (error group minus comparator; |d| < 0.2 small, ~0.5 medium, >= 0.8 large). n:", {nm: (int(masks[a].sum()), int(masks[b].sum())) for nm, a, b in CMP})
display(piv.round(2))

# Ceiling statistics: how wrong should a perfectly calibrated model be? (expected error = mean(min(p, 1-p)))
exp_err, obs_err = float(np.minimum(p_va, 1 - p_va).mean()), float(((p_va >= .5) != (y_va == 1)).mean())
hc_hi, hc_lo = p_va >= .8, p_va <= .2
ceiling = pd.DataFrame([
    {"quantity": "observed error rate at 0.50 (validation)", "value": obs_err},
    {"quantity": "expected error rate if probabilities were perfectly calibrated: mean(min(p, 1-p))", "value": exp_err},
    {"quantity": "n in high band (p >= 0.80)", "value": int(hc_hi.sum())}, {"quantity": "observed NotPlaced share in high band (error rate there)", "value": float(1 - y_va[hc_hi].mean())},
    {"quantity": "expected error in high band if calibrated: mean(1-p)", "value": float((1 - p_va[hc_hi]).mean())},
    {"quantity": "n in low band (p <= 0.20)", "value": int(hc_lo.sum())}, {"quantity": "observed Placed share in low band (error rate there)", "value": float(y_va[hc_lo].mean())},
    {"quantity": "expected error in low band if calibrated: mean(p)", "value": float(p_va[hc_lo].mean())},
    {"quantity": "share of validation students with 0.2 < p < 0.8", "value": float(((p_va > .2) & (p_va < .8)).mean())},
    {"quantity": "share of all errors that fall in 0.2 < p < 0.8", "value": float((((p_va >= .5) != (y_va == 1)) & (p_va > .2) & (p_va < .8)).sum() / max(1, ((p_va >= .5) != (y_va == 1)).sum()))}])
bins_ = np.linspace(0, 1, 26); h1, _ = np.histogram(p_va[y_va == 1], bins_, density=False); h0, _ = np.histogram(p_va[y_va == 0], bins_, density=False)
ceiling = pd.concat([ceiling, pd.DataFrame([{"quantity": "overlap coefficient of predicted-probability distributions, Placed vs NotPlaced (25 bins; 1 = identical)", "value": float(np.minimum(h1 / h1.sum(), h0 / h0.sum()).sum())}])], ignore_index=True)
ceiling.to_csv(MET / "07_error_ceiling.csv", index=False); display(ceiling.round(4))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]; B_ = np.linspace(0, 1, 26)
ax.hist(p_va[y_va == 0], bins=B_, color=NOT_C, alpha=0.65, label="Actual NotPlaced"); ax.hist(p_va[y_va == 1], bins=B_, color=PLACED_C, alpha=0.65, label="Actual Placed")
for v in (0.2, 0.5, 0.8): ax.axvline(v, color=NEUTRAL, ls="--" if v == 0.5 else ":", lw=1.3)
ax.set_xlabel("Predicted placement probability (validation)"); ax.set_ylabel("Students")
ax.text(0.205, ax.get_ylim()[1] * 0.93, f"HC FN\n(p<=0.2, Placed)\nn={masks['HC_FN: p <= 0.20, Placed'].sum()}", fontsize=8.5, va="top", color=PLACED_C)
ax.text(0.805, ax.get_ylim()[1] * 0.93, f"HC FP\n(p>=0.8, NotPlaced)\nn={masks['HC_FP: p >= 0.80, NotPlaced'].sum()}", fontsize=8.5, va="top", color=NOT_C)
ax.set_title("Errors live in the overlap: probability by actual outcome"); ax.legend(frameon=False, loc="upper center", fontsize=9)
ax = axes[1]; yy = np.arange(len(FEATURES))[::-1]
cmp_n = {nm: (int(masks[a].sum()), int(masks[b].sum())) for nm, a, b in CMP}
for off, (nm, c, mk) in zip((-0.16, 0.16), [("HC_FP vs HC_TP", NOT_C, "o"), ("HC_FN vs HC_TN", PLACED_C, "s")]):
    ax.scatter(piv[nm].reindex(FEATURES), yy + off, color=c, marker=mk, s=44, label=f"{nm} (n = {cmp_n[nm][0]} vs {cmp_n[nm][1]})", zorder=3)
ax.axvline(0, color="#1d2733", lw=1); ax.axvspan(-0.2, 0.2, color="#e3e6ea", alpha=0.6, zorder=0)
ax.set_yticks(yy); ax.set_yticklabels(FEATURES, fontsize=9); ax.set_xlabel("Standardised difference (error group minus correctly classified group on the same side)")
ax.set_title("How high-confidence errors differ in profile (small n: descriptive only)"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1); ax.grid(axis="y", visible=False)
savefig(fig, "phase6_error_analysis")
""")

md(r"""
# PART F - Does the dataset's structure limit predictive performance?
Evidence already established in Phases 1-3 (not re-derived here), read together with the results above:
- **Step-like and non-monotonic count variables.** Placement rates for Internships (33% / 33% / 69%), Workshops (28% / 21% / 73% / 56%) and Projects (60% with n = 30, 17%, 31%, 71%) are not smooth trends; they are steps that persist after stratifying on aptitude (Phase 3).
- **Bounded scores with hard floors and ceilings** (CGPA 6.5, aptitude 90, soft skills 4.8, SSC 55) and **very large, systematic effects across every predictor**.
- **Heavy overlap between classes** despite high AUC (Phase 4/5; see the overlap coefficient and the mid-probability error share in Part E) and **8 conflicting-label profiles** (16 rows; identical predictors, different outcome) concentrated in the top profile region, i.e. a label that depends on something the features do not capture, or noise.
- Logistic regression, gradient boosting and random forest are **practically tied** (Phase 4/5), so flexible models do not extract extra signal from interactions or steps beyond what categorical counts already supply.

The observed structure is **consistent with a rule-based or constructed dataset** (the dataset card gives no provenance, and nothing here establishes how the data were generated). If so, the attainable accuracy is bounded by the generating rule plus label noise, and model performance here should **not** be generalised automatically to real university placement populations, other campuses or other years.
""")

md("## Phase 6 conclusions")
findings = Path("src/phase6_findings.md").read_text(encoding="utf-8") if Path("src/phase6_findings.md").exists() else "_Findings pending._"
md(findings)

md("## Test-set audit")
code(r"""
data_ext = (".csv", ".parquet", ".json", ".joblib", ".pkl", ".feather", ".xlsx")
opened = sorted({os.path.normpath(p) for p in _OPENED if p.lower().endswith(data_ext)})
print("Data files opened during this notebook run:")
for p in opened: print("  ", p)
names_ = [os.path.basename(p).lower() for p in opened]
assert "test.csv" not in names_, "test.csv (processed or raw) was opened!"
assert not any(("raw" in Path(p).parts and "data" in Path(p).parts) for p in opened), "a raw data file was opened"
assert "modeling_dataset.csv" not in names_ and "campus_recruitment_combined_flagged.csv" not in names_
assert {"train.csv", "validation.csv"} <= set(names_)
assert "X_test" not in globals() and "y_test" not in globals() and "test" not in globals()
print("\nASSERTIONS PASSED: test.csv (raw or processed) was never opened. Threshold analysis used Phase 5 OOF + validation predictions; explanations, permutation importance and error analysis used validation; ablation used train CV + one validation evaluation.")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/07_threshold_explainability_ablation.ipynb")
print("wrote notebooks/07_threshold_explainability_ablation.ipynb")
