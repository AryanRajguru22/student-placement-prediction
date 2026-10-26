"""Builds notebooks/06_calibration_and_cross_validation.ipynb (Phase 5).
Final conclusions are read from src/phase5_findings.md so they can be written against the executed numbers."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 06 - Cross-Validation, Calibration and Model Stability (Phase 5)
**Questions**
1. Are the three leading models (logistic regression, gradient boosting, random forest) stable under cross-validation?
2. How well calibrated are their probabilities?
3. Does recalibration (sigmoid / isotonic) materially improve probability quality?
4. Is logistic regression with categorical counts still a defensible *provisional* choice?
5. Are there meaningful differences in probability quality between the models?

## Data policy (enforced in code)
| Split | Phase 5 use |
|---|---|
| TRAIN (`train.csv`, 6,938) | **all** cross-validation, out-of-fold (OOF) predictions, calibration analysis and calibrator fitting |
| VALIDATION (`validation.csv`, 1,487) | one controlled **confirmation** at the end (Part 7); not used for CV folds, not used to tune anything here |
| TEST (`test.csv`) | **off limits.** An audit hook *raises an error the moment any process-level `open()` touches a file named `test.csv`* (raw or processed), and the notebook ends by printing every data file opened and asserting the test set was never accessed. |

*Disclosure (carried over from earlier phases):* the Phase 2-3 EDA/statistics used all modelling rows to describe associations and no model was fitted or scored on test rows. The Phase 4 hyper-parameters were chosen on the validation split, so training-CV estimates for those configurations are not independent of validation.

## Models and why the decision tree is omitted
Leading configurations are taken **unchanged from Phase 4** by cloning the saved pipelines (`models/phase4/*.joblib`), so preprocessing and hyper-parameters are exactly those of Phase 4:
**LR_B** (logistic regression, categorical counts, C = 1), **GB_B** (gradient boosting, 300 trees, learning rate 0.05, depth 2) and **RF_B** (random forest, 300 trees, depth 10, min leaf 10, sqrt features).
The decision tree is excluded from the detailed analysis because Phase 4 showed it is clearly weaker (validation log loss 0.48-0.50 vs 0.44, with a paired CI excluding 0) and its probabilities take only a few coarse values, so it is not a candidate for the final probability model.
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
            raise PermissionError(f"PHASE 5 VIOLATION: attempt to open {p} (the test set is off limits)")
sys.addaudithook(_audit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, sklearn
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models" / "phase4"
MET, FIG = ROOT / "results" / "metrics" / "modeling", ROOT / "results" / "figures" / "modeling"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

SEED, K, B_BOOT, NB = 42, 5, 1000, 10          # random seed, CV folds, bootstrap resamples, calibration bins
EPS = 1e-6                                       # clipping for logit(p)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 90)
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12,
                                      "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
PLACED_C, NOT_C, NEUTRAL = "#2166ac", "#e08214", "#6b7785"
MODEL_IDS = ["LR_B", "GB_B", "RF_B"]
LABEL = {"LR_B": "Logistic regression (B)", "GB_B": "Gradient boosting (B)", "RF_B": "Random forest (B)"}
COL = {"LR_B": "#0072B2", "GB_B": "#D55E00", "RF_B": "#009E73"}             # same family colours as Phase 4
METH_C = {"original": "#1d2733", "sigmoid": "#E69F00", "isotonic": "#CC79A7"}
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); plt.show()
def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh: json.dump(obj, fh, indent=2, default=str)
print("scikit-learn", sklearn.__version__, "| seed", SEED, "| folds", K)
""")

md("## 1. Load TRAIN and VALIDATION only; clone the exact Phase 4 pipelines")
code(r"""
TARGET = "PlacementStatus"
FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
train = pd.read_csv(PROC / "train.csv"); val = pd.read_csv(PROC / "validation.csv")
for d in (train, val):
    assert d.isna().sum().sum() == 0 and set(d[TARGET]) == {"Placed", "NotPlaced"}
assert not (set(map(tuple, train[FEATURES].values)) & set(map(tuple, val[FEATURES].values)))
X_tr, X_va = train[FEATURES].copy(), val[FEATURES].copy()
y_tr = (train[TARGET] == "Placed").astype(int).to_numpy(); y_va = (val[TARGET] == "Placed").astype(int).to_numpy()
for c in COUNT: X_tr[c] = X_tr[c].astype(int); X_va[c] = X_va[c].astype(int)
train_row_id = np.arange(len(train))                 # stable id = 0-based row position in train.csv (dataset untouched)
print(f"train {X_tr.shape} (Placed {y_tr.mean():.4f}) | validation {X_va.shape} (Placed {y_va.mean():.4f})")

with open(MODELS / "model_configs.json", encoding="utf-8") as fh: P4 = json.load(fh)
BASE = {m: clone(joblib.load(MODELS / f"{m}.joblib")) for m in MODEL_IDS}      # UNFITTED clones with identical Phase 4 parameters
for m in MODEL_IDS:
    assert not hasattr(BASE[m].named_steps["model"], "classes_")
    print(f"{m}: {P4['models'][m]['config']}")
with open(MET / "05_provisional_selection.json", encoding="utf-8") as fh: print("\nPhase 4 provisional selection:", json.load(fh)["provisional_model_id"])
""")

md(r"""
## 2. Metric helpers
- **ROC-AUC, log loss, Brier score:** primary measures of ranking and probability quality. Accuracy/precision/recall/F1 (threshold 0.5, **not tuned**) are secondary.
- **Calibration intercept and slope** (Cox / logistic recalibration): fit the unpenalised logistic regression `logit P(Placed) = a + b * logit(p_hat)` on out-of-sample predictions.
  *Slope* **b**: 1 is ideal; **b < 1** means predictions are too extreme (over-dispersed: high probabilities too high and low ones too low, the usual signature of over-fitting); **b > 1** means predictions are too timid (under-dispersed). *Intercept* **a** (with slope estimated jointly) is hard to read alone because it depends on the slope, so the **calibration-in-the-large (CITL)** is also reported: the intercept when the slope is fixed at 1 (an offset model); CITL = 0 means the average predicted risk matches the observed rate, > 0 means the model under-predicts placement on average, < 0 over-predicts. These are summaries to interpret *in context* with the reliability curve, not mechanical pass/fail tests.
- **Expected calibration error (ECE):** `sum_b (n_b / N) * |mean predicted_b - observed rate_b|`. ECE depends on binning, so the primary version uses **10 equal-count (quantile) bins** (ties broken by stable sort), and an equal-width 10-bin version (empty bins skipped) is shown as a sensitivity check. ECE is supplementary; log loss, Brier and the slope/intercept are the primary calibration evidence.
- **ECE yardstick:** because a finite sample never gives ECE = 0, each ECE is shown next to `ece_q_null_mean` / `ece_q_null_p95`: the mean and 95th percentile of ECE obtained by simulating outcomes from the model's own probabilities (i.e. *perfect* calibration plus sampling noise only; 1,000 simulations). An observed ECE at or below the null 95th percentile is statistically unremarkable.
- **Reliability bins:** 10 equal-count bins, so every bin holds about N/10 observations (about 690 for the training OOF set, about 150 for validation) and no bin is tiny; Wilson 95% intervals show the sampling noise.
""")
code(r"""
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def quantile_bins(p, y, nb=NB):
    order = np.argsort(p, kind="mergesort"); rows = []
    for i, ch in enumerate(np.array_split(order, nb)):
        k, n = int(y[ch].sum()), len(ch); lo, hi = wilson(k, n)
        rows.append({"bin": i + 1, "n": n, "p_min": p[ch].min(), "p_max": p[ch].max(), "mean_pred": p[ch].mean(), "obs_rate": k / n,
                     "calibration_gap": p[ch].mean() - k / n, "ci_lo": lo, "ci_hi": hi})     # gap = mean predicted - observed (>0: over-prediction)
    return pd.DataFrame(rows)

def ece_quantile(p, y, nb=NB):
    b = quantile_bins(p, y, nb); return float((b.n / b.n.sum() * (b.mean_pred - b.obs_rate).abs()).sum())

def ece_uniform(p, y, nb=NB):
    idx = np.minimum((p * nb).astype(int), nb - 1); tot = 0.0
    for b in range(nb):
        m = idx == b
        if m.any(): tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)

def ece_null_summary(p, B=1000, seed=SEED):
    # ECE expected from sampling noise ALONE if the predicted probabilities were perfectly calibrated:
    # simulate outcomes y ~ Bernoulli(p) and recompute the equal-count-bin ECE (bins fixed by p). Gives ECE a yardstick.
    rng = np.random.default_rng(seed); chunks = np.array_split(np.argsort(p, kind="mergesort"), NB); N = len(p)
    mp = [p[c].mean() for c in chunks]; out = np.empty(B)
    for b in range(B):
        yb = (rng.random(N) < p).astype(int)
        out[b] = sum(len(c) / N * abs(m_ - yb[c].mean()) for c, m_ in zip(chunks, mp))
    return {"ece_q_null_mean": float(out.mean()), "ece_q_null_p95": float(np.percentile(out, 95))}

def cal_metrics(p, y):
    z = logit(p); n = len(y)
    citl = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=z).fit()
    joint = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    ci_c, ci_j = citl.conf_int(), joint.conf_int()
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "citl": float(citl.params[0]), "citl_ci_lo": float(ci_c[0][0]), "citl_ci_hi": float(ci_c[0][1]),
            "cal_intercept": float(joint.params[0]), "cal_slope": float(joint.params[1]), "slope_ci_lo": float(ci_j[1][0]), "slope_ci_hi": float(ci_j[1][1]),
            "ece_quantile": ece_quantile(p, y), "ece_uniform": ece_uniform(p, y), **ece_null_summary(p),
            "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}

def clf_metrics(p, y, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}
""")

md(r"""
## 3. Part 1-2: 5-fold stratified cross-validation and out-of-fold predictions (TRAIN only)
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. For each fold, the **whole Phase 4 pipeline** (preprocessing + model) is cloned and fitted on the fold's training portion only, then predicts the held-out fold. Concatenating the held-out predictions gives **one out-of-fold probability per training student, from a model that never saw that student**; all calibration diagnostics use these OOF probabilities.
Each fold model is trained on 80% of the training data (about 5,550 students), so OOF probabilities are from slightly less-trained models than the full-train models of Phase 4.
""")
code(r"""
outer = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED).split(X_tr, y_tr))
fold_of = np.empty(len(y_tr), int)
for k, (_, te) in enumerate(outer, 1): fold_of[te] = k
assert sorted(np.concatenate([te for _, te in outer])) == list(range(len(y_tr)))      # every training row held out exactly once
assert all(not (set(tr) & set(te)) for tr, te in outer)                                # no student is in both the fitting and the held-out part of any fold

OOF = {m: np.full(len(y_tr), np.nan) for m in MODEL_IDS}; fold_rows = []
t0 = time.time()
for k, (tr, te) in enumerate(outer, 1):
    for m in MODEL_IDS:
        pipe = clone(BASE[m]).fit(X_tr.iloc[tr], y_tr[tr])
        p = pipe.predict_proba(X_tr.iloc[te])[:, 1]; OOF[m][te] = p
        fold_rows.append({"model_id": m, "model": LABEL[m], "fold": k, "n_train": len(tr), "n_heldout": len(te), "heldout_placed_rate": y_tr[te].mean(), **clf_metrics(p, y_tr[te])})
assert all(not np.isnan(OOF[m]).any() for m in MODEL_IDS)
folds = pd.DataFrame(fold_rows); folds.to_csv(MET / "06_cv_fold_results.csv", index=False)
print(f"CV complete in {time.time() - t0:.0f}s; folds:", dict(zip(*np.unique(fold_of, return_counts=True))))

METRICS = ["roc_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1"]
summ = (folds.melt(id_vars=["model_id", "model", "fold"], value_vars=METRICS, var_name="metric")
        .groupby(["model_id", "model", "metric"], sort=False).value.agg(mean="mean", sd="std", min="min", max="max").reset_index())
summ["tier"] = np.where(summ.metric.isin(["roc_auc", "log_loss", "brier"]), "primary", "secondary")
summ.to_csv(MET / "06_cv_summary.csv", index=False)
wide = summ.pivot(index="metric", columns="model_id", values=["mean", "sd", "min", "max"]).reindex(METRICS)
print("5-fold CV on TRAIN (mean / sd / min / max across folds):")
display(wide.round(4))
""")
code(r"""
oof_df = pd.DataFrame({"train_row_id": train_row_id, "actual_target": y_tr, "logistic_probability": OOF["LR_B"],
                       "gradient_boosting_probability": OOF["GB_B"], "random_forest_probability": OOF["RF_B"], "fold": fold_of})
# OOF integrity: one row per training student, one fold each, no missing/out-of-range probabilities, and (with the disjoint-fold assertion
# above) each probability comes from a model whose fitting rows exclude that student
assert len(oof_df) == len(train) and oof_df.train_row_id.is_unique and set(oof_df.train_row_id) == set(range(len(train)))
assert oof_df.fold.between(1, K).all() and oof_df.fold.notna().all() and (oof_df.actual_target.to_numpy() == y_tr).all()
assert oof_df[["logistic_probability", "gradient_boosting_probability", "random_forest_probability"]].apply(lambda s: s.between(0, 1).all()).all()
assert not oof_df.isna().any().any()
oof_df.to_csv(MET / "06_oof_predictions.csv", index=False)
print("Saved OOF predictions:", oof_df.shape, "- integrity checks passed (each student exactly once, never scored by a model fitted on them)")
# sanity: OOF metrics on all training rows
pd.DataFrame({m: clf_metrics(OOF[m], y_tr) for m in MODEL_IDS}).T.round(4)
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8)); xs = np.arange(len(MODEL_IDS))
for ax, (met, ttl) in zip(axes, [("roc_auc", "ROC-AUC (higher is better)"), ("log_loss", "Log loss (lower is better)"), ("brier", "Brier score (lower is better)")]):
    piv = folds.pivot(index="fold", columns="model_id", values=met)[MODEL_IDS]
    for k in piv.index: ax.plot(xs, piv.loc[k].values, color="#c4cad2", lw=1, zorder=1)           # same fold connected across models (paired)
    for i, m in enumerate(MODEL_IDS):
        ax.scatter(np.full(len(piv), i), piv[m], color=COL[m], s=34, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.6)
        mu, sd = piv[m].mean(), piv[m].std()
        ax.errorbar(i + 0.22, mu, yerr=sd, fmt="D", color="#1d2733", mfc="white", capsize=4, ms=6, zorder=4)
        ax.text(i + 0.30, mu, f"{mu:.3f}\n+/-{sd:.3f}", fontsize=8, va="center")
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in MODEL_IDS], rotation=12, ha="right", fontsize=9); ax.set_xlim(-0.4, 2.9); ax.set_title(ttl)
fig.suptitle("5-fold cross-validation on TRAIN: dots = folds (grey lines join the same fold), diamond = mean +/- SD", y=1.02, fontsize=13, fontweight="semibold")
savefig(fig, "phase5_cv_comparison")
""")

md(r"""
## 4. Part 8: Is the Phase 4 tie stable? (paired fold differences)
All three models are evaluated on the **same five folds**, so differences are paired. For each metric the table gives Logistic minus the other model (negative = logistic regression better for log loss/Brier; positive = better for AUC).
Five folds give very little inferential precision, and fold training sets overlap, so the naive t-interval (SE = s/sqrt(5)) is too narrow. The **Nadeau-Bengio corrected** interval inflates the variance by (1/K + n_test/n_train) / (1/K) = 2.25 for 5-fold CV and is reported as the primary interval; both are shown. Treat them as rough guides, not precise inference.
""")
code(r"""
pd_rows = []
tcrit = stats.t.ppf(0.975, K - 1)
for other in ["GB_B", "RF_B"]:
    for met in ["roc_auc", "log_loss", "brier"]:
        a = folds[folds.model_id == "LR_B"].set_index("fold")[met]; b = folds[folds.model_id == other].set_index("fold")[met]
        d = (a - b).to_numpy(); mu, s = d.mean(), d.std(ddof=1)
        se_naive = s / np.sqrt(K); se_nb = s * np.sqrt(1 / K + (len(outer[0][1]) / len(outer[0][0])))
        pd_rows.append({"comparison": f"LR_B - {other}", "metric": met, "mean_diff": mu, "sd_diff": s, "folds_lr_better": int((d < 0).sum()) if met != "roc_auc" else int((d > 0).sum()),
                        "naive_ci_lo": mu - tcrit * se_naive, "naive_ci_hi": mu + tcrit * se_naive, "nb_ci_lo": mu - tcrit * se_nb, "nb_ci_hi": mu + tcrit * se_nb,
                        "nb_ci_includes_0": (mu - tcrit * se_nb) <= 0 <= (mu + tcrit * se_nb)})
paired = pd.DataFrame(pd_rows); paired.to_csv(MET / "06_cv_paired_differences.csv", index=False)
paired.round(4)
""")
code(r"""
print("Between-model spread of the fold means vs fold-to-fold SD (primary metrics):")
for met in ["roc_auc", "log_loss", "brier"]:
    mu = folds.groupby("model_id")[met].mean()[MODEL_IDS]; sd = folds.groupby("model_id")[met].std()[MODEL_IDS]
    print(f"  {met:9s} range of means = {mu.max() - mu.min():.4f}  | typical fold SD = {sd.mean():.4f}  | ratio = {(mu.max() - mu.min()) / sd.mean():.2f}")
""")

md(r"""
## 5. Part 3-4: Calibration of the original probabilities (training OOF predictions)
""")
code(r"""
cal_rows, bin_rows = [], []
for m in MODEL_IDS:
    cm = cal_metrics(OOF[m], y_tr); cal_rows.append({"model_id": m, "model": LABEL[m], "method": "original", "dataset": "train_OOF", "n": len(y_tr), **cm})
    b = quantile_bins(OOF[m], y_tr); b.insert(0, "model_id", m); b.insert(1, "method", "original"); b.insert(2, "dataset", "train_OOF"); bin_rows.append(b)
cal_orig = pd.DataFrame(cal_rows)
cols = ["model", "roc_auc", "log_loss", "brier", "citl", "citl_ci_lo", "citl_ci_hi", "cal_intercept", "cal_slope", "slope_ci_lo", "slope_ci_hi", "ece_quantile", "ece_q_null_mean", "ece_q_null_p95", "ece_uniform", "mean_pred", "obs_rate"]
print("Calibration of ORIGINAL probabilities, training OOF (out-of-sample) predictions:")
display(cal_orig.set_index("model_id")[cols[1:]].round(4).join(cal_orig.set_index("model_id")[["model"]]))
""")
code(r"""
bins_orig = pd.concat(bin_rows, ignore_index=True)
print("Reliability bins (10 equal-count bins of ~%d students each):" % (len(y_tr) // NB))
for m in MODEL_IDS:
    print(f"--- {LABEL[m]} ---"); display(bins_orig[bins_orig.model_id == m].drop(columns=["model_id", "method", "dataset"]).set_index("bin").round(3))
""")
code(r"""
# Reliability: overlay of the three original models
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2, label="Perfect calibration")
for m in MODEL_IDS:
    b = bins_orig[bins_orig.model_id == m]; ax.plot(b.mean_pred, b.obs_rate, "o-", color=COL[m], lw=1.8, ms=5, label=LABEL[m])
ax.set_xlabel("Mean predicted probability (equal-count bin)"); ax.set_ylabel("Observed placement rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title("Reliability of the three leading models (train OOF, 10 equal-count bins)"); ax.legend(frameon=False, loc="upper left", fontsize=9)
savefig(fig, "phase5_reliability_curves")

# Reliability per model with Wilson intervals and prediction histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), gridspec_kw={"height_ratios": [3.2, 1]}, sharex="col")
for j, m in enumerate(MODEL_IDS):
    b = bins_orig[bins_orig.model_id == m]; ax = axes[0, j]
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2)
    ax.errorbar(b.mean_pred, b.obs_rate, yerr=[b.obs_rate - b.ci_lo, b.ci_hi - b.obs_rate], fmt="o-", color=COL[m], ecolor=COL[m], capsize=3, lw=1.6, ms=5)
    cmx = cal_orig.set_index("model_id").loc[m]
    ax.text(0.04, 0.95, f"slope {cmx.cal_slope:.2f}  CITL {cmx.citl:+.2f}\nECE(q) {cmx.ece_quantile:.3f}  n/bin ~{int(b.n.mean())}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(LABEL[m]); ax.set_ylabel("Observed placement rate" if j == 0 else "")
    axes[1, j].hist(OOF[m], bins=np.linspace(0, 1, 26), color=COL[m], alpha=0.8); axes[1, j].set_xlabel("Predicted probability"); axes[1, j].set_ylabel("Students" if j == 0 else "")
fig.suptitle("Reliability diagrams for the original probabilities (train OOF; bars = Wilson 95% CI)", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "phase5_reliability_original")
""")

md(r"""
## 6. Part 5: Recalibration - nested, leakage-safe comparison of Original vs Sigmoid vs Isotonic (TRAIN only)
**Procedure (for every outer CV fold k and each model):**
1. Fit the pipeline on the outer-training portion and predict the held-out fold: these are the *original* probabilities (identical to the OOF predictions above; asserted).
2. *Inner* 5-fold CV **within the outer-training portion only**: obtain out-of-sample probabilities for every outer-training student.
3. Fit the calibration map on those inner out-of-sample probabilities: **sigmoid (Platt)** = unpenalised logistic regression of the outcome on `logit(p)` (two parameters: slope and intercept); **isotonic** = monotone non-decreasing step function (`IsotonicRegression`, out-of-bounds clipped; outputs clipped to [0.001, 0.999] to avoid infinite log loss).
4. Apply the map to the held-out fold's original probabilities.

The held-out fold is never used to fit the base model *or* the calibration map, so every calibrated training probability is genuinely out-of-sample; calibration metrics are computed on these pooled nested predictions and per fold. No validation or test data are used. Isotonic is flexible (many parameters) and can over-fit with small samples; sigmoid can only fix a monotone logistic-shaped miscalibration.
""")
code(r"""
def fit_sigmoid(p, y):
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000).fit(logit(p).reshape(-1, 1), y)
    f = lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]; f.params = (float(lr.intercept_[0]), float(lr.coef_[0, 0])); return f

def fit_isotonic(p, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(p, y)
    return lambda q: np.clip(iso.predict(q), 1e-3, 1 - 1e-3)

CAL = {m: {"original": OOF[m].copy(), "sigmoid": np.full(len(y_tr), np.nan), "isotonic": np.full(len(y_tr), np.nan)} for m in MODEL_IDS}
rfold, sig_params = [], []
t0 = time.time()
for k, (tr, te) in enumerate(outer, 1):
    inner = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED + k)
    for m in MODEL_IDS:
        p_in = np.full(len(tr), np.nan)
        for itr, ite in inner.split(X_tr.iloc[tr], y_tr[tr]):
            pipe = clone(BASE[m]).fit(X_tr.iloc[tr[itr]], y_tr[tr[itr]]); p_in[ite] = pipe.predict_proba(X_tr.iloc[tr[ite]])[:, 1]
        assert not np.isnan(p_in).any()
        sg, io = fit_sigmoid(p_in, y_tr[tr]), fit_isotonic(p_in, y_tr[tr])
        CAL[m]["sigmoid"][te] = sg(OOF[m][te]); CAL[m]["isotonic"][te] = io(OOF[m][te])
        sig_params.append({"model_id": m, "fold": k, "sigmoid_intercept": sg.params[0], "sigmoid_slope": sg.params[1]})
        for meth in ["original", "sigmoid", "isotonic"]:
            mm = clf_metrics(CAL[m][meth][te], y_tr[te]); rfold.append({"model_id": m, "method": meth, "fold": k, **mm})
print(f"Nested calibration done in {time.time() - t0:.0f}s")
assert all(not np.isnan(CAL[m][x]).any() for m in MODEL_IDS for x in ("sigmoid", "isotonic"))
recal_fold = pd.DataFrame(rfold); recal_fold.to_csv(MET / "06_recalibration_fold_results.csv", index=False)
sig_df = pd.DataFrame(sig_params); sig_df.to_csv(MET / "06_sigmoid_parameters_by_fold.csv", index=False)
print("Sigmoid map per outer fold, logit(p_cal) = intercept + slope * logit(p):"); display(sig_df.pivot(index="fold", columns="model_id", values="sigmoid_slope").round(3).add_prefix("slope_").join(sig_df.pivot(index="fold", columns="model_id", values="sigmoid_intercept").round(3).add_prefix("intercept_")))
""")
code(r"""
rows, bins_all = [], [bins_orig]
for m in MODEL_IDS:
    for meth in ["original", "sigmoid", "isotonic"]:
        p = CAL[m][meth]; cm = cal_metrics(p, y_tr)
        rows.append({"model_id": m, "model": LABEL[m], "method": meth, "dataset": "train_nested_CV", "n": len(y_tr), **cm,
                     "n_distinct_probabilities": int(len(np.unique(np.round(p, 8)))), "share_p_at_clip_bounds_%": float(((p <= 1e-3 + 1e-12) | (p >= 1 - 1e-3 - 1e-12)).mean() * 100)})
        if meth != "original":
            b = quantile_bins(p, y_tr); b.insert(0, "model_id", m); b.insert(1, "method", meth); b.insert(2, "dataset", "train_nested_CV"); bins_all.append(b)
# paired bootstrap of the change vs original (same resamples for all models/methods)
rng = np.random.default_rng(SEED); IDX = rng.integers(0, len(y_tr), (B_BOOT, len(y_tr)))
def ll_vec(p, y): q = np.clip(p, 1e-15, 1 - 1e-15); return -(y * np.log(q) + (1 - y) * np.log(1 - q))
rc = pd.DataFrame(rows)
for i, r in rc.iterrows():
    if r.method == "original": continue
    for met, vec in [("log_loss", lambda p: ll_vec(p, y_tr)), ("brier", lambda p: (p - y_tr) ** 2)]:
        d = (vec(CAL[r.model_id][r.method]) - vec(CAL[r.model_id]["original"]))[IDX].mean(axis=1)
        rc.loc[i, f"d_{met}"] = d.mean(); rc.loc[i, f"d_{met}_ci_lo"], rc.loc[i, f"d_{met}_ci_hi"] = np.percentile(d, [2.5, 97.5])
base_auc = rc[rc.method == "original"].set_index("model_id").roc_auc
rc["d_roc_auc"] = rc.apply(lambda r: np.nan if r.method == "original" else r.roc_auc - base_auc[r.model_id], axis=1)
rc.to_csv(MET / "06_recalibration_comparison.csv", index=False)
pd.concat(bins_all, ignore_index=True).to_csv(MET / "06_calibration_bins.csv", index=False)
show = ["model_id", "method", "roc_auc", "d_roc_auc", "log_loss", "d_log_loss", "d_log_loss_ci_lo", "d_log_loss_ci_hi", "brier", "d_brier", "d_brier_ci_lo", "d_brier_ci_hi", "citl", "cal_intercept", "cal_slope", "ece_quantile", "ece_q_null_mean", "ece_q_null_p95", "ece_uniform"]
print("Original vs sigmoid vs isotonic, TRAIN, nested leakage-safe (d_* = change vs original; negative log loss/Brier change = improvement):")
rc[show].set_index(["model_id", "method"]).round(4)
""")
code(r"""
# Calibration-metrics file requested for the original probabilities (+ recalibrated versions, clearly labelled)
cm_out = rc[["model_id", "model", "method", "dataset", "n", "roc_auc", "log_loss", "brier", "citl", "citl_ci_lo", "citl_ci_hi", "cal_intercept", "cal_slope", "slope_ci_lo", "slope_ci_hi",
             "ece_quantile", "ece_q_null_mean", "ece_q_null_p95", "ece_uniform", "mean_pred", "obs_rate"]]
cm_out.to_csv(MET / "06_calibration_metrics.csv", index=False)
print("AUC check - does recalibration change ranking?  (isotonic can create ties; sigmoid is monotone and should leave AUC unchanged)")
display(rc[rc.method != "original"].pivot(index="model_id", columns="method", values="d_roc_auc").round(5))
print("Isotonic: distinct output values and share of predictions at the clip bounds:")
display(rc[rc.method == "isotonic"].set_index("model_id")[["n_distinct_probabilities", "share_p_at_clip_bounds_%"]].round(2))
for m in MODEL_IDS:        # consistency with Part 2: the 'original' arm equals the OOF predictions
    assert np.allclose(CAL[m]["original"], OOF[m])
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
for ax, m in zip(axes, MODEL_IDS):
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2)
    for meth in ["original", "sigmoid", "isotonic"]:
        b = quantile_bins(CAL[m][meth], y_tr)
        ax.plot(b.mean_pred, b.obs_rate, "o-", color=METH_C[meth], lw=1.7, ms=4.5, label=meth, alpha=0.95)
    r_ = rc.set_index(["model_id", "method"])
    ax.text(0.04, 0.96, "slope / CITL / ECE(q)\n" + "\n".join(f"{x[:4]}: {r_.loc[(m, x), 'cal_slope']:.2f} / {r_.loc[(m, x), 'citl']:+.2f} / {r_.loc[(m, x), 'ece_quantile']:.3f}" for x in ["original", "sigmoid", "isotonic"]),
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Mean predicted probability (equal-count bin)"); ax.set_ylabel("Observed placement rate"); ax.set_title(LABEL[m]); ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.suptitle("Original vs recalibrated probabilities (train, nested out-of-sample; 10 equal-count bins)", y=1.02, fontsize=13, fontweight="semibold")
savefig(fig, "phase5_reliability_recalibrated")

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True)
BINS = np.linspace(0, 1, 26)
for j, m in enumerate(MODEL_IDS):
    p = OOF[m]
    axes[0, j].hist(p[y_tr == 0], bins=BINS, color=NOT_C, alpha=0.65, weights=np.full((y_tr == 0).sum(), 100 / (y_tr == 0).sum()), label="Actual NotPlaced")
    axes[0, j].hist(p[y_tr == 1], bins=BINS, color=PLACED_C, alpha=0.65, weights=np.full((y_tr == 1).sum(), 100 / (y_tr == 1).sum()), label="Actual Placed")
    axes[0, j].set_title(f"{LABEL[m]}: original, by actual outcome"); axes[0, j].set_ylabel("% of that class" if j == 0 else "")
    for meth in ["original", "sigmoid", "isotonic"]:
        axes[1, j].hist(CAL[m][meth], bins=BINS, histtype="step", lw=2, color=METH_C[meth], label=meth, weights=np.full(len(y_tr), 100 / len(y_tr)))
    axes[1, j].set_xlabel("Predicted probability"); axes[1, j].set_title("All students: original vs recalibrated"); axes[1, j].set_ylabel("% of students" if j == 0 else "")
axes[0, 0].legend(frameon=False, fontsize=8.5); axes[1, 0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Predicted-probability distributions (train, out-of-sample)", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "phase5_probability_distributions")
""")
md(r"""
**Distribution summary (original probabilities, train OOF).** A confident (extreme) probability is not evidence of a good prediction: it is only informative if the reliability diagram shows the observed rate matches it. For each model: quantiles of the predicted probability, the share of students in the low (< 0.2), middle (0.2-0.8) and high (> 0.8) bands, and - for the two extreme bands - the *observed* placement rate in that band, which is the check that the confidence is justified.
""")
code(r"""
drows = []
for m in MODEL_IDS:
    p = OOF[m]; low, mid, high = p < 0.2, (p >= 0.2) & (p <= 0.8), p > 0.8
    drows.append({"model_id": m, "model": LABEL[m], "min": p.min(), **{f"q{int(q * 100):02d}": np.quantile(p, q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)}, "max": p.max(),
                  "share_below_0.2": low.mean(), "share_0.2_to_0.8": mid.mean(), "share_above_0.8": high.mean(),
                  "observed_rate_below_0.2": y_tr[low].mean(), "observed_rate_0.2_to_0.8": y_tr[mid].mean(), "observed_rate_above_0.8": y_tr[high].mean(),
                  "mean_pred_below_0.2": p[low].mean(), "mean_pred_above_0.8": p[high].mean()})
dist = pd.DataFrame(drows); dist.to_csv(MET / "06_probability_distribution_summary.csv", index=False)
dist.set_index("model_id").drop(columns="model").round(3).T
""")

md(r"""
## 7. Part 6: Calibration-method decision (evidence, not sophistication)
**Pre-declared materiality rule.** A recalibration method is judged to *materially* improve probability quality for a model only if **both** hold in the nested train analysis: (i) the paired-bootstrap 95% CI for the change in log loss lies entirely below 0 **and** the point improvement is at least 0.005 (about 1% of the 0.44 log loss), and (ii) the Brier change is also negative with a CI below 0. The bootstrap resamples students from pooled OOF predictions and ignores fold dependence, so it slightly understates uncertainty; the per-fold consistency table below is a second check. Smaller changes are called *negligible*, whatever their p-value or CI. Calibration slope/intercept, the reliability curves, the sample size per parameter, and fold-to-fold stability are read alongside.
""")
code(r"""
sel = rc[rc.method != "original"].copy()
sel["material"] = (sel.d_log_loss <= -0.005) & (sel.d_log_loss_ci_hi < 0) & (sel.d_brier_ci_hi < 0)
rf_ = recal_fold.pivot_table(index=["model_id", "fold"], columns="method", values="log_loss")
for meth in ["sigmoid", "isotonic"]:
    d = (rf_[meth] - rf_["original"]).groupby("model_id")
    sel[f"folds_improved_{meth}"] = sel.model_id.map(d.apply(lambda s: int((s < 0).sum()))).where(sel.method == meth)
    sel[f"mean_fold_dLL_{meth}"] = sel.model_id.map(d.mean()).where(sel.method == meth)
decision = sel[["model_id", "method", "d_log_loss", "d_log_loss_ci_lo", "d_log_loss_ci_hi", "d_brier", "d_brier_ci_lo", "d_brier_ci_hi", "cal_slope", "citl", "ece_quantile", "material", "folds_improved_sigmoid", "folds_improved_isotonic"]]
decision.round(4).set_index(["model_id", "method"])
""")
code(r"""
orig = rc[rc.method == "original"].set_index("model_id")
print("Original slope / CITL / ECE(q):", {m: (round(orig.loc[m, 'cal_slope'], 3), round(orig.loc[m, 'citl'], 3), round(orig.loc[m, 'ece_quantile'], 4)) for m in MODEL_IDS})
for m in MODEL_IDS:
    s = sel[sel.model_id == m]; print(f"{m}: material recalibration -> {dict(zip(s.method, s.material))}")
decision.to_csv(MET / "06_calibration_method_decision.csv", index=False)
""")

md(r"""
## 8. Part 7: Validation-set confirmation (calibrators fitted on TRAIN only)
The base models are fitted on the **full training set** (as in Phase 4; predictions are asserted to match the saved Phase 4 validation predictions). Calibration maps are fitted **only on the training OOF probabilities** from section 3 (each from a model that did not see that student), then applied to the validation predictions. The validation set is used purely to evaluate. This is a *confirmation*, kept separate from the training-CV estimates, and the validation set remains the model-development set (no threshold tuning, no new search).
Note the small mismatch inherent in this standard recipe: OOF probabilities come from models trained on 80% of the training data, while the final models use 100%.
""")
code(r"""
p4 = pd.read_csv(MET / "05_validation_predictions.csv")            # Phase 4 validation predictions (validation data only)
assert (p4.y_true.to_numpy() == y_va).all()
VAL = {}; vrows, vbins = [], []
for m in MODEL_IDS:
    full = clone(BASE[m]).fit(X_tr, y_tr); p_raw = full.predict_proba(X_va)[:, 1]
    assert np.allclose(p_raw, p4[m].to_numpy(), atol=1e-8), f"{m}: refit does not reproduce Phase 4 validation predictions"
    sg, io = fit_sigmoid(OOF[m], y_tr), fit_isotonic(OOF[m], y_tr)               # calibrators from TRAIN OOF only
    VAL[m] = {"original": p_raw, "sigmoid": sg(p_raw), "isotonic": io(p_raw)}
    for meth, p in VAL[m].items():
        vrows.append({"model_id": m, "model": LABEL[m], "method": meth, "dataset": "validation_confirmation", "n": len(y_va), **cal_metrics(p, y_va)})
        b = quantile_bins(p, y_va); b.insert(0, "model_id", m); b.insert(1, "method", meth); b.insert(2, "dataset", "validation"); vbins.append(b)
print("Refit models reproduce the Phase 4 validation probabilities exactly.")
rng2 = np.random.default_rng(SEED + 7); IDXV = rng2.integers(0, len(y_va), (B_BOOT, len(y_va)))
vc = pd.DataFrame(vrows)
for i, r in vc.iterrows():
    if r.method == "original": continue
    for met, vec in [("log_loss", lambda p: ll_vec(p, y_va)), ("brier", lambda p: (p - y_va) ** 2)]:
        d = (vec(VAL[r.model_id][r.method]) - vec(VAL[r.model_id]["original"]))[IDXV].mean(axis=1)
        vc.loc[i, f"d_{met}"] = d.mean(); vc.loc[i, f"d_{met}_ci_lo"], vc.loc[i, f"d_{met}_ci_hi"] = np.percentile(d, [2.5, 97.5])
vc.to_csv(MET / "06_validation_confirmation.csv", index=False)
pd.concat(vbins, ignore_index=True).to_csv(MET / "06_validation_calibration_bins.csv", index=False)
pd.DataFrame({"y_true": y_va, **{f"{m}_{meth}": VAL[m][meth] for m in MODEL_IDS for meth in VAL[m]}}).to_csv(MET / "06_validation_confirmation_predictions.csv", index=False)
print("VALIDATION confirmation (train-fitted calibrators):")
vc[["model_id", "method", "roc_auc", "log_loss", "d_log_loss", "d_log_loss_ci_lo", "d_log_loss_ci_hi", "brier", "d_brier", "citl", "cal_intercept", "cal_slope", "slope_ci_lo", "slope_ci_hi", "ece_quantile", "ece_q_null_mean", "ece_q_null_p95", "ece_uniform"]].set_index(["model_id", "method"]).round(4)
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
for ax, m in zip(axes, MODEL_IDS):
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2)
    for meth in ["original", "sigmoid", "isotonic"]:
        b = quantile_bins(VAL[m][meth], y_va); ax.plot(b.mean_pred, b.obs_rate, "o-", color=METH_C[meth], lw=1.7, ms=4.5, label=meth)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(LABEL[m]); ax.set_xlabel("Mean predicted probability (10 equal-count bins, ~149 each)"); ax.set_ylabel("Observed placement rate"); ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.suptitle("Validation confirmation: reliability with calibrators fitted on TRAIN only (noisier: only ~149 students per bin)", y=1.02, fontsize=13, fontweight="semibold")
savefig(fig, "phase5_validation_reliability")
pd.DataFrame({"train_row_id": train_row_id, "actual_target": y_tr, **{f"{m}_{meth}": CAL[m][meth] for m in MODEL_IDS for meth in ("sigmoid", "isotonic")}, "fold": fold_of}).to_csv(MET / "06_oof_recalibrated_predictions.csv", index=False)
""")

md(r"""
## 9. Part 10: How to read a predicted probability
A predicted probability of 0.80 should **not** be read as "this particular student has an 80% chance of being placed". A reasonable calibrated model supports the weaker, population-level statement: **among students who receive predicted probabilities around 0.80 (and are similar to the students in this dataset), roughly 80% are expected to be placed.**
That is exactly what a reliability diagram checks. The probability summarises a *group* of students with similar profiles; any individual either is or is not placed, and the model uses only ten recorded variables and observational data from one (possibly rule-generated) dataset, so these probabilities describe association, not what would happen if a student changed any characteristic. Calibration quality also does not transfer automatically to other campuses or years.
""")

md("## 10. Phase 5 conclusions")
findings = Path("src/phase5_findings.md").read_text(encoding="utf-8") if Path("src/phase5_findings.md").exists() else "_Findings pending._"
md(findings)

md("## 11. Test-set audit")
code(r"""
data_ext = (".csv", ".parquet", ".json", ".joblib", ".pkl", ".feather", ".xlsx")
opened = sorted({os.path.normpath(p) for p in _OPENED if p.lower().endswith(data_ext)})
print("Data files opened during this notebook run:")
for p in opened: print("  ", p)
names = [os.path.basename(p).lower() for p in opened]
assert "test.csv" not in names, "test.csv (processed or raw) was opened!"
assert not any(("raw" in Path(p).parts and "data" in Path(p).parts) for p in opened), "a raw data file was opened"
assert "modeling_dataset.csv" not in names and "campus_recruitment_combined_flagged.csv" not in names
assert {"train.csv", "validation.csv"} <= set(names)
assert "X_test" not in globals() and "y_test" not in globals() and "test" not in globals()
print("\nASSERTIONS PASSED: test.csv (raw or processed) was never opened. CV, OOF predictions and calibration used TRAIN only; validation was used once for confirmation.")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/06_calibration_and_cross_validation.ipynb")
print("wrote notebooks/06_calibration_and_cross_validation.ipynb")
