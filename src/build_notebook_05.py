"""Builds notebooks/05_preprocessing_and_baselines.ipynb (Phase 4).
The end-of-phase summary prose is read from src/phase4_findings.md so it can be written against the executed numbers."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 05 - Preprocessing and Baseline Modelling (Phase 4)
**Question:** how accurately can we estimate the probability that an unseen student will be placed?
This phase does both **classification** and **probability estimation**, and judges models primarily on probability quality
(ROC-AUC, log loss, Brier score) rather than accuracy alone.

## Data policy (enforced in code)
| Split | File | Use in this notebook |
|---|---|---|
| TRAIN | `data/processed/train.csv` | fit every learned preprocessing step and every model |
| VALIDATION | `data/processed/validation.csv` | compare variants, choose hyper-parameters, provisional model selection |
| TEST | `data/processed/test.csv` | **never opened.** An audit hook records every file this notebook opens and a final assertion proves `test.csv` (processed or raw) was not read. Reserved for Phase 6. |

The raw supplied splits and the full `modeling_dataset.csv` are also not used. The classification threshold stays at the default 0.5 (no threshold tuning), and calibration is inspected in Phase 5.

*Disclosure:* the Phase 2-3 EDA/statistics used all modelling rows (including those now in the test split) to describe associations. They informed the choice to compare
numeric vs categorical coding of the count variables, but no model was fitted or scored on test rows, and this notebook's decisions rest on train/validation only.

## Method summary
- **Two preprocessing variants**, each in a scikit-learn `Pipeline` + `ColumnTransformer`, so scaling/encoding parameters are learned from the training data only inside the pipeline.
- **Models:** regularised logistic regression (plus a `class_weight="balanced"` comparison), decision tree, random forest, gradient boosting; each with a small, explicit validation-based search.
- **Within-family selection rule:** lowest validation log loss (the proper scoring rule closest to the project goal); ties broken by ROC-AUC. The complete search grids are saved so the extent of selection on validation is transparent.
- **Uncertainty:** the validation set has only about 1,487 students, so 95% bootstrap CIs (1,000 resamples) are attached to metrics, and *paired* bootstrap is used to compare models fairly.
""")

code(r"""
import warnings
warnings.simplefilter("always")
import sys, json, itertools, time
from pathlib import Path

# ---- audit hook: records every file opened by this process so we can PROVE the test set was never read ----
_OPENED = []
def _audit(event, args):
    if event == "open":
        try: _OPENED.append(str(args[0]))
        except Exception: pass
sys.addaudithook(_audit)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
                             brier_score_loss, confusion_matrix, roc_curve)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "phase4"
MET = ROOT / "results" / "metrics" / "modeling"
FIG = ROOT / "results" / "figures" / "modeling"
for d in (MODELS, MET, FIG): d.mkdir(parents=True, exist_ok=True)

SEED = 42
B_BOOT = 1000
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 80)
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12,
                                      "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
PLACED_C, NOT_C, NEUTRAL = "#2166ac", "#e08214", "#6b7785"
FAM_C = {"LR": "#0072B2", "DT": "#E69F00", "RF": "#009E73", "GB": "#D55E00"}      # Okabe-Ito, colour-blind safe
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); plt.show()
def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh: json.dump(obj, fh, indent=2, default=str)
print("scikit-learn", sklearn.__version__, "| seed", SEED)
""")

md("## 1. Load TRAIN and VALIDATION only; features and target")
code(r"""
TARGET = "PlacementStatus"
CONT = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
BIN = ["ExtracurricularActivities", "PlacementTraining"]
FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
assert sorted(FEATURES) == sorted(CONT + COUNT + BIN) and len(FEATURES) == 10

train = pd.read_csv(PROC / "train.csv"); val = pd.read_csv(PROC / "validation.csv")
for name, d in [("train", train), ("validation", val)]:
    assert list(d.columns) == FEATURES + [TARGET] or sorted(d.columns) == sorted(FEATURES + [TARGET]), name
    assert d.isna().sum().sum() == 0 and set(d[TARGET]) == {"Placed", "NotPlaced"}, name
# leakage guard between the two splits we use
tp = set(map(tuple, train[FEATURES].values)); vp = set(map(tuple, val[FEATURES].values))
assert not (tp & vp), "feature-profile overlap between train and validation"

X_train, X_val = train[FEATURES].copy(), val[FEATURES].copy()
y_train = (train[TARGET] == "Placed").astype(int).to_numpy()     # Placed = 1, NotPlaced = 0
y_val = (val[TARGET] == "Placed").astype(int).to_numpy()
for c in COUNT: X_train[c] = X_train[c].astype(int); X_val[c] = X_val[c].astype(int)
print(f"train {X_train.shape}, Placed rate {y_train.mean():.4f}   |   validation {X_val.shape}, Placed rate {y_val.mean():.4f}")
print("Predictors:", FEATURES); print("No ID/index/source/split columns are used; target encoded Placed=1, NotPlaced=0.")
""")

md(r"""
## 2. Preprocessing variants
**Variant A - numeric counts.** Internships, Projects and Workshops/Certifications stay numeric; all eight numeric columns are standardised (for logistic regression); the two binary predictors are one-hot encoded.

**Variant B - categorical counts.** The three count variables are one-hot encoded together with the two binary predictors (5 categorical variables); only the five scale-type variables (CGPA, aptitude, soft skills, SSC, HSC) are standardised.

**Reference categories (logistic regression).** One level per categorical variable is dropped so coefficients are contrasts against that reference. Binary variables use `No`. For the count variables the reference is the **most frequent level in TRAIN**, so every coefficient is compared with a large group
(e.g. only a handful of training students have Projects = 0, which would make a poor reference). **Tree models are not standardised** (splits are scale-invariant) and keep all one-hot levels.
No imputation is used: there are no missing values.
""")
code(r"""
REFS = {c: int(X_train[c].mode().iloc[0]) for c in COUNT}          # learned from TRAIN only
print("Reference levels for count variables (train mode):", REFS, "| binary reference: 'No'")
print("Train level counts:", {c: X_train[c].value_counts().sort_index().to_dict() for c in COUNT})

def make_preprocessor(variant, scale, drop_ref):
    # variant 'A': counts numeric ; 'B': counts one-hot.  scale: StandardScaler on numeric columns ; drop_ref: drop reference level (LR)
    bin_enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error")        # reference = 'No'
    scaler = StandardScaler() if scale else "passthrough"
    if variant == "A":
        parts = [("num", scaler, CONT + COUNT), ("bin", bin_enc, BIN)]
    else:
        cnt_enc = OneHotEncoder(drop=[REFS[c] for c in COUNT] if drop_ref else None, sparse_output=False, handle_unknown="error")
        parts = [("num", scaler, CONT), ("cnt", cnt_enc, COUNT), ("bin", bin_enc, BIN)]
    return ColumnTransformer(parts, verbose_feature_names_out=False).set_output(transform="pandas")

def make_pipeline(kind, variant, **params):
    if kind == "LR":
        pre = make_preprocessor(variant, scale=True, drop_ref=True)
        model = LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=5000, random_state=SEED, **params)   # l1_ratio=0 == L2 (sklearn>=1.8 spelling of penalty='l2')
    else:
        pre = make_preprocessor(variant, scale=False, drop_ref=False)
        cls = {"DT": DecisionTreeClassifier, "RF": RandomForestClassifier, "GB": GradientBoostingClassifier}[kind]
        extra = {"n_jobs": -1} if kind == "RF" else {}
        model = cls(random_state=SEED, **extra, **params)
    return Pipeline([("prep", pre), ("model", model)])

# demonstration: learned parameters come from TRAIN only
demo = make_preprocessor("A", True, True).fit(X_train)
sc = demo.named_transformers_["num"]
assert np.allclose(sc.mean_, X_train[CONT + COUNT].mean().to_numpy())
assert not np.allclose(sc.mean_, X_val[CONT + COUNT].mean().to_numpy()), "scaler appears to have seen validation data"
print("\nScaler means equal TRAIN means (not validation means) -> preprocessing is fitted on train only.")
for v in ["A", "B"]:
    print(f"\nVariant {v} output columns ({'LR' if True else ''}, reference dropped):", list(make_preprocessor(v, True, True).fit(X_train).get_feature_names_out()))
""")

md("## 3. Metric helpers and reference baseline")
code(r"""
def evaluate(p, y, thr=0.5):
    pred = (p >= thr).astype(int)           # default threshold, deliberately NOT tuned
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p)}

def fit_score(pipe):
    pipe.fit(X_train, y_train)
    return pipe, pipe.predict_proba(X_train)[:, 1], pipe.predict_proba(X_val)[:, 1]

dummy, p_tr_d, p_va_d = fit_score(Pipeline([("model", DummyClassifier(strategy="prior"))]))
# (Dummy gets raw frame but ignores it)
base_val = evaluate(p_va_d, y_val)
print("Reference: always predict the TRAIN placement rate (%.3f) -> validation" % y_train.mean())
pd.Series(base_val).round(4)
""")

md(r"""
## 4. Baseline 1 - regularised logistic regression
L2-regularised (`C = 1.0`, `lbfgs`, `max_iter = 5000`), fitted on both preprocessing variants. A `class_weight="balanced"` version of each is trained **only as a comparison**: re-weighting classes deliberately shifts predicted probabilities away from the true base rate, so it is not eligible for the final choice (the project goal is probability estimation).
""")
code(r"""
REG = {}          # model_id -> dict(pipe, p_train, p_val, family, variant, config, label)
def register(mid, family, variant, label, config, pipe, ptr, pva):
    REG[mid] = dict(family=family, variant=variant, label=label, config=config, pipe=pipe, p_train=ptr, p_val=pva)

for v in ["A", "B"]:
    for cw, tag in [(None, ""), ("balanced", "_balanced")]:
        pipe, ptr, pva = fit_score(make_pipeline("LR", v, C=1.0, class_weight=cw))
        assert pipe.named_steps["model"].n_iter_[0] < 5000, "LR did not converge"
        register(f"LR_{v}{tag}", "LR", v, "Logistic regression" + (" (balanced)" if cw else ""), {"C": 1.0, "class_weight": cw}, pipe, ptr, pva)
print("Logistic regression fitted (converged) for variants A/B, plain and balanced.")
pd.DataFrame({k: evaluate(r["p_val"], y_val) for k, r in REG.items() if r["family"] == "LR"}).T.round(4)
""")
code(r"""
# Sensitivity of validation log loss to the regularisation strength (reported only; C = 1.0 is kept as the baseline)
rows = []
for v in ["A", "B"]:
    for C_ in [0.01, 0.1, 1.0, 10.0, 100.0]:
        p = make_pipeline("LR", v, C=C_).fit(X_train, y_train).predict_proba(X_val)[:, 1]
        rows.append({"variant": v, "C": C_, **{k: evaluate(p, y_val)[k] for k in ["roc_auc", "log_loss", "brier"]}})
c_sens = pd.DataFrame(rows); c_sens.to_csv(MET / "05_lr_C_sensitivity.csv", index=False)
c_sens.pivot(index="C", columns="variant", values="log_loss").round(4)
""")

md(r"""
## 5. Baseline 2 - decision tree
Controlled search: `max_depth` in {3, 5, 7, None} x `min_samples_leaf` in {1, 5, 10} (12 configurations per variant), Gini criterion. Selection per variant: lowest validation log loss.
""")
code(r"""
GRIDS = []                  # every configuration tried, for transparency
def run_grid(kind, variant, grid, label):
    keys = list(grid); best = None
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals)); t0 = time.time()
        pipe, ptr, pva = fit_score(make_pipeline(kind, variant, **cfg))
        m = evaluate(pva, y_val); mt = evaluate(ptr, y_train)
        GRIDS.append({"family": kind, "variant": variant, "config": json.dumps(cfg, default=str), **{f"val_{k}": v for k, v in m.items()},
                      "train_roc_auc": mt["roc_auc"], "train_log_loss": mt["log_loss"], "fit_seconds": round(time.time() - t0, 2)})
        key = (m["log_loss"], -m["roc_auc"])
        if best is None or key < best[0]: best = (key, cfg, pipe, ptr, pva)
    _, cfg, pipe, ptr, pva = best
    register(f"{kind}_{variant}", kind, variant, label, cfg, pipe, ptr, pva)
    return cfg

dt_grid = {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}
for v in ["A", "B"]: print(f"DT variant {v} selected:", run_grid("DT", v, dt_grid, "Decision tree"))
g = pd.DataFrame(GRIDS); g[g.family == "DT"].assign(config=lambda d: d.config).pivot_table(index="config", columns="variant", values="val_log_loss").round(4)
""")

md(r"""
## 6. Model 3 - random forest
`n_estimators = 300`; searched: `max_depth` in {5, 10, None}, `min_samples_leaf` in {1, 5, 10}, `max_features` in {"sqrt", 0.5} (18 configurations per variant).
""")
code(r"""
rf_grid = {"n_estimators": [300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5]}
for v in ["A", "B"]: print(f"RF variant {v} selected:", run_grid("RF", v, rf_grid, "Random forest"))
g = pd.DataFrame(GRIDS); g[g.family == "RF"].pivot_table(index="config", columns="variant", values="val_log_loss").round(4).sort_values("A").head(8)
""")

md(r"""
## 7. Model 4 - gradient boosting
Searched: `n_estimators` in {100, 300}, `learning_rate` in {0.05, 0.1}, `max_depth` in {2, 3}, `min_samples_leaf` in {1, 20} (16 configurations per variant).
""")
code(r"""
gb_grid = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3], "min_samples_leaf": [1, 20]}
for v in ["A", "B"]: print(f"GB variant {v} selected:", run_grid("GB", v, gb_grid, "Gradient boosting"))
grids = pd.DataFrame(GRIDS); grids.to_csv(MET / "05_tuning_grids.csv", index=False)
print("\nConfigurations evaluated:", grids.groupby(["family", "variant"]).size().to_dict())
grids[grids.family == "GB"].pivot_table(index="config", columns="variant", values="val_log_loss").round(4).sort_values("A").head(8)
""")

md(r"""
## 8. Validation model comparison
All selected configurations, default threshold 0.5. 95% CIs come from 1,000 bootstrap resamples of the validation students (the same resamples for every model, which also allows *paired* comparisons).
""")
code(r"""
order = ["LR_A", "LR_B", "LR_A_balanced", "LR_B_balanced", "DT_A", "DT_B", "RF_A", "RF_B", "GB_A", "GB_B"]
assert set(order) == set(REG)
EPS = 1e-15
def ll_vec(p): q = np.clip(p, EPS, 1 - EPS); return -(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))
rng = np.random.default_rng(SEED); n_val = len(y_val); IDX = rng.integers(0, n_val, (B_BOOT, n_val))     # shared resamples (paired)
BOOT = {}
for mid in order + ["DUMMY"]:
    p = p_va_d if mid == "DUMMY" else REG[mid]["p_val"]
    BOOT[mid] = {"auc": np.array([roc_auc_score(y_val[i], p[i]) for i in IDX]),
                 "ll": ll_vec(p)[IDX].mean(axis=1), "brier": ((p - y_val) ** 2)[IDX].mean(axis=1)} if mid != "DUMMY" else \
                {"auc": np.full(B_BOOT, 0.5), "ll": ll_vec(p)[IDX].mean(axis=1), "brier": ((p - y_val) ** 2)[IDX].mean(axis=1)}
ci = lambda a: np.percentile(a, [2.5, 97.5])

def cfg_text(r):
    c = {k: v for k, v in r["config"].items() if k not in ("n_estimators",) or r["family"] == "GB"}
    return ", ".join(f"{k}={v}" for k, v in r["config"].items())
rows = []
for mid in order:
    r = REG[mid]; m = evaluate(r["p_val"], y_val)
    rows.append({"model_id": mid, "Model": r["label"], "Feature representation": "A: numeric counts" if r["variant"] == "A" else "B: categorical counts",
                 "Config": cfg_text(r), **{k: m[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier"]},
                 "auc_ci_lo": ci(BOOT[mid]["auc"])[0], "auc_ci_hi": ci(BOOT[mid]["auc"])[1], "ll_ci_lo": ci(BOOT[mid]["ll"])[0], "ll_ci_hi": ci(BOOT[mid]["ll"])[1],
                 "brier_ci_lo": ci(BOOT[mid]["brier"])[0], "brier_ci_hi": ci(BOOT[mid]["brier"])[1],
                 "mean_pred": r["p_val"].mean(), "observed_rate": y_val.mean()})
rows.append({"model_id": "DUMMY", "Model": "Reference: train placement rate", "Feature representation": "-", "Config": "strategy=prior",
             **{k: base_val[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier"]},
             "ll_ci_lo": ci(BOOT["DUMMY"]["ll"])[0], "ll_ci_hi": ci(BOOT["DUMMY"]["ll"])[1], "brier_ci_lo": ci(BOOT["DUMMY"]["brier"])[0],
             "brier_ci_hi": ci(BOOT["DUMMY"]["brier"])[1], "mean_pred": p_va_d.mean(), "observed_rate": y_val.mean()})
results = pd.DataFrame(rows)
results["brier_skill_vs_base_rate"] = 1 - results.brier / base_val["brier"]
results.to_csv(MET / "05_validation_results.csv", index=False)
cols = ["Model", "Feature representation", "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier"]
results[cols].round(4)
""")
code(r"""
print("Selected hyper-parameters per model:")
for mid in order: print(f"  {mid:15s} {cfg_text(REG[mid])}")
print("\n95% bootstrap CIs (validation):")
results.set_index("model_id")[["roc_auc", "auc_ci_lo", "auc_ci_hi", "log_loss", "ll_ci_lo", "ll_ci_hi", "brier", "brier_ci_lo", "brier_ci_hi"]].round(4)
""")
code(r"""
# Model comparison plot (point estimate +/- 95% bootstrap CI); hollow markers = class-balanced comparison models
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
rr = results[results.model_id != "DUMMY"].reset_index(drop=True); yy = np.arange(len(rr))[::-1]
labels = [f"{r['Model']} | {r['model_id'].split('_')[1]}" for _, r in rr.iterrows()]
for ax, (m, lo, hi, title, ref) in zip(axes, [("roc_auc", "auc_ci_lo", "auc_ci_hi", "ROC-AUC (higher is better)", 0.5),
                                                ("log_loss", "ll_ci_lo", "ll_ci_hi", "Log loss (lower is better)", base_val["log_loss"]),
                                                ("brier", "brier_ci_lo", "brier_ci_hi", "Brier score (lower is better)", base_val["brier"])]):
    for yi, (_, r) in zip(yy, rr.iterrows()):
        fam = r.model_id.split("_")[0]; hollow = r.model_id.endswith("balanced")
        ax.errorbar(r[m], yi, xerr=[[r[m] - r[lo]], [r[hi] - r[m]]], fmt="o", color=FAM_C[fam], ecolor=FAM_C[fam], mfc="white" if hollow else FAM_C[fam], capsize=3, ms=7)
    ax.axvline(ref, color=NEUTRAL, ls="--", lw=1); ax.set_title(title); ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
axes[1].text(base_val["log_loss"], yy.max() + 0.55, "base rate", fontsize=8, color=NEUTRAL, ha="center")
fig.suptitle("Validation comparison of selected configurations (A = numeric counts, B = categorical counts; hollow = class-balanced)", y=1.02, fontsize=13, fontweight="semibold")
savefig(fig, "model_01_comparison")
""")

md("## 9. ROC curves, confusion matrices and probability distributions (validation)")
code(r"""
core = [m for m in order if not m.endswith("balanced")]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
for ax, v in zip(axes, ["A", "B"]):
    for fam in ["LR", "DT", "RF", "GB"]:
        r = REG[f"{fam}_{v}"]; fpr, tpr, _ = roc_curve(y_val, r["p_val"]); auc = roc_auc_score(y_val, r["p_val"])
        ax.plot(fpr, tpr, color=FAM_C[fam], lw=2, label=f"{r['label']} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Variant A: numeric counts" if v == "A" else "Variant B: categorical counts"); ax.legend(frameon=False, loc="lower right", fontsize=9)
savefig(fig, "model_02_validation_roc")

fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4)); cm_rows = []
for ax, mid in zip(axes.ravel(), [f"{f}_{v}" for v in ["A", "B"] for f in ["LR", "DT", "RF", "GB"]]):
    pred = (REG[mid]["p_val"] >= 0.5).astype(int); cm = confusion_matrix(y_val, pred, labels=[0, 1])
    for i, a in enumerate(["NotPlaced", "Placed"]):
        for j, b in enumerate(["NotPlaced", "Placed"]): cm_rows.append({"model_id": mid, "actual": a, "predicted": b, "count": int(cm[i, j])})
    ann = np.array([[f"{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.0f}% of row)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=ann, fmt="", cmap="Blues", cbar=False, ax=ax, xticklabels=["NotPlaced", "Placed"], yticklabels=["NotPlaced", "Placed"], linewidths=1, linecolor="white", annot_kws={"fontsize": 9})
    ax.set_title(f"{REG[mid]['label']} | {mid[-1]}", fontsize=10.5); ax.set_xlabel("Predicted (threshold 0.5)"); ax.set_ylabel("Actual")
pd.DataFrame(cm_rows).to_csv(MET / "05_confusion_matrices.csv", index=False)
savefig(fig, "model_03_confusion_matrices")
""")
code(r"""
BINS = np.linspace(0, 1, 21); prob_rows = []
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True)
for ax, mid in zip(axes.ravel(), [f"{f}_{v}" for v in ["A", "B"] for f in ["LR", "DT", "RF", "GB"]]):
    p = REG[mid]["p_val"]
    hp, _ = np.histogram(p[y_val == 1], BINS); hn, _ = np.histogram(p[y_val == 0], BINS)
    wp, wn = hp / hp.sum(), hn / hn.sum()
    ax.bar(BINS[:-1], wn * 100, width=0.05, align="edge", color=NOT_C, alpha=0.65, label="Actual NotPlaced")
    ax.bar(BINS[:-1], wp * 100, width=0.05, align="edge", color=PLACED_C, alpha=0.65, label="Actual Placed")
    ax.set_title(f"{REG[mid]['label']} | {mid[-1]}", fontsize=10.5); ax.set_xlim(0, 1)
    prob_rows.append({"model_id": mid, "overlap_coefficient": float(np.minimum(wp, wn).sum()), "median_p_placed": float(np.median(p[y_val == 1])),
                      "median_p_notplaced": float(np.median(p[y_val == 0])), "share_uncertain_0.2_0.8_%": float(((p > 0.2) & (p < 0.8)).mean() * 100),
                      "share_extreme_<0.02_or_>0.98_%": float(((p < 0.02) | (p > 0.98)).mean() * 100), "mean_pred": float(p.mean()), "observed_rate": float(y_val.mean())})
for ax in axes[:, 0]: ax.set_ylabel("% of that class")
for ax in axes[1]: ax.set_xlabel("Predicted P(Placed)")
axes[0, 0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Predicted-probability distributions by actual outcome (validation); overlap shows where the model cannot tell the groups apart", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "model_04_probability_distributions")
prob = pd.DataFrame(prob_rows).set_index("model_id"); prob.to_csv(MET / "05_probability_diagnostics.csv"); prob.round(3)
""")
md(r"""
**Reading the probability plots.** Less overlap between the two histograms means the model ranks Placed students above NotPlaced students more reliably, but even well-separated distributions do **not** mean perfect prediction: the overlap coefficient above quantifies the shared region, the "uncertain" share lists students given probabilities between 0.2 and 0.8, and extreme probabilities from flexible models can reflect over-confidence rather than accuracy (calibration is examined in Phase 5).
""")

md(r"""
## 10. Logistic-regression coefficients
Coefficients are log-odds changes. **Numeric predictors are standardised, so one unit = one training-set standard deviation** (`train_sd` is given so effects per original unit can be derived: `OR_per_original_unit = exp(coef / train_sd)`).
One-hot coefficients are contrasts with the **reference** level named in the table. Wording: "holding the other model variables constant, the fitted model associates ... with ..."; these are associations in observational data, **not causal effects**. The L2 penalty shrinks coefficients slightly, and no standard errors/CIs are reported for the penalised fit.
""")
code(r"""
coef_rows = []
for mid in ["LR_A", "LR_B", "LR_A_balanced", "LR_B_balanced"]:
    pipe = REG[mid]["pipe"]; prep, lr = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out()); sc = prep.named_transformers_["num"]; sd = dict(zip(sc.feature_names_in_, sc.scale_))
    for nme, b in zip(names, lr.coef_[0]):
        if nme in sd: unit, ref, tsd = "per 1 SD (train)", "-", sd[nme]
        else:
            base = next(f for f in FEATURES if nme.startswith(f + "_")); lvl = nme[len(base) + 1:]
            ref = "No" if base in BIN else str(REFS[base]); unit, tsd = f"{base}={lvl} vs {ref}", np.nan
        coef_rows.append({"model_id": mid, "feature": nme, "coefficient": b, "odds_ratio": np.exp(b), "unit": unit, "reference_level": ref, "train_sd": tsd,
                          "odds_ratio_per_original_unit": np.exp(b / tsd) if not np.isnan(tsd) else np.nan})
    coef_rows.append({"model_id": mid, "feature": "(intercept)", "coefficient": lr.intercept_[0], "odds_ratio": np.exp(lr.intercept_[0]), "unit": "baseline log-odds at reference/mean profile",
                      "reference_level": "-", "train_sd": np.nan, "odds_ratio_per_original_unit": np.nan})
coefs = pd.DataFrame(coef_rows); coefs.to_csv(MET / "05_logistic_coefficients.csv", index=False)
for mid in ["LR_A", "LR_B"]:
    print(f"=== {mid} ({REG[mid]['label']}; {'numeric' if mid.endswith('A') else 'categorical'} counts) ===")
    display(coefs[coefs.model_id == mid].drop(columns="model_id").set_index("feature").round(3))
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
for ax, mid, ttl in zip(axes, ["LR_A", "LR_B"], ["Variant A (numeric counts)", "Variant B (categorical counts)"]):
    d = coefs[(coefs.model_id == mid) & (coefs.feature != "(intercept)")].sort_values("coefficient")
    cols_ = [PLACED_C if b > 0 else NOT_C for b in d.coefficient]
    ax.barh(d.feature, d.coefficient, color=cols_, height=0.65)
    for yi, (b, o) in enumerate(zip(d.coefficient, d.odds_ratio)): ax.text(b + (0.03 if b >= 0 else -0.03), yi, f"OR {o:.2f}", va="center", ha="left" if b >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="#1d2733", lw=1); ax.set_xlabel("Coefficient (log-odds); numeric = per 1 SD, one-hot = vs reference level"); ax.set_title(ttl); ax.grid(axis="y", visible=False)
    lo_, hi_ = ax.get_xlim(); ax.set_xlim(lo_ - 0.5, hi_ + 0.5)
fig.suptitle("Logistic regression (L2, C=1): associations holding other model variables constant. Not causal effects.", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "model_05_logistic_coefficients")
""")

md(r"""
## 11. Tree-model feature importance
Impurity-based (mean decrease in impurity) importances from the selected decision tree, random forest and gradient boosting models, shown per encoded column and **aggregated back to the original predictor** (sum over one-hot levels).
These describe how heavily a fitted model *uses* a feature; they are **not causal**, are biased towards features with many distinct values, and are shared among correlated predictors. Permutation importance or SHAP can be used later for stronger interpretation.
""")
code(r"""
def original_of(name): return next(f for f in FEATURES if name == f or name.startswith(f + "_"))
imp_rows = []
for mid in [f"{f}_{v}" for f in ["DT", "RF", "GB"] for v in ["A", "B"]]:
    pipe = REG[mid]["pipe"]; names = pipe.named_steps["prep"].get_feature_names_out(); imp = pipe.named_steps["model"].feature_importances_
    for n_, i_ in zip(names, imp): imp_rows.append({"model_id": mid, "level": "encoded", "feature": n_, "original_feature": original_of(n_), "importance": i_})
    for f_, s_ in pd.Series(imp, index=[original_of(n_) for n_ in names]).groupby(level=0).sum().items():
        imp_rows.append({"model_id": mid, "level": "original", "feature": f_, "original_feature": f_, "importance": s_})
imp_df = pd.DataFrame(imp_rows); imp_df.to_csv(MET / "05_tree_feature_importance.csv", index=False)
agg = imp_df[imp_df.level == "original"].pivot(index="feature", columns="model_id", values="importance")[[f"{f}_{v}" for v in ["A", "B"] for f in ["DT", "RF", "GB"]]]
display(agg.round(3))
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
ordr = agg.mean(axis=1).sort_values().index
for ax, mid in zip(axes.ravel(), agg.columns):
    ax.barh(ordr, agg.loc[ordr, mid], color=FAM_C[mid.split("_")[0]], height=0.65)
    for yi, v in enumerate(agg.loc[ordr, mid]): ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_title(f"{REG[mid]['label']} | variant {mid[-1]}"); ax.grid(axis="y", visible=False)
for ax in axes[1]: ax.set_xlabel("Impurity-based importance (sums to 1)")
fig.suptitle("Tree-model feature importance, aggregated to the original predictors (not causal)", y=1.01, fontsize=13, fontweight="semibold")
savefig(fig, "model_06_tree_feature_importance")
""")

md(r"""
## 12. Overfitting check (train vs validation)
Heuristic flags (not statistical tests): **suspicious** if the ROC-AUC gap (train - validation) exceeds 0.03 or the log-loss gap (validation - train) exceeds 0.05; **large** if the AUC gap exceeds 0.05 or the log-loss gap exceeds 0.10. A high training score never drives selection.
""")
code(r"""
og = []
for mid in order:
    r = REG[mid]; mt, mv = evaluate(r["p_train"], y_train), evaluate(r["p_val"], y_val)
    ag, lg = mt["roc_auc"] - mv["roc_auc"], mv["log_loss"] - mt["log_loss"]
    flag = "large" if (ag > 0.05 or lg > 0.10) else "suspicious" if (ag > 0.03 or lg > 0.05) else "ok"
    og.append({"model_id": mid, "Model": r["label"], "Representation": r["variant"], "train_roc_auc": mt["roc_auc"], "val_roc_auc": mv["roc_auc"], "auc_gap": ag,
               "train_log_loss": mt["log_loss"], "val_log_loss": mv["log_loss"], "log_loss_gap": lg, "train_brier": mt["brier"], "val_brier": mv["brier"], "flag": flag})
gap = pd.DataFrame(og); gap.to_csv(MET / "05_train_validation_comparison.csv", index=False)
gap.drop(columns=["Model"]).set_index("model_id").round(4)
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); x = np.arange(len(gap)); w = 0.38
for ax, (a, b, ttl) in zip(axes, [("train_roc_auc", "val_roc_auc", "ROC-AUC"), ("train_log_loss", "val_log_loss", "Log loss")]):
    ax.bar(x - w / 2, gap[a], w, color=NEUTRAL, label="Train"); ax.bar(x + w / 2, gap[b], w, color=PLACED_C, label="Validation")
    ax.set_xticks(x); ax.set_xticklabels(gap.model_id, rotation=45, ha="right", fontsize=9); ax.set_title(f"{ttl}: train vs validation"); ax.legend(frameon=False)
    if ttl == "ROC-AUC": ax.set_ylim(0.5, 1.02)
savefig(fig, "model_07_train_vs_validation")
""")

md(r"""
## 13. Numeric (A) vs categorical (B) count representation
For each model family at its selected configuration: B minus A for ROC-AUC and log loss, with **paired** bootstrap 95% CIs (same validation resamples). A CI that includes 0 means the data do not distinguish the two representations.
""")
code(r"""
comp = []
for fam in ["LR", "LR_balanced", "DT", "RF", "GB"]:
    a, b = (f"LR_A_balanced", f"LR_B_balanced") if fam == "LR_balanced" else (f"{fam}_A", f"{fam}_B")
    dll, dauc, dbr = BOOT[b]["ll"] - BOOT[a]["ll"], BOOT[b]["auc"] - BOOT[a]["auc"], BOOT[b]["brier"] - BOOT[a]["brier"]
    ra, rb = results.set_index("model_id").loc[a], results.set_index("model_id").loc[b]
    comp.append({"family": fam, "AUC_A": ra.roc_auc, "AUC_B": rb.roc_auc, "dAUC_(B-A)": rb.roc_auc - ra.roc_auc, "dAUC_ci_lo": ci(dauc)[0], "dAUC_ci_hi": ci(dauc)[1],
                 "LL_A": ra.log_loss, "LL_B": rb.log_loss, "dLL_(B-A)": rb.log_loss - ra.log_loss, "dLL_ci_lo": ci(dll)[0], "dLL_ci_hi": ci(dll)[1],
                 "dBrier_(B-A)": rb.brier - ra.brier, "dBrier_ci_lo": ci(dbr)[0], "dBrier_ci_hi": ci(dbr)[1],
                 "B_distinguishable_from_A_on_logloss": not (ci(dll)[0] <= 0 <= ci(dll)[1])})
comp = pd.DataFrame(comp); comp.to_csv(MET / "05_variant_A_vs_B.csv", index=False); comp.set_index("family").round(4)
""")

md(r"""
## 14. Provisional model selection (VALIDATION ONLY)
**Rule, fixed before looking at the final ranking.**
1. **Eligibility:** exclude the class-balanced comparison models (they intentionally distort probabilities); exclude models with a **large** train-validation gap (section 12); exclude models whose mean predicted probability differs from the observed validation rate by more than 3 percentage points (probabilities must be sensible).
2. **Primary comparison:** among eligible models find the one with the lowest validation log loss (the proper scoring rule closest to the project's purpose). Every eligible model whose paired-bootstrap 95% CI for (log loss minus best log loss) includes 0 is *statistically indistinguishable* from the best on this validation set.
3. **Tie-break within the indistinguishable set:** prefer the more interpretable family (logistic regression, then decision tree, then gradient boosting / random forest), then lower log loss. ROC-AUC and Brier score are inspected for agreement; F1/recall at 0.5 serve as a sanity check, not a deciding criterion.
This selection is provisional: it uses a single validation split of 1,487 students, and the search was also run on this split, so validation scores of the chosen model are slightly optimistic. Phase 5 should confirm with cross-validation on the training data and examine calibration.
""")
code(r"""
rk = {"LR": 1, "DT": 2, "GB": 3, "RF": 3}
R = results.set_index("model_id").join(gap.set_index("model_id")[["auc_gap", "log_loss_gap", "flag"]])
elig = R.loc[[m for m in order if not m.endswith("balanced")]].copy()
elig["prob_sane"] = (elig.mean_pred - elig.observed_rate).abs() <= 0.03
elig["eligible"] = (elig.flag != "large") & elig.prob_sane
print("Eligibility:"); display(elig[["flag", "prob_sane", "eligible", "log_loss", "roc_auc", "brier"]].round(4))
E = elig[elig.eligible]; best = E.log_loss.idxmin()
sel_rows = []
for m in E.index:
    d = BOOT[m]["ll"] - BOOT[best]["ll"]; lo, hi = ci(d)
    sel_rows.append({"model_id": m, "log_loss": E.loc[m, "log_loss"], "dLL_vs_best": E.loc[m, "log_loss"] - E.loc[best, "log_loss"], "ci_lo": lo, "ci_hi": hi,
                     "indistinguishable_from_best": lo <= 0, "interpretability_rank": rk[m.split("_")[0]], "roc_auc": E.loc[m, "roc_auc"], "brier": E.loc[m, "brier"],
                     "f1": E.loc[m, "f1"], "recall": E.loc[m, "recall"]})
sel = pd.DataFrame(sel_rows).set_index("model_id")
cand = sel[sel.indistinguishable_from_best].sort_values(["interpretability_rank", "log_loss"])
chosen = cand.index[0]
print(f"\nBest validation log loss among eligible models: {best}")
display(sel.round(4))
print(f"\nStatistically indistinguishable from best: {list(cand.index)}")
print(f"PROVISIONAL SELECTION -> {chosen}: {REG[chosen]['label']}, {'numeric' if REG[chosen]['variant']=='A' else 'categorical'} counts, config {cfg_text(REG[chosen])}")
""")
code(r"""
r = results.set_index("model_id").loc[chosen]; g_ = gap.set_index("model_id").loc[chosen]
print(f"{chosen}: val ROC-AUC {r.roc_auc:.4f} [{r.auc_ci_lo:.4f}, {r.auc_ci_hi:.4f}] | log loss {r.log_loss:.4f} [{r.ll_ci_lo:.4f}, {r.ll_ci_hi:.4f}] | Brier {r.brier:.4f} (skill vs base rate {r.brier_skill_vs_base_rate:.3f})")
print(f"accuracy {r.accuracy:.4f}, precision {r.precision:.4f}, recall {r.recall:.4f}, F1 {r.f1:.4f}")
print(f"train-validation: AUC gap {g_.auc_gap:+.4f}, log-loss gap {g_.log_loss_gap:+.4f}, flag = {g_.flag}")
print(f"mean predicted {r.mean_pred:.3f} vs observed {r.observed_rate:.3f}")
write_json({"provisional_model_id": chosen, "label": REG[chosen]["label"], "variant": REG[chosen]["variant"], "config": REG[chosen]["config"],
           "selected_on": "validation only (train used for fitting; test never opened)", "rule": "lowest val log loss among eligible, statistically indistinguishable set, tie-break by interpretability",
           "indistinguishable_set": list(cand.index), "validation_log_loss": float(r.log_loss), "validation_roc_auc": float(r.roc_auc), "validation_brier": float(r.brier)},
          MET / "05_provisional_selection.json")
""")

md("## 15. Save pipelines, configurations and prediction tables")
code(r"""
for mid in order: joblib.dump(REG[mid]["pipe"], MODELS / f"{mid}.joblib")
joblib.dump(dummy, MODELS / "DUMMY_prior.joblib")
configs = {"seed": SEED, "sklearn_version": sklearn.__version__, "features": FEATURES, "target": TARGET, "target_encoding": {"Placed": 1, "NotPlaced": 0},
           "count_reference_levels_(LR)": REFS, "binary_reference_level": "No", "threshold": 0.5,
           "preprocessing": {"A": "StandardScaler on 5 scale vars + 3 counts; one-hot ExtracurricularActivities & PlacementTraining (drop No) [LR]; trees: unscaled",
                             "B": "StandardScaler on 5 scale vars; one-hot Internships/Projects/Workshops (drop train-mode level for LR; all levels for trees) + 2 binary [LR]"},
           "models": {mid: {"family": REG[mid]["family"], "variant": REG[mid]["variant"], "config": REG[mid]["config"]} for mid in order},
           "search_grids": {"LR_C_sensitivity": [0.01, 0.1, 1, 10, 100], "DT": dt_grid, "RF": rf_grid, "GB": gb_grid},
           "files": {mid: f"models/phase4/{mid}.joblib" for mid in order}}
write_json(configs, MODELS / "model_configs.json")
pd.DataFrame({"y_true": y_val, **{mid: REG[mid]["p_val"] for mid in order}}).to_csv(MET / "05_validation_predictions.csv", index=False)
# reload check: a saved pipeline reproduces its validation probabilities exactly
re_ = joblib.load(MODELS / f"{chosen}.joblib")
assert np.allclose(re_.predict_proba(X_val)[:, 1], REG[chosen]["p_val"]), "saved pipeline does not reproduce predictions"
print("Saved", len(order) + 1, "pipelines to", MODELS, "and reload check passed.")
""")

md("## 16. End-of-phase summary")
findings = Path("src/phase4_findings.md").read_text(encoding="utf-8") if Path("src/phase4_findings.md").exists() else "_Summary pending._"
md(findings)

md("## 17. Proof that the test set was not used")
code(r"""
opened = [Path(p) for p in _OPENED if isinstance(p, str) and p.lower().endswith((".csv", ".parquet", ".json", ".joblib", ".pkl"))]
csv_read = sorted({p.name for p in opened if p.suffix == ".csv"})
print("CSV files opened during this notebook run:", csv_read)
assert "test.csv" not in [p.name.lower() for p in opened], "test.csv was opened (processed or raw)!"
assert not any("data" in p.parts and "raw" in p.parts for p in opened), "a raw data file was opened"
assert "modeling_dataset.csv" not in csv_read and "campus_recruitment_combined_flagged.csv" not in csv_read
assert set(c for c in csv_read if c in ("train.csv", "validation.csv")) == {"train.csv", "validation.csv"}
assert "X_test" not in globals() and "y_test" not in globals() and "test" not in globals()
print("ASSERTIONS PASSED: test.csv was never opened; all fitting used TRAIN and all selection used VALIDATION only.")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/05_preprocessing_and_baselines.ipynb")
print("wrote notebooks/05_preprocessing_and_baselines.ipynb")
