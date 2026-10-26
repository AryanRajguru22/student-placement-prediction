"""Builds notebooks/08_final_test_evaluation.ipynb (Phase 7: the one-time FINAL TEST EVALUATION).

Usage (run from the project root):
    python src/build_notebook_08.py                    # write the notebook (findings cell = placeholder)
    python src/build_notebook_08.py --embed-findings   # AFTER the single real execution: patch ONLY the findings
                                                       # markdown cell of the executed notebook (no re-execution)
Execution modes of the notebook itself (environment variables):
    PHASE7_DRY_RUN=1 PHASE7_DRY_OUT=<dir>   smoke test: validation.csv stands in for test.csv, every output goes to <dir>,
                                            the real test.csv is not read.
    (neither set)                           the real, one-time evaluation.

Protocol difference from Phases 4-6: the notebook must be executed against the real test set EXACTLY ONCE, so the usual
"two-pass" build (execute, write findings, rebuild, re-execute) is replaced by an in-place patch of the findings cell.
"""
import sys
from pathlib import Path
import nbformat as nbf

NB_PATH = Path("notebooks/08_final_test_evaluation.ipynb")
FINDINGS_MARK = "<!--PHASE7_FINDINGS-->"
FINDINGS_PATH = Path("src/phase7_findings.md")

if "--embed-findings" in sys.argv:
    nb = nbf.read(NB_PATH, as_version=4)
    hit = [c for c in nb.cells if c.cell_type == "markdown" and c.source.startswith(FINDINGS_MARK)]
    assert len(hit) == 1, "findings cell not found"
    assert any(c.cell_type == "code" and c.get("outputs") for c in nb.cells), "notebook has not been executed yet"
    hit[0].source = FINDINGS_MARK + "\n" + FINDINGS_PATH.read_text(encoding="utf-8")
    nbf.write(nb, NB_PATH)
    print("embedded", FINDINGS_PATH, "into the executed notebook (no cell was re-executed)")
    sys.exit(0)

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 08 - Final Test Evaluation (Phase 7)
**This notebook is the one-time, pre-registered final evaluation.** Phases 4-6 (development evidence) are *not* revisited or rewritten here.

## Frozen decisions (recorded before the test set is opened)
| Item | Frozen value | Source |
|---|---|---|
| **Primary model** | **Logistic Regression B** (`LR_B`): CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks standardised; Internships, Projects, Workshops/Certifications one-hot categorical; ExtracurricularActivities, PlacementTraining binary; L2, C = 1, **no class weighting** | Phase 4 `models/phase4/model_configs.json` |
| **Probabilities** | **raw**, no recalibration (Phase 5: sigmoid negligible, isotonic not helpful; no method met the materiality rule) | Phase 5 |
| **Threshold 0.50** | reference / default | Phase 4 |
| **Threshold 0.35** | *selected operating threshold under the F1-oriented criterion* (highest validation F1, ties to nearest 0.50). **Not** called "the best threshold". | Phase 6 `07_threshold_selection.json` |
| **Secondary comparators** | `GB_B` and `RF_B`, frozen Phase 4 configurations, evaluated descriptively. **No re-selection, LR_B is not replaced on test performance.** | Phase 4 |

**Rules after the test set is opened:** no model selection, threshold search, calibration selection, feature selection, hyper-parameter tuning or preprocessing change. The test set is evaluation-only.

## Phase 4/5/6 development model vs Phase 7 final model
| | Phase 4/5/6 *development* model | Phase 7 *final* model |
|---|---|---|
| Specification | frozen pipeline above | **identical** (`sklearn.clone` of the Phase 4 pipeline; parameters asserted equal) |
| Fitted on | **train** (6,938) | **train + validation** (8,425), after all decisions were frozen |
| Used for | validation results, CV, calibration, thresholds, explainability | test predictions only |

Refitting re-estimates the fitted quantities (coefficients, `StandardScaler` mean / SD) on the larger development set; no specification, preprocessing step, hyper-parameter, reference level, feature, regularisation, calibration choice or threshold changes. `OneHotEncoder(drop=[1, 3, 0])` keeps the Phase 4 reference levels explicitly.

## Test-set access history (disclosed)
Test data were not used for model development, selection, tuning, calibration, or threshold selection. Earlier exploratory notebooks (Phases 1-3) accessed the eventual test rows for descriptive analysis (data understanding and split construction, EDA, statistics); these rows were not used to fit or evaluate predictive models. Before this notebook ran, two orientation shell commands (`head -2 data/processed/test.csv` and `wc -l data/processed/*.csv`) read the processed test file outside the audit hook; neither was used for any model decision, tuning, threshold, calibration or fitting. The Phase 4-6 audit hooks describe those notebooks only. In Phase 7 the test set is intentionally accessed **once**, under the audit gate below, after the modelling protocol has been frozen.

## Test-access controls (enforced in code)
1. An audit hook **raises** on any open of the raw `test.csv`, and on any open of the processed `test.csv` unless the gate is open. The gate opens only in the cell that prints the banner, admits **exactly one** open and then raises on a second.
2. The file is read once as bytes (SHA-256 taken from those bytes), then parsed from memory.
3. A marker file `08_test_access_log.json` is written at that moment and **blocks any re-execution** that would open the test set again.
4. The frozen protocol is written to `08_frozen_protocol.json` *before* the test set is opened.
5. The notebook is executed once on the real data; the findings text is embedded afterwards by patching one markdown cell (no re-execution).
""")

code(r"""
import warnings
warnings.simplefilter("always")
import os, sys, io, json, time, hashlib, datetime, ast, re
from pathlib import Path

DRY = os.environ.get("PHASE7_DRY_RUN") == "1"

# ---------------- TEST-SET AUDIT HOOK ----------------
_OPENED = []
GATE = {"allowed": False, "opens": 0}
def make_hook(state, opened):
    def _audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0]); opened.append(p); parts = [s.lower() for s in Path(p).parts]
            if os.path.basename(p).lower() == "test.csv":
                if "raw" in parts:
                    raise PermissionError(f"PHASE 7 VIOLATION: the RAW test file may never be opened ({p})")
                if not state["allowed"]:
                    raise PermissionError(f"PHASE 7 VIOLATION: test.csv opened outside the gated final-evaluation cell ({p})")
                if state["opens"] >= 1:
                    raise PermissionError(f"PHASE 7 VIOLATION: test.csv would be opened a second time ({p})")
                state["opens"] += 1
    return _audit

# unit-test the hook logic on synthetic events (no file is touched)
_st = {"allowed": False, "opens": 0}; _h = make_hook(_st, [])
def _raises(f):
    try: f()
    except PermissionError: return True
    return False
assert _raises(lambda: _h("open", ("data/raw/test.csv", "r", 0)))
assert _raises(lambda: _h("open", ("data/processed/test.csv", "rb", 0)))
_st["allowed"] = True
assert _raises(lambda: _h("open", ("data/raw/test.csv", "rb", 0)))
_h("open", ("data/processed/test.csv", "rb", 0)); assert _st["opens"] == 1
assert _raises(lambda: _h("open", ("data/processed/test.csv", "rb", 0)))
_h("open", ("data/processed/train.csv", "rb", 0))
print("audit-hook logic self-test passed (synthetic events only)")
sys.addaudithook(make_hook(GATE, _OPENED))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, sklearn
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC, RAW, MODELS = ROOT / "data" / "processed", ROOT / "data" / "raw", ROOT / "models" / "phase4"
if DRY:
    OUT = Path(os.environ["PHASE7_DRY_OUT"]); MET, FIG, MOD7 = OUT / "metrics", OUT / "figures", OUT / "models"
    print("*** DRY RUN: validation.csv stands in for test.csv; all outputs go to", OUT, "***")
else:
    MET, FIG, MOD7 = ROOT / "results" / "metrics" / "modeling", ROOT / "results" / "figures" / "modeling", ROOT / "models" / "phase7"
for d in (MET, FIG, MOD7): d.mkdir(parents=True, exist_ok=True)
PHASE_RESULTS = ROOT / "results" / "metrics" / "modeling"        # Phase 4-6 files: read only

SEED, B_BOOT, NB, EPS = 42, 2000, 10, 1e-12
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 110)
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12,
                                      "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
INK, NEUTRAL = "#1d2733", "#6b7785"
MODEL_COL = {"LR_B": "#0072B2", "GB_B": "#009E73", "RF_B": "#D55E00"}      # fixed order, colour follows the model
MODEL_NAME = {"LR_B": "Logistic regression B (primary)", "GB_B": "Gradient boosting B", "RF_B": "Random forest B"}
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); plt.show()
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
print("scikit-learn", sklearn.__version__, "| seed", SEED, "| bootstrap resamples", B_BOOT)
""")

md("## 1. Development data (TRAIN + VALIDATION only), frozen artifacts, integrity baselines")
code(r"""
TARGET = "PlacementStatus"
FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
MODEL_IDS = ["LR_B", "GB_B", "RF_B"]

# ---- integrity baselines at the START of the run (nothing below touches test.csv) ----
def snapshot():
    snap = {"processed_train": sha_file(PROC / "train.csv"), "processed_validation": sha_file(PROC / "validation.csv"),
            "processed_modeling_dataset": sha_file(PROC / "modeling_dataset.csv"),
            "raw_train": sha_file(RAW / "train.csv"), "raw_validation": sha_file(RAW / "validation.csv"),
            "model_configs.json": sha_file(MODELS / "model_configs.json")}
    for m in MODEL_IDS: snap[f"{m}.joblib"] = sha_file(MODELS / f"{m}.joblib")
    return snap
def stat_only(p):                      # os.stat does not open the file
    s = os.stat(p); return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}
SNAP_START = snapshot()
TEST_STAT_START = {"processed_test": stat_only(PROC / "test.csv"), "raw_test": stat_only(RAW / "test.csv")}

# raw files must equal the Phase 1.5 downloaded originals (hash dictionary stored in notebook 02; the raw TEST file is checked by size/mtime only)
nb02 = json.loads((ROOT / "notebooks" / "02_data_quality_and_split.ipynb").read_text(encoding="utf-8"))
src02 = "\n".join("".join(c["source"]) for c in nb02["cells"] if c["cell_type"] == "code")
EXPECTED_RAW = ast.literal_eval(re.search(r"EXPECTED_SHA256\s*=\s*(\{.*?\})", src02, re.S).group(1))
assert SNAP_START["raw_train"] == EXPECTED_RAW["train"] and SNAP_START["raw_validation"] == EXPECTED_RAW["validation"]
print("raw train / validation files identical to the Phase 1.5 originals (SHA-256).")

train = pd.read_csv(PROC / "train.csv"); val = pd.read_csv(PROC / "validation.csv")
for d in (train, val):
    assert d.isna().sum().sum() == 0 and set(d[TARGET]) == {"Placed", "NotPlaced"}
split_sum = pd.read_csv(ROOT / "results" / "metrics" / "02_split_summary.csv").set_index("split")
assert len(train) == split_sum.loc["train", "rows"] and len(val) == split_sum.loc["validation", "rows"]
N_TEST_EXPECTED = int(split_sum.loc["test", "rows"])
assert not (set(map(tuple, train[FEATURES].values)) & set(map(tuple, val[FEATURES].values)))
dev = pd.concat([train, val], ignore_index=True)          # development data = train then validation
X_dev, y_dev = dev[FEATURES].copy(), (dev[TARGET] == "Placed").astype(int).to_numpy()
X_tr, y_tr = train[FEATURES].copy(), (train[TARGET] == "Placed").astype(int).to_numpy()
X_va, y_va = val[FEATURES].copy(), (val[TARGET] == "Placed").astype(int).to_numpy()
for X in (X_dev, X_tr, X_va):
    for c in COUNT: X[c] = X[c].astype(int)
print(f"development data: train {len(train):,} + validation {len(val):,} = {len(dev):,} rows (Placed rate {y_dev.mean():.4f}); expected test rows {N_TEST_EXPECTED:,}")

with open(MODELS / "model_configs.json", encoding="utf-8") as fh: P4 = json.load(fh)
thr_sel = json.loads((PHASE_RESULTS / "07_threshold_selection.json").read_text(encoding="utf-8"))
prov = json.loads((PHASE_RESULTS / "05_provisional_selection.json").read_text(encoding="utf-8"))
T_DEFAULT, T_SEL = 0.50, float(thr_sel["selected_threshold"])
assert prov["provisional_model_id"] == "LR_B" and T_SEL == 0.35 and P4["threshold"] == 0.5
THRESHOLDS = [T_DEFAULT, T_SEL]
""")

md(r"""
## 2. Rebuild the frozen models: specification check, then refit on train + validation
For each of `LR_B`, `GB_B`, `RF_B`: the fitted Phase 4 pipeline is loaded, **cloned** (unfitted copy with identical parameters), and the clone's parameters are asserted equal to the Phase 4 pipeline and to `model_configs.json`.
*Specification-reproduction check:* refitting the clone on **train only** must reproduce the saved Phase 4 validation probabilities (this proves that the clone really is the Phase 4 specification and that the fit is deterministic). Only then is the clone refitted on **train + validation**. Nothing about the test set is involved so far.
""")
code(r"""
def params_equal(a, b):
    pa, pb = a.get_params(deep=True), b.get_params(deep=True)
    if pa.keys() != pb.keys(): return False
    for k in pa:
        if k in ("memory",): continue
        va, vb = pa[k], pb[k]
        if hasattr(va, "get_params"): continue                   # nested estimators are compared through their own flattened keys
        if isinstance(va, (list, tuple)) and any(hasattr(e, "get_params") or (isinstance(e, tuple) and any(hasattr(x, "get_params") for x in e)) for e in va): continue
        if va != vb and not (va is None and vb is None): return False
    return True

p4val = pd.read_csv(PHASE_RESULTS / "05_validation_predictions.csv"); assert (p4val.y_true.to_numpy() == y_va).all()
dev_models, final_models, spec_rows = {}, {}, []
t0 = time.time()
for m in MODEL_IDS:
    dev_models[m] = joblib.load(MODELS / f"{m}.joblib")                    # Phase 4 development model (fitted on train)
    cfg = P4["models"][m]["config"]
    spec = clone(dev_models[m]); assert params_equal(spec, dev_models[m])
    mp = spec.get_params()
    for k, v in cfg.items(): assert mp[f"model__{k}"] == v, (m, k, mp[f"model__{k}"], v)
    # specification-reproduction check: clone fitted on TRAIN only reproduces the saved Phase 4 validation probabilities
    chk = clone(spec).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    assert np.allclose(chk, p4val[m].to_numpy(), atol=1e-8), f"{m}: clone fitted on train does not reproduce Phase 4 validation probabilities"
    assert np.allclose(dev_models[m].predict_proba(X_va)[:, 1], p4val[m].to_numpy(), atol=1e-10)
    final_models[m] = clone(spec).fit(X_dev, y_dev)                         # PHASE 7 FINAL MODEL: frozen spec on train + validation
    spec_rows.append({"model": m, "config": json.dumps(cfg), "spec_equals_phase4": True, "reproduces_phase4_validation_probs": True, "n_fit": len(X_dev)})
    print(f"{m}: specification identical to Phase 4 ({cfg}); reproduction check passed; refitted on {len(X_dev):,} rows")
print(f"done in {time.time() - t0:.0f}s")

lr_final = final_models["LR_B"]; lr_dev = dev_models["LR_B"]
assert lr_final.get_params()["model__C"] == 1.0 and lr_final.get_params()["model__class_weight"] is None
assert list(lr_final.named_steps["prep"].named_transformers_["cnt"].drop) == [1, 3, 0]
sc_dev, sc_fin = lr_dev.named_steps["prep"].named_transformers_["num"], lr_final.named_steps["prep"].named_transformers_["num"]
cmp = pd.DataFrame({"feature": ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"], "mean_train_only": sc_dev.mean_, "mean_train_plus_val": sc_fin.mean_,
                    "sd_train_only": sc_dev.scale_, "sd_train_plus_val": sc_fin.scale_}).round(4)
cf = pd.DataFrame({"feature": lr_dev.named_steps["prep"].get_feature_names_out(), "coef_train_only": lr_dev.named_steps["model"].coef_[0], "coef_train_plus_val": lr_final.named_steps["model"].coef_[0]})
cf["abs_change"] = (cf.coef_train_plus_val - cf.coef_train_only).abs()
print("StandardScaler means / SDs are re-estimated on train + validation (part of 'refit the same specification'):"); display(cmp.set_index("feature"))
print(f"LR_B coefficients: max |change| from refitting on train + validation = {cf.abs_change.max():.4f} log-odds"); display(cf.round(4).set_index("feature"))
""")

md(r"""
## 3. Pre-registration record (written BEFORE the test set is opened)
`08_frozen_protocol.json` stores the model specification, thresholds, metric list, bootstrap settings and calibration-bin scheme, plus SHA-256 fingerprints of every frozen artifact.
""")
code(r"""
PROTOCOL = {
    "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "written_before_test_open": True, "dry_run": DRY,
    "primary_model": {"id": "LR_B", "config": P4["models"]["LR_B"]["config"], "representation": "B (5 standardised continuous; Internships/Projects/Workshops one-hot; 2 binary one-hot)",
                      "probabilities": "raw, no recalibration", "class_weight": None},
    "secondary_comparators": {m: P4["models"][m]["config"] for m in ["GB_B", "RF_B"]},
    "thresholds": {"reference_default": T_DEFAULT, "selected_operating_threshold_under_F1_oriented_criterion": T_SEL, "rule": "predict Placed when p >= threshold; no other threshold is evaluated"},
    "final_fit": {"fitted_on": "train + validation", "n": int(len(X_dev)), "specification_changed": False},
    "metrics": ["roc_auc", "log_loss", "brier", "calibration_intercept (joint Cox)", "calibration_slope", "citl", "ece (10 equal-count bins)", "accuracy", "precision", "recall/sensitivity", "f1", "specificity", "TP/TN/FP/FN"],
    "bootstrap": {"resamples": B_BOOT, "seed": SEED, "interval": "percentile 95%", "unit": "test student"},
    "calibration_bins": f"{NB} equal-count bins, Wilson 95% intervals; no recalibration after seeing the test result",
    "prohibited_after_open": ["model selection", "threshold tuning", "calibration selection", "feature selection", "hyper-parameter tuning", "preprocessing changes"],
    "artifact_sha256_at_start": SNAP_START, "test_file_stat_at_start": TEST_STAT_START}
with open(MET / "08_frozen_protocol.json", "w", encoding="utf-8") as fh: json.dump(PROTOCOL, fh, indent=2)
print(json.dumps({k: PROTOCOL[k] for k in ("primary_model", "secondary_comparators", "thresholds", "final_fit", "bootstrap")}, indent=2))
""")

md("# THE TEST SET IS OPENED ONCE, IN THE NEXT CELL")
code(r"""
MARKER = MET / "08_test_access_log.json"
if MARKER.exists():
    raise RuntimeError(f"{MARKER.name} exists: the test set was already opened. A second evaluation is not allowed.")
TEST_PATH = (PROC / "validation.csv") if DRY else (PROC / "test.csv")
print("FINAL TEST EVALUATION — TEST SET IS NOW BEING OPENED" + ("   [DRY RUN: validation.csv as stand-in]" if DRY else ""))
GATE["allowed"] = True
with open(TEST_PATH, "rb") as _fh: _test_bytes = _fh.read()           # the ONE read of the test file
GATE["allowed"] = False
TEST_SHA = hashlib.sha256(_test_bytes).hexdigest()
test = pd.read_csv(io.BytesIO(_test_bytes)); del _test_bytes
with open(MARKER, "w", encoding="utf-8") as fh:
    json.dump({"opened_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "file": TEST_PATH.name, "sha256": TEST_SHA, "rows": int(len(test)), "dry_run": DRY,
               "note": "written immediately after the single read; its existence blocks any second execution of the opening cell"}, fh, indent=2)
print("test rows:", len(test), "| SHA-256:", TEST_SHA, "| gate opens used:", GATE["opens"])

# ---- schema checks only (no modelling decision can depend on these) ----
assert list(test.columns) == FEATURES + [TARGET], "unexpected test columns"
assert test.isna().sum().sum() == 0 and set(test[TARGET]) == {"Placed", "NotPlaced"}
assert DRY or len(test) == N_TEST_EXPECTED, (len(test), N_TEST_EXPECTED)
X_te, y_te = test[FEATURES].copy(), (test[TARGET] == "Placed").astype(int).to_numpy()
for c in COUNT: X_te[c] = X_te[c].astype(int)
unseen = {c: sorted(set(X_te[c]) - set(X_dev[c])) for c in COUNT}
assert not any(unseen.values()), f"count levels unseen in train+validation: {unseen}"      # would make OneHotEncoder(handle_unknown='error') fail: report, do not work around
N_TE = len(test); TEST_RATE = y_te.mean()
print(f"test set: n = {N_TE:,}; observed placement rate {TEST_RATE:.4f} (train+validation {y_dev.mean():.4f}); no unseen count levels")
""")
code(r"""
# data-integrity check on the test split (a leakage check, not an analysis): feature profiles shared with train+validation
overlap = len(set(map(tuple, test[FEATURES].values)) & set(map(tuple, dev[FEATURES].values)))
print("feature profiles in test that also occur in train+validation:", overlap, "(Phase 1.5 split design: expected 0)")
assert overlap == 0 or DRY
""")

md(r"""
## 4. Predictions (one per test student) from the three frozen, refitted models
""")
code(r"""
proba = {m: final_models[m].predict_proba(X_te)[:, 1] for m in MODEL_IDS}
for m in MODEL_IDS:
    assert proba[m].shape == (N_TE,) and np.isfinite(proba[m]).all() and ((proba[m] >= 0) & (proba[m] <= 1)).all()
pred = pd.DataFrame({"test_row_id": np.arange(N_TE), "actual_target": y_te,
                     "lr_probability": proba["LR_B"], "gb_probability": proba["GB_B"], "rf_probability": proba["RF_B"]})
for m, short in [("LR_B", "lr"), ("GB_B", "gb"), ("RF_B", "rf")]:
    pred[f"{short}_prediction_050"] = (proba[m] >= 0.50).astype(int); pred[f"{short}_prediction_035"] = (proba[m] >= 0.35).astype(int)
assert len(pred) == N_TE and pred.test_row_id.is_unique and pred.isna().sum().sum() == 0
pred.to_csv(MET / "08_test_predictions.csv", index=False)
print(f"saved {len(pred):,} predictions (one per test row; no missing; probabilities in [0, 1]):"); display(pred.head())
""")

md(r"""
## 5. Metric helpers (identical definitions to Phases 5-6 so comparisons are like-for-like)
Classification rule: **predict Placed when `p >= threshold`**. Calibration: unpenalised logistic regression `logit P(Placed) = a + b * logit(p_hat)` gives the joint calibration **intercept** `a` and **slope** `b` (ideal 0 and 1); **CITL** is the intercept with the slope fixed at 1; **ECE** is the weighted mean absolute gap over 10 equal-count bins, shown beside its noise yardstick (ECE expected if the probabilities were perfectly calibrated).
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
    rng = np.random.default_rng(seed); chunks = np.array_split(np.argsort(p, kind="mergesort"), NB); N = len(p)
    mp = [p[c].mean() for c in chunks]; out = np.empty(B)
    for b in range(B):
        yb = (rng.random(N) < p).astype(int); out[b] = sum(len(c) / N * abs(m_ - yb[c].mean()) for c, m_ in zip(chunks, mp))
    return {"ece_q_null_mean": float(out.mean()), "ece_q_null_p95": float(np.percentile(out, 95))}
def cal_metrics(p, y):
    z = logit(p); n = len(y)
    citl = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=z).fit(); joint = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    ci_c, ci_j = citl.conf_int(), joint.conf_int()
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "citl": float(citl.params[0]), "citl_ci_lo": float(ci_c[0][0]), "citl_ci_hi": float(ci_c[0][1]),
            "cal_intercept": float(joint.params[0]), "cal_intercept_ci_lo": float(ci_j[0][0]), "cal_intercept_ci_hi": float(ci_j[0][1]),
            "cal_slope": float(joint.params[1]), "slope_ci_lo": float(ci_j[1][0]), "slope_ci_hi": float(ci_j[1][1]),
            "ece_quantile": ece_quantile(p, y), "ece_uniform": ece_uniform(p, y), **ece_null_summary(p), "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}
def class_metrics(y, p, t):
    pr = p >= t; tp = int((pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum()); n = len(y)
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn); spec = tn / (tn + fp)
    return {"threshold": t, "n": n, "accuracy": (tp + tn) / n, "precision": prec, "recall": rec, "f1": 2 * tp / (2 * tp + fp + fn), "specificity": spec, "sensitivity": rec,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "predicted_placement_rate": (tp + fp) / n}
""")

md(r"""
# PART A - Primary model `LR_B` on the test set
## A1. Discrimination and probability quality; calibration
""")
code(r"""
cal = {m: cal_metrics(proba[m], y_te) for m in MODEL_IDS}
lr = cal["LR_B"]
print(f"LR_B on TEST (n = {N_TE:,}, observed placement rate {TEST_RATE:.4f}, mean predicted {lr['mean_pred']:.4f})")
prim = pd.DataFrame({"value": {"ROC-AUC": lr["roc_auc"], "Log loss": lr["log_loss"], "Brier": lr["brier"], "Calibration intercept (joint)": lr["cal_intercept"],
                               "Calibration slope": lr["cal_slope"], "CITL (slope fixed at 1)": lr["citl"], "ECE (10 equal-count bins)": lr["ece_quantile"],
                               "ECE if perfectly calibrated: mean": lr["ece_q_null_mean"], "ECE if perfectly calibrated: 95th pct": lr["ece_q_null_p95"], "ECE (10 equal-width bins; sensitivity)": lr["ece_uniform"]}})
prim["95% CI"] = ["", "", "", f"({lr['cal_intercept_ci_lo']:+.3f}, {lr['cal_intercept_ci_hi']:+.3f})", f"({lr['slope_ci_lo']:.3f}, {lr['slope_ci_hi']:.3f})", f"({lr['citl_ci_lo']:+.3f}, {lr['citl_ci_hi']:+.3f})", "", "", "", ""]
display(prim.round(4))
pd.DataFrame(cal).T.rename_axis("model").reset_index().to_csv(MET / "08_test_calibration_metrics.csv", index=False)
""")

md("## A2. Classification metrics at the two frozen thresholds (LR_B)")
code(r"""
rows = []
for m in MODEL_IDS:
    for t in THRESHOLDS: rows.append({"model": m, **class_metrics(y_te, proba[m], t)})
cls = pd.DataFrame(rows); cls.to_csv(MET / "08_test_classification_metrics.csv", index=False)
show = ["threshold", "accuracy", "precision", "recall", "f1", "specificity", "sensitivity", "TP", "TN", "FP", "FN", "predicted_placement_rate"]
for t in THRESHOLDS:
    label = "reference / default" if t == T_DEFAULT else "selected operating threshold under the F1-oriented criterion"
    print(f"LR_B at threshold {t:.2f} ({label}):"); display(cls[(cls.model == "LR_B") & (cls.threshold == t)][show].round(4).set_index("threshold"))
""")

md(r"""
## A3. Threshold comparison (LR_B) - the two pre-declared thresholds only
No other threshold is evaluated and nothing is searched on the test set.
""")
code(r"""
cmp_thr = cls[cls.model == "LR_B"][["threshold", "accuracy", "precision", "recall", "f1", "specificity", "sensitivity", "TP", "TN", "FP", "FN", "predicted_placement_rate"]].reset_index(drop=True)
cmp_thr.to_csv(MET / "08_test_threshold_comparison.csv", index=False)
display(cmp_thr.round(4).set_index("threshold"))
a, b = cmp_thr[cmp_thr.threshold == T_DEFAULT].iloc[0], cmp_thr[cmp_thr.threshold == T_SEL].iloc[0]
print(f"Moving from 0.50 to 0.35: FN {int(a.FN)} -> {int(b.FN)} ({int(b.FN - a.FN):+d}), FP {int(a.FP)} -> {int(b.FP)} ({int(b.FP - a.FP):+d}); "
      f"recall {a.recall:.3f} -> {b.recall:.3f}, precision {a.precision:.3f} -> {b.precision:.3f}, specificity {a.specificity:.3f} -> {b.specificity:.3f}, "
      f"accuracy {a.accuracy:.3f} -> {b.accuracy:.3f}, F1 {a.f1:.3f} -> {b.f1:.3f}")
""")

md(r"""
# PART B - Bootstrap 95% confidence intervals
2,000 resamples of test students, `numpy.random.default_rng(42)`, percentile intervals. The **same resamples** are used for every metric and model, so paired differences between models are valid. Classification metrics are reported at the two frozen thresholds. The intervals describe sampling uncertainty of this one test set; they are not used to change anything.
""")
code(r"""
rng = np.random.default_rng(SEED); IDX = rng.integers(0, N_TE, (B_BOOT, N_TE))
Y = y_te[IDX]
def boot_auc(p): return np.array([roc_auc_score(Y[i], p[IDX[i]]) for i in range(B_BOOT)])
def boot_ll(p): pb = np.clip(p[IDX], EPS, 1 - EPS); return -(Y * np.log(pb) + (1 - Y) * np.log(1 - pb)).mean(1)
def boot_brier(p): return ((p[IDX] - Y) ** 2).mean(1)
def boot_cls(p, t):
    pr = p[IDX] >= t; tp = (pr & (Y == 1)).sum(1); fp = (pr & (Y == 0)).sum(1); fn = (~pr & (Y == 1)).sum(1); tn = (~pr & (Y == 0)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": (tp + tn) / N_TE, "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn), "specificity": tn / (tn + fp)}
BOOT = {}
for m in MODEL_IDS:
    p = proba[m]; BOOT[(m, None, "roc_auc")] = boot_auc(p); BOOT[(m, None, "log_loss")] = boot_ll(p); BOOT[(m, None, "brier")] = boot_brier(p)
    for t in THRESHOLDS:
        for k, v in boot_cls(p, t).items(): BOOT[(m, t, k)] = v
est = {}
for m in MODEL_IDS:
    est[(m, None, "roc_auc")], est[(m, None, "log_loss")], est[(m, None, "brier")] = cal[m]["roc_auc"], cal[m]["log_loss"], cal[m]["brier"]
    for t in THRESHOLDS:
        r = class_metrics(y_te, proba[m], t)
        for k in ("accuracy", "precision", "recall", "f1", "specificity"): est[(m, t, k)] = r[k]
ci_rows = []
for (m, t, k), v in BOOT.items():
    v = v[np.isfinite(v)]; lo, hi = np.percentile(v, [2.5, 97.5])
    ci_rows.append({"model": m, "threshold": t if t is not None else "", "metric": k, "estimate": est[(m, t, k)], "lower_95": lo, "upper_95": hi, "n_bootstrap_used": len(v)})
# paired differences (LR_B minus comparator), same resamples
for o in ["GB_B", "RF_B"]:
    for k in ("roc_auc", "log_loss", "brier"):
        d = BOOT[("LR_B", None, k)] - BOOT[(o, None, k)]; lo, hi = np.percentile(d, [2.5, 97.5])
        ci_rows.append({"model": f"LR_B minus {o} (paired)", "threshold": "", "metric": k, "estimate": est[("LR_B", None, k)] - est[(o, None, k)], "lower_95": lo, "upper_95": hi, "n_bootstrap_used": B_BOOT})
ci = pd.DataFrame(ci_rows); ci.to_csv(MET / "08_test_bootstrap_ci.csv", index=False)
print("LR_B (primary), 95% bootstrap intervals:")
display(ci[ci.model == "LR_B"][["threshold", "metric", "estimate", "lower_95", "upper_95"]].round(4).reset_index(drop=True))
print("Secondary models and paired differences (LR_B minus other; negative log loss / Brier favours LR_B):")
display(ci[(ci.model != "LR_B") & (ci.metric.isin(["roc_auc", "log_loss", "brier"]))][["model", "metric", "estimate", "lower_95", "upper_95"]].round(4).reset_index(drop=True))
""")

md(r"""
# PART C - Test calibration of `LR_B` (reliability diagram)
10 equal-count bins (about 149 students each), Wilson 95% intervals. **No recalibration is performed**, whatever the result: any imperfection is reported as a limitation.
""")
code(r"""
bins = quantile_bins(proba["LR_B"], y_te); bins.insert(0, "model", "LR_B")
bins[["model", "bin", "n", "p_min", "p_max", "mean_pred", "obs_rate", "calibration_gap", "ci_lo", "ci_hi"]].to_csv(MET / "08_test_calibration_bins.csv", index=False)
bins["gap_outside_wilson_ci"] = (bins.mean_pred < bins.ci_lo) | (bins.mean_pred > bins.ci_hi)
display(bins.rename(columns={"mean_pred": "mean_predicted"}).drop(columns="model").round(4).set_index("bin"))
print(f"bins whose mean prediction lies outside the Wilson 95% interval of the observed rate: {int(bins.gap_outside_wilson_ci.sum())} of {NB} (about 0.5 expected by chance if calibrated)")
print(f"max |gap| = {bins.calibration_gap.abs().max():.3f}; ECE = {lr['ece_quantile']:.4f} (perfect-calibration noise: mean {lr['ece_q_null_mean']:.4f}, 95th pct {lr['ece_q_null_p95']:.4f}); Brier = {lr['brier']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]; ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2, label="Perfect calibration")
ax.errorbar(bins.mean_pred, bins.obs_rate, yerr=[bins.obs_rate - bins.ci_lo, bins.ci_hi - bins.obs_rate], fmt="o-", color=MODEL_COL["LR_B"], ecolor=MODEL_COL["LR_B"], elinewidth=1.4, capsize=3, ms=6, lw=1.5,
            markeredgecolor="white", label="LR_B, test (observed rate, Wilson 95%)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.set_xlabel("Mean predicted probability (bin)"); ax.set_ylabel("Observed placement rate (bin)")
ax.set_title("Reliability diagram, test set (10 equal-count bins)"); ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.text(0.97, 0.05, f"intercept {lr['cal_intercept']:+.2f}   slope {lr['cal_slope']:.2f}\nCITL {lr['citl']:+.2f}   ECE {lr['ece_quantile']:.3f}\nBrier {lr['brier']:.3f}   n = {N_TE:,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c9ced4"))
ax = axes[1]
ax.bar(bins.bin, -bins.calibration_gap, color=np.where(bins.gap_outside_wilson_ci, MODEL_COL["RF_B"], MODEL_COL["LR_B"]), width=0.7)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(bins.bin); ax.set_xlabel("Bin (1 = lowest predicted probability)"); ax.set_ylabel("Observed minus mean predicted")
ax.set_title("Per-bin calibration gap (above 0: observed higher than predicted)"); ax.grid(axis="x", visible=False)
note = ("orange bar: mean prediction outside the Wilson 95% interval" if bins.gap_outside_wilson_ci.any()
        else f"all {NB} bin mean predictions lie inside the Wilson 95% interval of the observed rate")
ax.text(0.015, 0.985, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NEUTRAL)
savefig(fig, "phase7_test_reliability")
""")

md(r"""
# PART D - Test probability distribution of `LR_B`
Compared descriptively with the Phase 5 out-of-fold training distribution (a different data set; differences are not treated as deterioration without evidence).
""")
code(r"""
p = proba["LR_B"]; q1, med, q3 = np.percentile(p, [25, 50, 75])
lo_m, mid_m, hi_m = p < 0.2, (p >= 0.2) & (p <= 0.8), p > 0.8
dist = {"min": p.min(), "max": p.max(), "mean": p.mean(), "median": med, "q1": q1, "q3": q3,
        "share_below_0.2": lo_m.mean(), "share_0.2_to_0.8": mid_m.mean(), "share_above_0.8": hi_m.mean(),
        "n_below_0.2": int(lo_m.sum()), "n_0.2_to_0.8": int(mid_m.sum()), "n_above_0.8": int(hi_m.sum()),
        "observed_rate_below_0.2": y_te[lo_m].mean(), "observed_rate_0.2_to_0.8": y_te[mid_m].mean(), "observed_rate_above_0.8": y_te[hi_m].mean(),
        "mean_pred_below_0.2": p[lo_m].mean(), "mean_pred_0.2_to_0.8": p[mid_m].mean(), "mean_pred_above_0.8": p[hi_m].mean()}
pd.DataFrame([{"model": "LR_B", **dist}]).to_csv(MET / "08_test_probability_distribution_summary.csv", index=False)
p5 = pd.read_csv(PHASE_RESULTS / "06_probability_distribution_summary.csv"); p5 = p5[p5.model_id == "LR_B"].iloc[0]
tab = pd.DataFrame({"Phase 7 test (refit model)": [dist["min"], dist["max"], dist["mean"], dist["median"], dist["q1"], dist["q3"], dist["share_below_0.2"], dist["share_0.2_to_0.8"], dist["share_above_0.8"],
                                                    dist["observed_rate_below_0.2"], dist["observed_rate_above_0.8"]],
                    "Phase 5 train OOF": [p5["min"], p5["max"], np.nan, p5.q50, p5.q25, p5.q75, p5["share_below_0.2"], p5["share_0.2_to_0.8"], p5["share_above_0.8"],
                                          p5["observed_rate_below_0.2"], p5["observed_rate_above_0.8"]]},
                   index=["minimum", "maximum", "mean", "median", "Q1", "Q3", "share below 0.20", "share 0.20-0.80", "share above 0.80", "observed placed rate | p < 0.20", "observed placed rate | p > 0.80"])
display(tab.round(4))
print(f"Extreme groups on test: p < 0.20: n = {dist['n_below_0.2']} ({dist['share_below_0.2']:.1%}), observed placed rate {dist['observed_rate_below_0.2']:.3f} (mean predicted {dist['mean_pred_below_0.2']:.3f}); "
      f"p > 0.80: n = {dist['n_above_0.8']} ({dist['share_above_0.8']:.1%}), observed placed rate {dist['observed_rate_above_0.8']:.3f} (mean predicted {dist['mean_pred_above_0.8']:.3f}); "
      f"middle band: n = {dist['n_0.2_to_0.8']} ({dist['share_0.2_to_0.8']:.1%}), observed {dist['observed_rate_0.2_to_0.8']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]; edges = np.linspace(0, 1, 26)
ax.hist([p[y_te == 0], p[y_te == 1]], bins=edges, stacked=True, color=["#E69F00", "#0072B2"], edgecolor="white", linewidth=0.6, label=["Actually NotPlaced", "Actually Placed"])
for x, ls, lab in [(0.35, "--", "Threshold 0.35"), (0.50, ":", "Threshold 0.50")]: ax.axvline(x, color=INK, ls=ls, lw=1.4, label=lab)
ax.axvspan(0.2, 0.8, color="#e3e6ea", alpha=0.45, zorder=0, label="Middle band 0.20-0.80")
ax.set_xlabel("Predicted placement probability (LR_B, raw)"); ax.set_ylabel("Number of test students"); ax.set_title(f"Test-set probability distribution (n = {N_TE:,})"); ax.legend(frameon=False, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
ax = axes[1]; xb = np.arange(3); w = 0.38
v7 = [dist["share_below_0.2"], dist["share_0.2_to_0.8"], dist["share_above_0.8"]]; v5 = [p5["share_below_0.2"], p5["share_0.2_to_0.8"], p5["share_above_0.8"]]
b7 = ax.bar(xb - w / 2, v7, w, color="#0072B2", label="Phase 7 test"); b5 = ax.bar(xb + w / 2, v5, w, color="#9aa3ad", label="Phase 5 train OOF")
for bars in (b7, b5):
    for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.0%}", ha="center", fontsize=9, color=INK)
ax.set_xticks(xb); ax.set_xticklabels(["p < 0.20", "0.20 - 0.80", "p > 0.80"]); ax.set_ylabel("Share of students"); ax.set_ylim(0, max(v7 + v5) * 1.18); ax.grid(axis="x", visible=False)
ax.set_title("Share of students by probability band"); ax.legend(frameon=False, fontsize=9)
savefig(fig, "phase7_test_probability_distribution")
""")

md(r"""
# PART E - Development vs test comparison (LR_B)
Development numbers are read from the saved Phase 4/5 result files (not retyped). **Different models and data are compared**: Phase 5 CV and Phase 4 validation come from models fitted on train (or 80% of it); the Phase 7 test number comes from the refit on train + validation. The CV fold standard deviation (folds of about 1,390 students, similar to the test size of 1,487) is a rough yardstick for single-split sampling variation. The validation split also informed model selection, so its numbers are slightly optimistic.
""")
code(r"""
cv = pd.read_csv(PHASE_RESULTS / "06_cv_summary.csv"); cv = cv[cv.model_id == "LR_B"].set_index("metric")
v4 = pd.read_csv(PHASE_RESULTS / "05_validation_results.csv").set_index("model_id").loc["LR_B"]
cm5 = pd.read_csv(PHASE_RESULTS / "06_calibration_metrics.csv"); cm5 = cm5[(cm5.model_id == "LR_B") & (cm5.method == "original")].set_index("dataset").loc["train_nested_CV"]
vc = pd.read_csv(PHASE_RESULTS / "06_validation_confirmation.csv"); vc = vc[(vc.model_id == "LR_B") & (vc.method == "original")].iloc[0]
cit = ci[(ci.model == "LR_B") & (ci.threshold == "")].set_index("metric")
dvt = pd.DataFrame([
    {"source": "Phase 5 train 5-fold CV (mean over folds)", "model_fitted_on": "80% of train per fold", "n": 6938, "roc_auc": cv.loc["roc_auc", "mean"], "log_loss": cv.loc["log_loss", "mean"], "brier": cv.loc["brier", "mean"],
     "roc_auc_sd_or_ci": f"fold SD {cv.loc['roc_auc', 'sd']:.4f}", "log_loss_sd_or_ci": f"fold SD {cv.loc['log_loss', 'sd']:.4f}", "brier_sd_or_ci": f"fold SD {cv.loc['brier', 'sd']:.4f}",
     "cal_slope": cm5.cal_slope, "cal_intercept": cm5.cal_intercept, "ece_quantile": cm5.ece_quantile},
    {"source": "Phase 4 validation (single split)", "model_fitted_on": "train", "n": 1487, "roc_auc": v4.roc_auc, "log_loss": v4.log_loss, "brier": v4.brier,
     "roc_auc_sd_or_ci": f"95% CI {v4.auc_ci_lo:.3f}-{v4.auc_ci_hi:.3f}", "log_loss_sd_or_ci": f"95% CI {v4.ll_ci_lo:.3f}-{v4.ll_ci_hi:.3f}", "brier_sd_or_ci": f"95% CI {v4.brier_ci_lo:.3f}-{v4.brier_ci_hi:.3f}",
     "cal_slope": vc.cal_slope, "cal_intercept": vc.cal_intercept, "ece_quantile": vc.ece_quantile},
    {"source": "Phase 7 TEST (single split, final model)", "model_fitted_on": "train + validation", "n": N_TE, "roc_auc": lr["roc_auc"], "log_loss": lr["log_loss"], "brier": lr["brier"],
     "roc_auc_sd_or_ci": f"95% CI {cit.loc['roc_auc', 'lower_95']:.3f}-{cit.loc['roc_auc', 'upper_95']:.3f}", "log_loss_sd_or_ci": f"95% CI {cit.loc['log_loss', 'lower_95']:.3f}-{cit.loc['log_loss', 'upper_95']:.3f}",
     "brier_sd_or_ci": f"95% CI {cit.loc['brier', 'lower_95']:.3f}-{cit.loc['brier', 'upper_95']:.3f}", "cal_slope": lr["cal_slope"], "cal_intercept": lr["cal_intercept"], "ece_quantile": lr["ece_quantile"]}])
dvt.to_csv(MET / "08_development_vs_test.csv", index=False)
display(dvt.round(4).set_index("source").T)
print("Test minus Phase 5 CV mean, in units of the CV fold SD (positive AUC = better; positive log loss / Brier = worse):")
for k in ("roc_auc", "log_loss", "brier"): print(f"  {k:9s}: {lr[k] - cv.loc[k, 'mean']:+.4f}  = {(lr[k] - cv.loc[k, 'mean']) / cv.loc[k, 'sd']:+.2f} fold SD;  test minus validation: {lr[k] - v4[k]:+.4f}")
print("Does the test 95% bootstrap interval contain the Phase 5 CV mean / the Phase 4 validation value?")
for k in ("roc_auc", "log_loss", "brier"):
    lo_, hi_ = cit.loc[k, "lower_95"], cit.loc[k, "upper_95"]; print(f"  {k:9s}: CV mean {lo_ <= cv.loc[k, 'mean'] <= hi_}, validation {lo_ <= v4[k] <= hi_}")
""")

md(r"""
# PART F - Secondary models (descriptive only)
`GB_B` and `RF_B` are evaluated on the identical test set, frozen configurations, refitted on train + validation. They are **not** used to select or replace the primary model.
""")
code(r"""
rows = []
for m in MODEL_IDS:
    r = {"model": m, "model_name": MODEL_NAME[m], "roc_auc": cal[m]["roc_auc"], "log_loss": cal[m]["log_loss"], "brier": cal[m]["brier"],
         "roc_auc_lo95": ci[(ci.model == m) & (ci.metric == "roc_auc")].lower_95.iloc[0], "roc_auc_hi95": ci[(ci.model == m) & (ci.metric == "roc_auc")].upper_95.iloc[0],
         "log_loss_lo95": ci[(ci.model == m) & (ci.metric == "log_loss")].lower_95.iloc[0], "log_loss_hi95": ci[(ci.model == m) & (ci.metric == "log_loss")].upper_95.iloc[0],
         "brier_lo95": ci[(ci.model == m) & (ci.metric == "brier")].lower_95.iloc[0], "brier_hi95": ci[(ci.model == m) & (ci.metric == "brier")].upper_95.iloc[0],
         "f1_at_0.50": class_metrics(y_te, proba[m], 0.50)["f1"], "f1_at_0.35": class_metrics(y_te, proba[m], T_SEL)["f1"],
         "cal_slope": cal[m]["cal_slope"], "cal_intercept": cal[m]["cal_intercept"], "ece_quantile": cal[m]["ece_quantile"], "role": "primary (frozen)" if m == "LR_B" else "secondary comparator (frozen)"}
    rows.append(r)
mc = pd.DataFrame(rows); mc.to_csv(MET / "08_final_model_comparison.csv", index=False)
display(mc[["model", "roc_auc", "log_loss", "brier", "f1_at_0.50", "f1_at_0.35", "cal_slope", "ece_quantile"]].round(4).set_index("model"))
sec = cls[cls.model.isin(["GB_B", "RF_B"])][["model", "threshold", "accuracy", "precision", "recall", "f1", "specificity", "sensitivity", "TP", "TN", "FP", "FN"]]
print("Secondary models, classification metrics at the two frozen thresholds:"); display(sec.round(4).set_index(["model", "threshold"]))
print("Paired bootstrap differences, LR_B minus comparator (intervals including 0 = no distinguishable difference):")
display(ci[ci.model.str.contains("paired")][["model", "metric", "estimate", "lower_95", "upper_95"]].round(4).reset_index(drop=True))
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
for ax, (k, ttl, better) in zip(axes, [("roc_auc", "ROC-AUC", "higher is better"), ("log_loss", "Log loss", "lower is better"), ("brier", "Brier score", "lower is better")]):
    for i, m in enumerate(MODEL_IDS):
        r = mc[mc.model == m].iloc[0]
        ax.errorbar(r[k], i, xerr=[[r[k] - r[f"{k}_lo95"]], [r[f"{k}_hi95"] - r[k]]], fmt="o", color=MODEL_COL[m], ecolor=MODEL_COL[m], elinewidth=2, capsize=4, ms=9, markeredgecolor="white")
        ax.text(r[k], i - 0.28, f"{r[k]:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(["LR_B\n(primary)", "GB_B", "RF_B"] if ax is axes[0] else [""] * 3); ax.set_ylim(2.6, -0.7); ax.grid(axis="y", visible=False)
    ax.set_title(f"{ttl}  ({better})", fontsize=11); ax.set_xlabel(f"{ttl} on test (point = estimate, bar = bootstrap 95% CI)", fontsize=9)
fig.suptitle("Final test comparison of the frozen models: descriptive only, each panel has its own axis", y=1.03, fontsize=13, fontweight="semibold")
savefig(fig, "phase7_final_model_comparison")
""")

md(r"""
# PART G - Confusion matrices (LR_B), identical conventions
Rows = actual class, columns = predicted class; cell colour = share of the actual class (same 0-1 scale in both figures); text = count and share.
""")
code(r"""
for t, tag in [(0.50, "050"), (0.35, "035")]:
    r = class_metrics(y_te, proba["LR_B"], t); cmx = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    fig, ax = plt.subplots(figsize=(5.6, 5.0)); share = cmx / cmx.sum(1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (a_, b_), v in np.ndenumerate(cmx): ax.text(b_, a_, f"{v:,}\n({share[a_, b_]:.1%})", ha="center", va="center", fontsize=13, fontweight="semibold", color="white" if share[a_, b_] > 0.55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nNotPlaced", "Predicted\nPlaced"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nNotPlaced", "Actual\nPlaced"]); ax.grid(False)
    tl = "reference / default" if t == 0.50 else "selected operating threshold, F1-oriented criterion"
    ax.set_title(f"LR_B on test, threshold {t:.2f}\n({tl})", fontsize=11)
    ax.set_xlabel(f"accuracy {r['accuracy']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  specificity {r['specificity']:.3f}  F1 {r['f1']:.3f}", fontsize=8.6)
    savefig(fig, f"phase7_confusion_matrix_{tag}")
""")

md("## Phase 7 conclusions")
C.append(nbf.v4.new_markdown_cell(FINDINGS_MARK + "\n_Findings pending (embedded after the single real execution)._"))

md("# FINAL AUDIT")
code(r"""
SNAP_END = snapshot()
assert SNAP_END == SNAP_START, {k: (SNAP_START[k], SNAP_END[k]) for k in SNAP_START if SNAP_START[k] != SNAP_END[k]}
TEST_STAT_END = {"processed_test": stat_only(PROC / "test.csv"), "raw_test": stat_only(RAW / "test.csv")}
assert TEST_STAT_END == TEST_STAT_START, "test-file size/mtime changed"
# (the test file is NOT hashed again - that would be a second read; the single-read SHA-256 is recorded in the access marker)
print("DATA: raw train/validation SHA-256 = Phase 1.5 originals; processed train, validation, modeling_dataset and Phase 4 model files byte-identical at start and end of this run;")
print("      raw and processed test files unchanged (size and mtime); the raw test file was not opened by this notebook.")

# models: frozen specifications unchanged
with open(MODELS / "model_configs.json", encoding="utf-8") as fh: P4_END = json.load(fh)
assert P4_END == P4
for m in MODEL_IDS:
    assert params_equal(final_models[m], dev_models[m]) and final_models[m].get_params()["model__random_state"] == 42
    for k, v in P4["models"][m]["config"].items(): assert final_models[m].get_params()[f"model__{k}"] == v
assert THRESHOLDS == [0.50, 0.35] and set(cls.threshold) == {0.50, 0.35} and set(cmp_thr.threshold) == {0.50, 0.35}
print("MODELS: LR_B, GB_B, RF_B specifications identical to Phase 4 (parameters asserted); thresholds 0.50 and 0.35 only; raw probabilities, no recalibration.")

if not DRY:
    for m in MODEL_IDS: joblib.dump(final_models[m], MOD7 / f"{m}_final.joblib")
    with open(MOD7 / "final_model_configs.json", "w", encoding="utf-8") as fh:
        json.dump({"fitted_on": "train + validation", "n": int(len(X_dev)), "specification_source": "models/phase4/model_configs.json (unchanged)", "models": {m: P4["models"][m] for m in MODEL_IDS},
                   "thresholds": THRESHOLDS, "sklearn_version": sklearn.__version__, "test_file_sha256": TEST_SHA}, fh, indent=2)
    print("Phase 7 final models saved to models/phase7/ (Phase 4 artifacts untouched).")

# test protocol
data_ext = (".csv", ".parquet", ".json", ".joblib", ".pkl", ".feather", ".xlsx")
opened = [os.path.normpath(p) for p in _OPENED if p.lower().endswith(data_ext)]
test_opens = [p for p in opened if os.path.basename(p).lower() == "test.csv"]
raw_test_opens = [p for p in opened if os.path.basename(p).lower() == "test.csv" and "raw" in [s.lower() for s in Path(p).parts]]
print("\nData files opened during this run (distinct):"); [print("  ", p) for p in sorted(set(opened))]
assert not raw_test_opens, "raw test file was opened"
assert (len(test_opens) == 1 and GATE["opens"] == 1) if not DRY else (len(test_opens) == 0 and GATE["opens"] == 0)
assert MARKER.exists() and json.loads(MARKER.read_text(encoding="utf-8"))["sha256"] == TEST_SHA
print(f"\nTEST PROTOCOL: processed test.csv opened {GATE['opens'] if not DRY else 'n/a (dry run)'} time(s) by this notebook, inside the gated cell; the raw test file was not opened by this notebook; access marker {MARKER.name} written.")
print("   No model selection, threshold search, calibration selection, feature selection, hyper-parameter tuning or preprocessing change was performed after the test set was opened:")
print("   thresholds, models and calibration status are the frozen values asserted above and recorded in 08_frozen_protocol.json (written before the open).")
print("\nTest data were not used for model development, selection, tuning, calibration, or threshold selection. Phases 1-3 accessed the eventual test rows descriptively; Phase 7 performed the final gated test evaluation only after the protocol was frozen.")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH)
