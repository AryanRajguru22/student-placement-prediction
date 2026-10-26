"""Application-level verification (Phase 8A). Run from the project root:  python app/verify_app.py

Uses only manually constructed examples (plus validation rows for the Phase 6 explanation cross-check). It does NOT use the held-out test split:
an audit hook raises if any file named like the test split / test predictions is opened during this run, and the app source is scanned for such references.
"""
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP))

# ---- test-set protection while verifying ----
_FORBIDDEN = ("test" + ".csv", "08_test" + "_predictions")        # built by concatenation so this file does not trip its own scan
_OPENED = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]); _OPENED.append(p)
        if any(tok in os.path.basename(p).lower() for tok in _FORBIDDEN):
            raise PermissionError(f"verification must not touch the held-out test files: {p}")


sys.addaudithook(_audit)

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from explainability import explain
from prediction import (BANDS, FEATURES, LABELS, classify, load_bundle, predict, probability_band, to_frame, validate_inputs)

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAILED: {name} {detail}"
    OK.append(name)
    print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------------ 0. no test-set code path
print("0. test-set protection")
for py in sorted(APP.glob("*.py")):
    if py.name == "verify_app.py":
        continue
    txt = py.read_text(encoding="utf-8").lower()
    check(f"{py.name}: no reference to the test split / test predictions", not any(tok in txt for tok in _FORBIDDEN))

# ------------------------------------------------------------------ 1. model loads
print("1. model and metadata")
bundle = load_bundle()
meta = bundle.metadata
check("frozen Phase 7 LR_B loads and matches its recorded SHA-256", True, meta["artifact_sha256"][:16])
check("pipeline = Phase 4 LR_B specification (L2, C=1, no class weight, OneHot drop=[1,3,0])",
      bundle.pipeline.get_params()["model__C"] == 1.0 and bundle.pipeline.get_params()["model__class_weight"] is None
      and list(bundle.pipeline.named_steps["prep"].named_transformers_["cnt"].drop) == [1, 3, 0])
sel = json.loads((ROOT / "results/metrics/modeling/07_threshold_selection.json").read_text(encoding="utf-8"))
check("operating threshold equals the Phase 6 selected threshold (0.35); reference is 0.50",
      meta["thresholds"]["operating"] == sel["selected_threshold"] == 0.35 and meta["thresholds"]["reference"] == 0.5)

# ------------------------------------------------------------------ 2. manual examples
EXAMPLES = {
    "strong profile": {"CGPA": 8.7, "Internships": 2, "Projects": 3, "Workshops/Certifications": 2, "AptitudeTestScore": 87, "SoftSkillsRating": 4.6,
                       "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 80, "HSC_Marks": 85},
    "typical profile": {"CGPA": 7.7, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 80, "SoftSkillsRating": 4.4,
                        "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 70, "HSC_Marks": 73},
    "weak profile": {"CGPA": 6.6, "Internships": 0, "Projects": 1, "Workshops/Certifications": 1, "AptitudeTestScore": 62, "SoftSkillsRating": 3.2,
                     "ExtracurricularActivities": "No", "PlacementTraining": "No", "SSC_Marks": 56, "HSC_Marks": 60},
    "weaker mixed profile": {"CGPA": 7.4, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 74,
                                                                "SoftSkillsRating": 4.2, "ExtracurricularActivities": "No", "PlacementTraining": "Yes", "SSC_Marks": 62, "HSC_Marks": 68},
    "rare level (Projects = 0)": {"CGPA": 8.0, "Internships": 1, "Projects": 0, "Workshops/Certifications": 0, "AptitudeTestScore": 78, "SoftSkillsRating": 4.3,
                                  "ExtracurricularActivities": "Yes", "PlacementTraining": "No", "SSC_Marks": 66, "HSC_Marks": 72},
    "range minimum": {"CGPA": 6.5, "Internships": 0, "Projects": 0, "Workshops/Certifications": 0, "AptitudeTestScore": 60, "SoftSkillsRating": 3.0,
                      "ExtracurricularActivities": "No", "PlacementTraining": "No", "SSC_Marks": 55, "HSC_Marks": 57},
    "range maximum": {"CGPA": 9.1, "Internships": 2, "Projects": 3, "Workshops/Certifications": 3, "AptitudeTestScore": 90, "SoftSkillsRating": 4.8,
                      "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 90, "HSC_Marks": 88},
}
direct = joblib.load(ROOT / "models/phase7/LR_B_final.joblib")        # the saved pipeline, loaded independently of the app code
print("2. predictions, thresholds, direct comparison, explanation reconstruction")
for name, raw in EXAMPLES.items():
    print(f" - {name}")
    clean, errs = validate_inputs(raw, meta)
    check("inputs validate", not errs and clean == {k: (float(v) if k in meta["continuous"] else v) for k, v in raw.items()})
    pred = predict(bundle, clean)
    p = pred.probability
    check("probability is a float in [0, 1]", isinstance(p, float) and 0.0 <= p <= 1.0, f"{p:.6f}")
    # (A) the application vs (B) the saved pipeline called directly on an independently built dataframe
    df_direct = pd.DataFrame([raw])[FEATURES]
    for c in meta["count"]:
        df_direct[c] = df_direct[c].astype(int)
    p_direct = float(direct.predict_proba(df_direct)[0, 1])
    check("application probability == direct LR_B probability", abs(p - p_direct) < 1e-12, f"|diff| = {abs(p - p_direct):.2e}")
    check("threshold 0.35 applied (placed iff p >= 0.35)", pred.placed_at_operating == (p >= 0.35))
    check("reference threshold 0.50 applied (placed iff p >= 0.50)", pred.placed_at_reference == (p >= 0.5))
    check("band is consistent with the probability", pred.band == probability_band(p))
    exp = explain(bundle.pipeline, clean)
    z_direct = float(direct.decision_function(df_direct)[0])
    check("intercept + sum(contributions) == decision function (< 1e-9)", abs(exp.logit_reconstructed - z_direct) < 1e-9, f"|diff| = {abs(exp.logit_reconstructed - z_direct):.2e}")
    check("intercept + sum(contributions) == logit(probability) (< 1e-8)", abs(exp.logit_reconstructed - float(logit(p))) < 1e-8)
    check("expit(reconstructed log-odds) == probability (< 1e-12)", abs(float(expit(exp.logit_reconstructed)) - p) < 1e-12)
    # contributions recomputed by hand from the saved coefficients
    pre, lrm = direct.named_steps["prep"], direct.named_steps["model"]
    terms = dict(zip(pre.get_feature_names_out(), np.asarray(pre.transform(df_direct), dtype=float)[0] * lrm.coef_[0]))
    by_hand = {f: 0.0 for f in FEATURES}
    for k, v in terms.items():
        by_hand[max([g for g in FEATURES if k == g or k.startswith(g + "_")], key=len)] += float(v)
    check("per-feature contributions == coefficient x transformed value (hand computation)",
          all(abs(by_hand[f] - float(exp.contributions.set_index("feature").loc[f, "contribution"])) < 1e-12 for f in FEATURES))
    print(f"      p = {p:.4%} | op(0.35) = {'Placed' if pred.placed_at_operating else 'Not Placed'} | ref(0.50) = {'Placed' if pred.placed_at_reference else 'Not Placed'} | {pred.band}")

print("3. threshold boundaries (pure rule)")
check("p = 0.35 exactly -> Placed at 0.35", classify(0.35, 0.35) and not classify(0.3499999999, 0.35))
check("p = 0.50 exactly -> Placed at 0.50", classify(0.5, 0.5) and not classify(0.4999999999, 0.5))
check("a probability between 0.35 and 0.50 is Placed at 0.35 but Not Placed at 0.50", classify(0.42, 0.35) and not classify(0.42, 0.5))
check("probability bands", [probability_band(x) for x in (0.0, 0.19, 0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 1.0)] ==
      ["Very low estimated probability", "Very low estimated probability", "Low estimated probability", "Low estimated probability", "Moderate estimated probability",
       "Moderate estimated probability", "High estimated probability", "High estimated probability", "Very high estimated probability", "Very high estimated probability"])
pr = [predict(bundle, validate_inputs(r, meta)[0]) for r in EXAMPLES.values()]
check("the examples include both classifications at the 0.35 threshold", len({x.placed_at_operating for x in pr}) == 2)

# ------------------------------------------------------------------ 4. Phase 6 methodology cross-check (validation rows only)
print("4. Phase 6 explanation methodology (Phase 4 train-only LR_B on saved validation cases)")
p4 = joblib.load(ROOT / "models/phase4/LR_B.joblib")
val = pd.read_csv(ROOT / "data/processed/validation.csv")
expl6 = pd.read_csv(ROOT / "results/metrics/modeling/07_individual_explanations.csv")
for case, g in expl6.groupby("case", sort=False):
    row = val.iloc[int(g.val_row_id.iloc[0])]
    clean = {f: (int(row[f]) if f in meta["count"] else (float(row[f]) if f in meta["continuous"] else str(row[f]))) for f in FEATURES}
    e = explain(p4, clean).contributions.set_index("feature").contribution
    saved = g[g.feature.isin(FEATURES)].set_index("feature").contribution_log_odds
    diff = float((e.loc[saved.index] - saved).abs().max())
    check(f"{case}: app contributions reproduce the saved Phase 6 contributions", diff < 1e-9, f"max |diff| = {diff:.2e}")

# ------------------------------------------------------------------ 5. invalid input handling
print("5. input validation (nothing is clipped or silently corrected)")
base = EXAMPLES["typical profile"]
cases = {
    "missing value (None)": dict(base, CGPA=None), "NaN": dict(base, CGPA=float("nan")), "empty string": dict(base, PlacementTraining=""),
    "non-numeric text": dict(base, AptitudeTestScore="eighty"), "infinity": dict(base, CGPA=float("inf")),
    "CGPA above range": dict(base, CGPA=9.2), "CGPA below range": dict(base, CGPA=6.4), "aptitude above range": dict(base, AptitudeTestScore=91),
    "SSC below range": dict(base, SSC_Marks=54), "HSC above range": dict(base, HSC_Marks=89), "soft skills above range": dict(base, SoftSkillsRating=4.9),
    "unknown count level": dict(base, Projects=4), "non-integer count level": dict(base, Internships=1.5), "bool as a number": dict(base, Internships=True),
    "unexpected category": dict(base, PlacementTraining="Maybe"), "lower-case category": dict(base, ExtracurricularActivities="yes"),
    "unexpected extra field": dict(base, Gender="F"), "missing field": {k: v for k, v in base.items() if k != "HSC_Marks"},
}
for name, raw in cases.items():
    clean, errs = validate_inputs(raw, meta)
    check(f"rejected: {name}", len(errs) >= 1 and (name in ("unexpected extra field",) or len(clean) < 10), errs[0][:90])
edge_ok, errs = validate_inputs(dict(base, CGPA=6.5, AptitudeTestScore=90.0, Projects=3.0), meta)
check("range limits themselves and whole-number floats for counts are accepted", not errs and edge_ok["Projects"] == 3 and isinstance(edge_ok["Projects"], int))

# ------------------------------------------------------------------ 6. UI run (headless) with one complete prediction
print("6. complete prediction through the Streamlit UI (headless AppTest)")
from streamlit.testing.v1 import AppTest

PREDICT = lambda a: [b for b in a.button if b.label == "Predict Placement Probability"][0]
LOAD = lambda a: [b for b in a.button if b.label == "Load typical profile"][0]
CLEAR = lambda a: [b for b in a.button if b.label == "Clear profile"][0]
HERO = '<div class="hero-prob">'
shown_result = lambda a: any(HERO in m.value for m in a.markdown)
fields_state = lambda a: ([a.number_input(key=f).value for f in meta["continuous"]] + [a.selectbox(key=f).value for f in meta["count"]]
                          + [a.radio(key=f).value for f in meta["binary"]])
fill = lambda a, prof: [(a.number_input(key=f) if f in meta["continuous"] else a.selectbox(key=f) if f in meta["count"] else a.radio(key=f)).set_value(prof[f]) for f in FEATURES]
at = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
check("app starts without exception", not at.exception, str(list(at.exception)))
ui = EXAMPLES["strong profile"]
for f in FEATURES:
    key = f
    if f in meta["continuous"]:
        at.number_input(key=key).set_value(ui[f])
    elif f in meta["count"]:
        at.selectbox(key=key).set_value(ui[f])
    else:
        at.radio(key=key).set_value(ui[f])
PREDICT(at).click().run()
check("UI run completes without exception", not at.exception, str(list(at.exception)))
html = " ".join(m.value for m in at.markdown)
p_ui = predict(bundle, validate_inputs(ui, meta)[0]).probability
shown = f"{p_ui * 100:.1f}%"
check("UI shows the probability with one decimal", f'<div class="hero-prob">{shown}</div>' in html, shown)
check("UI shows the 0.35 and 0.50 classifications side by side", "Project threshold 0.35" in html and "Conventional threshold 0.50" in html)
check("UI result contains no 'None'/'nan' placeholder text", "None" not in html and not re.search(r"\bnan\b", html, re.I))
check("UI exposes the threshold ruler with a text alternative", 'role="img"' in html and "Project threshold at 35 percent" in html)
check("UI shows the factor chart and the disclaimer", "Factors that raise the model" in html and "not a guarantee of placement" in html)
for label, key, bad, msg in [("out-of-range CGPA", "CGPA", 9.5, "outside the range"), ("out-of-range aptitude", "AptitudeTestScore", 95, "outside the range"),
                              ("out-of-range HSC", "HSC_Marks", 40, "outside the range")]:
    at2 = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
    at2.number_input(key=key).set_value(bad)
    PREDICT(at2).click().run()
    check(f"UI blocks a prediction and explains why: {label}", any(msg in e.value for e in at2.error) and not any('<div class="hero-prob">' in m.value for m in at2.markdown), at2.error[0].value[:80].replace(chr(10), " ") if at2.error else "no error shown")
at3 = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
check("no result is shown before the button is pressed", not any('<div class="hero-prob">' in m.value for m in at3.markdown))
check("all ten fields start empty (nothing is assumed)", all(at3.number_input(key=f).value is None for f in meta["continuous"]) and all(at3.selectbox(key=f).value is None for f in meta["count"]) and all(at3.radio(key=f).value is None for f in meta["binary"]))
PREDICT(at3).click().run()
req = [e.value for e in at3.error]
check("pressing Predict with empty fields lists every missing value and makes no prediction", len(req) == 1 and req[0].count("a value is required") == 10 and not any('<div class="hero-prob">' in m.value for m in at3.markdown), f"{req[0].count('a value is required') if req else 0} fields flagged")
LOAD(at3).click().run()
typ = EXAMPLES["typical profile"]
check("'Load typical profile' fills the fields with the documented medians / most common levels", all(at3.number_input(key=f).value == typ[f] for f in meta["continuous"]) and all(at3.selectbox(key=f).value == typ[f] for f in meta["count"]) and all(at3.radio(key=f).value == typ[f] for f in meta["binary"]))
PREDICT(at3).click().run()
p_typ = predict(bundle, validate_inputs(typ, meta)[0]).probability
html3 = " ".join(m.value for m in at3.markdown)
check("typical profile through the UI: probability, 0.35 Placed vs 0.50 Not Placed", f'<div class="hero-prob">{p_typ * 100:.1f}%</div>' in html3 and 0.35 <= p_typ < 0.5, f"{p_typ:.4%}")

# ------------------------------------------------------------------ 6b. stale results, clear, repeat predictions (release-blocking behaviours)
print("6b. stale-result handling, Clear profile, repeated predictions (headless AppTest)")
at4 = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
fill(at4, EXAMPLES["strong profile"]); PREDICT(at4).click().run()
check("predict a valid profile: result shown", shown_result(at4) and not at4.exception)
at4.number_input(key="CGPA").set_value(7.0).run()
check("edit one field after a prediction: old result is removed", not shown_result(at4) and not at4.exception)
check("... and the user is told why and what to do", any("changed since the last prediction" in m.value for m in at4.markdown))
at4.number_input(key="CGPA").set_value(EXAMPLES["strong profile"]["CGPA"]).run()
check("restoring the original value does not resurrect the old result", not shown_result(at4))
for f in FEATURES:                                                    # every one of the ten inputs invalidates the result
    atx = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
    fill(atx, EXAMPLES["strong profile"]); PREDICT(atx).click().run()
    assert shown_result(atx)
    other = EXAMPLES["weak profile"][f]
    (atx.number_input(key=f) if f in meta["continuous"] else atx.selectbox(key=f) if f in meta["count"] else atx.radio(key=f)).set_value(other).run()
    check(f"editing {f} invalidates the result", not shown_result(atx) and not atx.exception)
PREDICT(at4).click().run()
check("predict again after the edit: a fresh result for the current inputs", shown_result(at4) and f'{HERO}{p_ui * 100:.1f}%</div>' in " ".join(m.value for m in at4.markdown))
fill(at4, EXAMPLES["weak profile"]); PREDICT(at4).click().run()
p_weak = predict(bundle, validate_inputs(EXAMPLES["weak profile"], meta)[0]).probability
check("second prediction in a row reflects the new profile only", f'{HERO}{p_weak * 100:.1f}%</div>' in " ".join(m.value for m in at4.markdown) and " ".join(m.value for m in at4.markdown).count(HERO) == 1)
CLEAR(at4).click().run()
check("Clear profile: all ten fields empty again", all(v is None for v in fields_state(at4)) and not at4.exception)
check("Clear profile: result removed and no stale notice", not shown_result(at4) and not any("changed since the last prediction" in m.value for m in at4.markdown))
PREDICT(at4).click().run()
check("Predict after Clear flags the ten missing values and makes no prediction", not shown_result(at4) and at4.error[0].value.count("a value is required") == 10)
fill(at4, EXAMPLES["strong profile"]); PREDICT(at4).click().run()
at4.number_input(key="SSC_Marks").set_value(60).run()
LOAD(at4).click().run()
check("edit then Load typical profile: fields filled and no stale result", all(at4.number_input(key=f).value == typ[f] for f in meta["continuous"]) and not shown_result(at4))
PREDICT(at4).click().run()
check("Load typical profile then Predict works", shown_result(at4) and f'{HERO}{p_typ * 100:.1f}%</div>' in " ".join(m.value for m in at4.markdown))
fill(at4, EXAMPLES["strong profile"]); at4.number_input(key="CGPA").set_value(9.5).run()
check("out-of-range value gets an inline message while typing", any("Out of range" in m.value for m in at4.markdown))
PREDICT(at4).click().run()
check("out-of-range value after an earlier prediction: blocked, no result", not shown_result(at4) and any("outside the range" in e.value for e in at4.error))

print("6c. prediction failure is reported safely")
import prediction as _pred_mod
_real_predict = _pred_mod.predict
_pred_mod.predict = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(r"C:\Users\secret\path\model.joblib exploded"))
atf = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
fill(atf, EXAMPLES["strong profile"]); PREDICT(atf).click().run()
_pred_mod.predict = _real_predict
msgs = " ".join(e.value for e in atf.error)
check("a prediction failure shows a generic message, no result, no exception text or path", "internal error" in msgs and "secret" not in msgs and "exploded" not in msgs and not shown_result(atf) and not atf.exception)

print("6d. model loading failures are sanitised")
_real_path, _real_meta = _pred_mod.MODEL_PATH, _pred_mod.METADATA_PATH
try:
    _pred_mod.MODEL_PATH = Path(tempfile.gettempdir()) / "definitely_missing_model_xyz.joblib"
    try:
        _pred_mod.load_bundle(); msg = ""
    except _pred_mod.ModelLoadError as e:
        msg = str(e)
    check("missing model artifact raises ModelLoadError without a file-system path", bool(msg) and ":\\" not in msg and "/tmp" not in msg and "definitely_missing" not in msg, msg[:80])
    _pred_mod.MODEL_PATH = _real_path
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "model_metadata.json"
        fake.write_text(json.dumps(dict(meta, sklearn_version_at_fit="0.0.1")), encoding="utf-8")
        _pred_mod.METADATA_PATH = fake
        try:
            _pred_mod.load_bundle(); msg = ""
        except _pred_mod.ModelLoadError as e:
            msg = str(e)
        check("scikit-learn version mismatch is reported clearly", "0.0.1" in msg and "pip install" in msg and td not in msg, msg[:90])
        fake.write_text("{not json", encoding="utf-8")
        try:
            _pred_mod.load_bundle(); msg = ""
        except _pred_mod.ModelLoadError as e:
            msg = str(e)
        check("unreadable metadata raises ModelLoadError without a path", bool(msg) and td not in msg)
finally:
    _pred_mod.MODEL_PATH, _pred_mod.METADATA_PATH = _real_path, _real_meta
check("the real model still loads after the failure tests", _pred_mod.load_bundle().metadata["artifact_sha256"] == meta["artifact_sha256"])

print("6e. result rendering around the thresholds (synthetic probabilities, no model)")
from prediction import Prediction
from ui_helpers import prob_text, result_html
for p in (0.0, 0.19996, 0.2, 0.3496, 0.3499999, 0.35, 0.3500001, 0.39996, 0.4, 0.42, 0.4999999, 0.5, 0.5000001, 0.59996, 0.6, 0.79996, 0.8, 0.9999999, 1.0):
    pr_ = Prediction(p, float(logit(min(max(p, 1e-12), 1 - 1e-12))), 0.35, 0.5, classify(p, 0.35), classify(p, 0.5), probability_band(p))
    left, right = result_html(pr_)
    txt = re.search(r'class="hero-prob">([^<]+)<', left).group(1)
    consistent = all((float(txt.rstrip("%")) / 100 >= t) == (p >= t) for t in (0.35, 0.5, 0.2, 0.4, 0.6, 0.8)) and probability_band(float(txt.rstrip("%")) / 100) == pr_.band
    pills = re.findall(r'pill-(yes|no)">', right)
    check(f"p = {p!r}: displayed '{txt}' is on the same side of both thresholds and every band edge; pills = {pills}", consistent and pills == [("yes" if p >= 0.35 else "no"), ("yes" if p >= 0.5 else "no")])
check("far from a threshold the display keeps one decimal", prob_text(0.3729, (0.35, 0.5)) == "37.3%" and prob_text(0.0, (0.35, 0.5)) == "0.0%")
from explainability import Explanation
from ui_helpers import factor_html
exp_t = explain(bundle.pipeline, validate_inputs(EXAMPLES["typical profile"], meta)[0])
fh = factor_html(exp_t)
c_t = exp_t.contributions.set_index("label").contribution
signs_ok = all((f'{c:+.2f}' in fh) for c in c_t if abs(c) >= 0.005)
check("factor values shown in the bars equal the contributions to 2 decimals, with their signs", signs_ok and "-0.00" not in fh and "+0.00" not in fh)
check("raising factors are listed under 'raise', lowering under 'lower'",
      all(fh.index(lab) < (fh.index("Factors that lower") if "Factors that lower" in fh else 10**9) for lab, c in c_t.items() if c >= 0.005)
      and all(fh.index(lab) > fh.index("Factors that lower") for lab, c in c_t.items() if c <= -0.005))

print("6f. documentation / metadata consistency (aggregate figures only)")
p7 = (ROOT / "src" / "phase7_findings.md").read_text(encoding="utf-8")
for t in ("0.50", "0.35"):
    row = re.search(rf"^\| {t} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", p7, re.M)
    m_ = meta["phase7_reported"]["threshold_metrics"][t]
    check(f"threshold {t}: accuracy/precision/recall/F1 in the app metadata equal the documented Phase 7 table",
          row is not None and [round(m_[k], 3) for k in ("accuracy", "precision", "recall", "f1")] == [float(x) for x in row.groups()])
rp = meta["phase7_reported"]
check("headline metrics equal the documented values (AUC 0.886, log loss 0.420, Brier 0.134, slope 1.05, ECE 0.016)",
      [round(rp["roc_auc"][0], 3), round(rp["log_loss"][0], 3), round(rp["brier"][0], 3), round(rp["calibration_slope"], 2), round(rp["ece"], 3)] == [0.886, 0.420, 0.134, 1.05, 0.016])
check("dataset facts: 9,912 modelling rows = 6,938 + 1,487 + 1,487; 8,425 development rows", meta["dataset"]["n_modeling"] == 6938 + 1487 + 1487 == 9912 and meta["n_fit"] == 6938 + 1487)

# ------------------------------------------------------------------ 7. audit
print("7. audit")
opened = sorted({os.path.basename(p) for p in _OPENED if p.lower().endswith((".csv", ".json", ".joblib"))})
check("no held-out test file was opened during verification", not any(any(tok in n.lower() for tok in _FORBIDDEN) for n in opened))
print("   data/model files opened:", ", ".join(opened))
print(f"\nALL {len(OK)} CHECKS PASSED")
