"""Product-layer verification (scenario comparison, downloadable report, plain-language content, randomised consistency, deployment files).
Run from the project root:  python app/verify_product.py      (complements app/verify_app.py, which stays unchanged)

Only synthetic profiles are used (random draws inside the documented ranges). The held-out test split is never opened: an audit hook raises if a file
named like it is opened.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP))

_FORBIDDEN = ("test" + ".csv", "08_test" + "_predictions")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if any(tok in os.path.basename(os.fsdecode(args[0])).lower() for tok in _FORBIDDEN):
            raise PermissionError("verification must not touch the held-out test files")


sys.addaudithook(_audit)

import joblib
import numpy as np
import pandas as pd

from content import FIELD_HELP, GROUPS
from explainability import explain
from prediction import BINARY, CONTINUOUS, COUNT, FEATURES, LABELS, load_bundle, predict, validate_inputs
from report import build_report_html
from scenarios import MAX_SCENARIOS, changed_fields, evaluate_scenarios
from ui_helpers import BAND_EDGES, interpretation_html, prob_text

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAILED: {name} {detail}"
    OK.append(name)
    print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))


bundle = load_bundle()
meta = bundle.metadata
direct = joblib.load(ROOT / "models/phase7/LR_B_final.joblib")

print("1. frozen artifact")
sha = hashlib.sha256((ROOT / "models/phase7/LR_B_final.joblib").read_bytes()).hexdigest()
check("LR_B_final.joblib SHA-256 equals the recorded value", sha == meta["artifact_sha256"] == "58e8b04e9666b13be47e0cb5c4850707312976776ab5bdfbaa436874ca3cb0eb", sha[:16])

print("2. plain-language content covers every feature, every group")
check("every one of the ten features has a description", set(FIELD_HELP) == set(FEATURES) and all(len(v) > 20 for v in FIELD_HELP.values()))
check("the three groups partition the ten features (Academic 3, Technical 4, Employability 3)",
      sorted(f for _, fs in GROUPS.values() for f in fs) == sorted(FEATURES) and [len(v[1]) for v in GROUPS.values()] == [3, 4, 3])

print("3. 300 random synthetic profiles: application == direct pipeline, log-odds reconstruction < 1e-9")
rng = np.random.default_rng(20261114)
worst_p, worst_z = 0.0, 0.0
for _ in range(300):
    raw = {}
    for f in CONTINUOUS:
        lo, hi = meta["ranges"][f]["min"], meta["ranges"][f]["max"]
        raw[f] = float(rng.uniform(lo, hi))
    for f in COUNT:
        raw[f] = int(rng.choice(meta["levels"][f]["levels"]))
    for f in BINARY:
        raw[f] = str(rng.choice(meta["levels"][f]["levels"]))
    clean, errs = validate_inputs(raw, meta)
    assert not errs, errs
    pr = predict(bundle, clean)
    df = pd.DataFrame([raw])[FEATURES]
    for c in COUNT:
        df[c] = df[c].astype(int)
    worst_p = max(worst_p, abs(pr.probability - float(direct.predict_proba(df)[0, 1])))
    ex = explain(bundle.pipeline, clean)
    worst_z = max(worst_z, abs(ex.logit_reconstructed - float(direct.decision_function(df)[0])))
    assert pr.placed_at_operating == (pr.probability >= 0.35) and pr.placed_at_reference == (pr.probability >= 0.5)
check("max |p_app - p_direct| over 300 profiles < 1e-12", worst_p < 1e-12, f"{worst_p:.1e}")
check("max |log-odds reconstruction error| over 300 profiles < 1e-9", worst_z < 1e-9, f"{worst_z:.1e}")
for f in CONTINUOUS:  # inclusive boundaries
    lo, hi = meta["ranges"][f]["min"], meta["ranges"][f]["max"]
    base = {"CGPA": 7.7, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 80, "SoftSkillsRating": 4.4,
            "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 70, "HSC_Marks": 73}
    ok_lo = not validate_inputs(dict(base, **{f: lo}), meta)[1]
    ok_hi = not validate_inputs(dict(base, **{f: hi}), meta)[1]
    bad_lo = bool(validate_inputs(dict(base, **{f: lo - 0.01}), meta)[1])
    bad_hi = bool(validate_inputs(dict(base, **{f: hi + 0.01}), meta)[1])
    check(f"{f}: both limits accepted, just outside both rejected", ok_lo and ok_hi and bad_lo and bad_hi)

TYP = {"CGPA": 7.7, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 80, "SoftSkillsRating": 4.4,
       "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 70, "HSC_Marks": 73}
clean_t, _ = validate_inputs(TYP, meta)
pred_t = predict(bundle, clean_t)
exp_t = explain(bundle.pipeline, clean_t)

print("4. scenario comparison")
tbl = pd.DataFrame([{"Scenario": "A", **TYP}, {"Scenario": "B", **dict(TYP, CGPA=8.9, PlacementTraining="No")}, {"Scenario": "bad", **dict(TYP, CGPA=9.9, Projects=7)},
                    {"Scenario": "missing", **dict(TYP, SSC_Marks=np.nan)}])
tbl["Projects"] = tbl["Projects"].astype(float)          # data_editor round-trips ints through float/NaN; the validator must still accept whole numbers
res = evaluate_scenarios(bundle, tbl)
check("valid scenarios are evaluated, invalid ones reported and not predicted", [r.pred is not None for r in res] == [True, True, False, False] and len(res[2].errors) == 2 and len(res[3].errors) == 1)
check("scenario A (unchanged profile) equals the main prediction exactly", res[0].pred.probability == pred_t.probability and changed_fields(clean_t, res[0].clean) == [])
dfb = pd.DataFrame([dict(TYP, CGPA=8.9, PlacementTraining="No")])[FEATURES]
for c in COUNT:
    dfb[c] = dfb[c].astype(int)
check("scenario B probability equals the direct frozen-pipeline probability (< 1e-12)", abs(res[1].pred.probability - float(direct.predict_proba(dfb)[0, 1])) < 1e-12)
check("changed fields are detected", changed_fields(clean_t, res[1].clean) == ["CGPA", "PlacementTraining"])
check("scenario count is capped at a small constant", 1 < MAX_SCENARIOS <= 10)

print("5. downloadable report")
fixed = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)
html = build_report_html(clean_t, pred_t, exp_t, meta, fixed)
shown = prob_text(pred_t.probability, (0.35, 0.5) + BAND_EDGES)
check("report shows the same displayed probability as the screen", f'<div class="big">{shown}</div>' in html, shown)
check("report states both classifications and both thresholds", "0.35 (project threshold)" in html and "0.50 (conventional threshold)" in html
      and html.count("<b>Placed</b>") + html.count("<b>Not Placed</b>") == 2)
check("report has a timestamp", "2026-11-14 09:30 UTC" in html)
check("report lists all ten inputs with their values", all(LABELS[f] in html for f in FEATURES) and "7.7" in html and "<td class='num'>Yes</td>" in html)
tot = float(re.search(r"total log-odds = ([+-]\d+\.\d+)", html).group(1))
check("report contributions are the app's contributions (total log-odds matches to 3 dp)", abs(tot - exp_t.logit_reconstructed) < 5e-4)
check("report includes limitations, association-not-causation and no-guarantee language", all(t in html for t in ("not causal effects", "not a guarantee", "not a recruitment decision", "rule-based or constructed")))
check("report loads nothing external (no scripts, links, images)", not re.search(r"<script|<link|<img|https?://(?!huggingface)", html))
evil = dict(meta, model_name="<script>alert(1)</script>", dataset=dict(meta["dataset"], source='"><img src=x onerror=alert(1)>'))
h2 = build_report_html(clean_t, pred_t, exp_t, evil, fixed)
check("report HTML-escapes injected strings", "<script>alert" not in h2 and "<img src=x" not in h2 and "&lt;script&gt;" in h2)

print("6. interpretation text")
from prediction import Prediction
mk = lambda p: Prediction(p, 0.0, 0.35, 0.5, p >= 0.35, p >= 0.5, "band")
t_between, t_hi, t_lo = interpretation_html(mk(0.42)), interpretation_html(mk(0.9)), interpretation_html(mk(0.1))
check("between thresholds the two labels are explained as expected, not contradictory", "disagree" in t_between and "<b>Placed</b> at 0.35" in t_between and "<b>Not Placed</b> at 0.50" in t_between)
check("both thresholds agree: one statement", "Both cut-offs" in t_hi and "Both cut-offs" in t_lo and "disagree" not in t_hi)
check("no guarantee language / certainty claims", not re.search(r"guarantee(s|d)? (that )?you|will be placed|will not be placed|certain", t_hi + t_lo + t_between, re.I))

print("7. UI: tabs, download button, scenario editor (headless AppTest)")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
check("landing: intro steps, limits expander and no result before predicting", any("steps" in m.value for m in at.markdown) and any("Before you start" in e.label for e in at.expander) and not at.get("download_button"))
check("landing: every input shows a plain-language description", all(FIELD_HELP[f] in " ".join(c.value for c in at.caption) for f in FEATURES))
for f in FEATURES:
    (at.number_input(key=f) if f in CONTINUOUS else at.selectbox(key=f) if f in COUNT else at.radio(key=f)).set_value(TYP[f])
[b for b in at.button if b.label == "Predict Placement Probability"][0].click().run()
check("result renders without exception", not at.exception and any('<div class="hero-prob">' in m.value for m in at.markdown))
check("three exploration tabs", [t.label for t in at.tabs] == ["Why this estimate", "Compare scenarios", "Profile and report"])
check("download button is present", len(at.get("download_button")) == 1)
check("interpretation text is shown", any("What this means" in m.value for m in at.markdown))
check("hypothetical-scenario disclaimer is shown", any("hypothetical model scenarios" in m.value for m in at.markdown))
check("scenario comparison table lists the entered profile", any("Your profile (as entered)" in str(d.value) for d in at.dataframe))
[b for b in at.button if b.label == "Clear profile"][0].click().run()
check("Clear profile removes the result, the tabs, the report button and the scenarios", not at.get("download_button") and not at.tabs and not at.exception)

print("8. deployment files")
cfg = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
check("config does not bind the server to localhost (hosting-neutral)", not re.search(r"^\s*address\s*=", cfg, re.M))
req = (APP / "requirements.txt").read_text(encoding="utf-8")
pins = dict(l.split("==") for l in req.splitlines() if "==" in l and not l.startswith("#"))
check("app/requirements.txt pins the lean runtime set incl. scikit-learn==1.9.1", set(pins) >= {"streamlit", "scikit-learn", "pandas", "numpy", "scipy", "joblib"} and pins["scikit-learn"] == meta["sklearn_version_at_fit"])
check("no research-only packages in the deployment requirements", not re.search(r"jupyter|statsmodels|matplotlib|seaborn|nbconvert", req))
print(f"\nALL {len(OK)} PRODUCT CHECKS PASSED")
