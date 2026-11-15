"""One-off: record the frozen LR_B_final.joblib outputs on fixed synthetic profiles (a regression snapshot, NOT a performance evaluation).

Refuses to overwrite an existing snapshot unless --force is given. Never opens any dataset. Run from the project root.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "golden_snapshot.json"
if OUT.exists() and "--force" not in sys.argv:
    sys.exit(f"{OUT} exists; pass --force to overwrite")

sys.path.insert(0, str(ROOT / "app"))
from prediction import BINARY, CONTINUOUS, COUNT, FEATURES  # noqa: E402

meta = json.loads((ROOT / "app" / "model_metadata.json").read_text(encoding="utf-8"))
model_path = ROOT / "models" / "phase7" / "LR_B_final.joblib"
pipe = joblib.load(model_path)
R, L = meta["ranges"], meta["levels"]
profiles = {
    "typical_example": {"CGPA": 7.7, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 80, "SoftSkillsRating": 4.4,
                        "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 70, "HSC_Marks": 73},
    "all_minimum": {**{f: R[f]["min"] for f in CONTINUOUS}, **{f: min(L[f]["levels"]) for f in COUNT}, **{f: "No" for f in BINARY}},
    "all_maximum": {**{f: R[f]["max"] for f in CONTINUOUS}, **{f: max(L[f]["levels"]) for f in COUNT}, **{f: "Yes" for f in BINARY}},
}
rng = np.random.default_rng(20261115)
for i in range(9):
    p = {f: round(float(rng.uniform(R[f]["min"], R[f]["max"])), 2) for f in CONTINUOUS}
    p.update({f: int(rng.choice(L[f]["levels"])) for f in COUNT})
    p.update({f: str(rng.choice(L[f]["levels"])) for f in BINARY})
    profiles[f"random_{i + 1}"] = p
rows = []
for name, p in profiles.items():
    df = pd.DataFrame([p], columns=FEATURES)
    for c in COUNT:
        df[c] = df[c].astype(int)
    rows.append({"name": name, "profile": p, "probability": float(pipe.predict_proba(df)[0, 1]), "log_odds": float(pipe.decision_function(df)[0])})
lr = pipe.named_steps["model"]
snapshot = {
    "note": "Regression snapshot of the frozen Phase 7 LR_B artifact on fixed synthetic profiles. Not an evaluation; no dataset was read to make it.",
    "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "coef": [float(x) for x in lr.coef_[0]], "intercept": float(lr.intercept_[0]), "classes": [int(c) for c in lr.classes_],
    "scaler_mean": [float(x) for x in pipe.named_steps["prep"].named_transformers_["num"].mean_],
    "scaler_scale": [float(x) for x in pipe.named_steps["prep"].named_transformers_["num"].scale_],
    "output_feature_names": list(pipe.named_steps["prep"].get_feature_names_out()),
    "rows": rows,
}
OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
print("wrote", OUT, len(rows), "profiles")
