"""One-off helper: writes app/model_metadata.json from FROZEN artifacts and DEVELOPMENT data only.

Reads: models/phase7/LR_B_final.joblib (+ final_model_configs.json), models/phase4/model_configs.json, data/processed/train.csv and
validation.csv (supported input ranges / valid levels / level counts), results/metrics/01_descriptive_statistics.csv (Phase 1 cross-check),
results/metrics/modeling/07_threshold_selection.json (thresholds) and the AGGREGATE Phase 7 summary metrics (bootstrap CI / calibration CSVs).
It never reads the held-out test split or any per-student test prediction file. It trains nothing and writes nothing but the JSON.

Run from the project root:  python app/build_app_metadata.py
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phase7" / "LR_B_final.joblib"
OUT = ROOT / "app" / "model_metadata.json"

FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
CONT = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
BINARY = ["ExtracurricularActivities", "PlacementTraining"]


def main() -> None:
    p4 = json.loads((ROOT / "models" / "phase4" / "model_configs.json").read_text(encoding="utf-8"))
    assert p4["features"] == FEATURES and p4["models"]["LR_B"]["config"] == {"C": 1.0, "class_weight": None}
    p7 = json.loads((ROOT / "models" / "phase7" / "final_model_configs.json").read_text(encoding="utf-8"))
    assert p7["fitted_on"] == "train + validation" and p7["n"] == 8425 and p7["models"]["LR_B"]["config"] == p4["models"]["LR_B"]["config"]
    thr = json.loads((ROOT / "results" / "metrics" / "modeling" / "07_threshold_selection.json").read_text(encoding="utf-8"))
    assert thr["selected_threshold"] == 0.35 and p7["thresholds"] == [0.5, 0.35]

    pipe = joblib.load(MODEL_PATH)
    assert pipe.get_params()["model__C"] == 1.0 and pipe.get_params()["model__class_weight"] is None
    assert list(pipe.named_steps["prep"].named_transformers_["cnt"].drop) == [1, 3, 0]

    dev = pd.concat([pd.read_csv(ROOT / "data" / "processed" / "train.csv"), pd.read_csv(ROOT / "data" / "processed" / "validation.csv")], ignore_index=True)
    assert len(dev) == p7["n"] and dev[FEATURES].isna().sum().sum() == 0

    # Phase 1 documented ranges (all supplied rows) must agree with the development-data ranges
    ph1 = pd.read_csv(ROOT / "results" / "metrics" / "01_descriptive_statistics.csv", index_col=0)
    ranges = {}
    for c in CONT:
        lo, hi = float(dev[c].min()), float(dev[c].max())
        assert (lo, hi) == (float(ph1.loc[c, "min"]), float(ph1.loc[c, "max"])), f"{c}: development range differs from Phase 1"
        ranges[c] = {"min": lo, "max": hi, "median": float(dev[c].median()), "train_mean": float(pipe.named_steps["prep"].named_transformers_["num"].mean_[CONT.index(c)]),
                     "integer_valued": bool((dev[c] == dev[c].round()).all())}
    levels = {}
    for c in COUNT:
        vc = dev[c].astype(int).value_counts().sort_index()
        assert (int(vc.index.min()), int(vc.index.max())) == (int(ph1.loc[c, "min"]), int(ph1.loc[c, "max"]))
        levels[c] = {"levels": [int(i) for i in vc.index], "n_per_level": {str(int(i)): int(n) for i, n in vc.items()}, "default": int(dev[c].astype(int).median())}
    for c in BINARY:
        vc = dev[c].value_counts()
        assert set(vc.index) == {"Yes", "No"}
        levels[c] = {"levels": ["No", "Yes"], "n_per_level": {k: int(v) for k, v in vc.items()}, "default": str(vc.idxmax())}

    cdir = ROOT / "results" / "metrics" / "modeling"
    ci = pd.read_csv(cdir / "08_test_bootstrap_ci.csv"); ci = ci[(ci.model == "LR_B") & (ci.threshold.isna() | (ci.threshold.astype(str) == ""))].set_index("metric")
    cal = pd.read_csv(cdir / "08_test_calibration_metrics.csv").set_index("model").loc["LR_B"]
    cls = pd.read_csv(cdir / "08_test_bootstrap_ci.csv")
    cls = cls[(cls.model == "LR_B") & cls.threshold.notna()]
    threshold_metrics = {f"{t:.2f}": {m: float(cls[(cls.threshold == t) & (cls.metric == m)].estimate.iloc[0]) for m in ("accuracy", "precision", "recall", "f1", "specificity")}
                         for t in (0.35, 0.5)}
    meta = {
        "model_id": "LR_B", "model_name": "Logistic Regression B", "artifact": "models/phase7/LR_B_final.joblib", "artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "sklearn_version_at_fit": p7["sklearn_version"], "fitted_on": p7["fitted_on"], "n_fit": p7["n"], "config": p7["models"]["LR_B"]["config"],
        "features": FEATURES, "continuous": CONT, "count": COUNT, "binary": BINARY,
        "reference_levels": {"Internships": 1, "Projects": 3, "Workshops/Certifications": 0, "ExtracurricularActivities": "No", "PlacementTraining": "No"},
        "ranges": ranges, "levels": levels, "rare_level_threshold_n": 50,
        "thresholds": {"operating": thr["selected_threshold"], "reference": 0.5,
                       "operating_rule": "selected operating threshold under the F1-oriented criterion (Phase 6, predeclared validation-F1 rule); not claimed to be the best threshold"},
        "phase7_reported": {"source": "src/phase7_findings.md / results/metrics/modeling/08_test_bootstrap_ci.csv, 08_test_calibration_metrics.csv (aggregate metrics)", "n_test": 1487,
                            "roc_auc": [float(ci.loc["roc_auc", "estimate"]), float(ci.loc["roc_auc", "lower_95"]), float(ci.loc["roc_auc", "upper_95"])],
                            "log_loss": [float(ci.loc["log_loss", "estimate"]), float(ci.loc["log_loss", "lower_95"]), float(ci.loc["log_loss", "upper_95"])],
                            "brier": [float(ci.loc["brier", "estimate"]), float(ci.loc["brier", "lower_95"]), float(ci.loc["brier", "upper_95"])],
                            "calibration_slope": float(cal["cal_slope"]), "calibration_intercept": float(cal["cal_intercept"]), "ece": float(cal["ece_quantile"]),
                            "threshold_metrics": threshold_metrics},
        "dataset": {"source": "Hugging Face Krooz/Campus_Recruitment_CSV", "url": "https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV", "license": "CC0-1.0",
                    "n_raw": 10000, "n_modeling": 9912, "n_train": 6938, "n_validation": 1487, "n_test": 1487},
    }
    OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
