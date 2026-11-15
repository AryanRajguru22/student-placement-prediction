"""A. Artifact and data-integrity tests: the frozen LR_B artifact, its pipeline, its schema, and the raw data."""
import hashlib
import json
import os
import re
import subprocess

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from conftest import APP, FROZEN_SHA256, MODEL, ROOT, sha256_of

FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
protocol = json.loads((ROOT / "results/metrics/modeling/08_frozen_protocol.json").read_text(encoding="utf-8"))
recorded = protocol["artifact_sha256_at_start"]


def test_frozen_artifact_exists_and_hash_matches_everywhere(meta):
    assert MODEL.is_file()
    assert sha256_of(MODEL) == FROZEN_SHA256 == meta["artifact_sha256"]


def test_committed_blob_equals_working_copy():
    """What Streamlit Cloud clones is the git blob, not the working file: line-ending conversion must not touch the joblib."""
    out = subprocess.run(["git", "show", "HEAD:models/phase7/LR_B_final.joblib"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip("artifact is not committed or git is unavailable")
    assert hashlib.sha256(out.stdout).hexdigest() == FROZEN_SHA256


def test_artifact_loads_and_is_the_documented_pipeline(bundle):
    pipe = bundle.pipeline
    assert list(pipe.named_steps) == ["prep", "model"]
    assert type(pipe.named_steps["model"]).__name__ == "LogisticRegression"
    assert pipe.named_steps["model"].C == 1.0 and pipe.named_steps["model"].class_weight is None
    assert list(pipe.feature_names_in_) == FEATURES


def test_class_labels(bundle):
    assert list(bundle.pipeline.classes_) == [0, 1]


def test_positive_class_is_placed_not_notplaced(bundle):
    """1 = Placed. Evidence from DEVELOPMENT data only (train.csv): the frozen model ranks placed students above not-placed ones (AUC well above
    0.5), which is only possible if the probability column the app shows is P(Placed). A direction check, not a performance claim."""
    from sklearn.metrics import roc_auc_score
    tr = pd.read_csv(ROOT / "data/raw/train.csv").iloc[::25]
    y = (tr["PlacementStatus"] == "Placed").astype(int)
    p = bundle.pipeline.predict_proba(tr[FEATURES])[:, 1]
    auc = roc_auc_score(y, p)
    assert auc > 0.8, f"AUC {auc:.3f}: the probability column does not behave like P(Placed)"


def test_target_encoding_in_training_code_is_placed_equals_one():
    src = (ROOT / "src/build_notebook_05.py").read_text(encoding="utf-8")
    assert '(train[TARGET] == "Placed").astype(int)' in src


def test_expected_feature_schema_and_encoders(bundle, meta):
    pre = bundle.pipeline.named_steps["prep"]
    assert meta["features"] == FEATURES
    kinds = {name: cols for name, _, cols in pre.transformers_ if name != "remainder"}
    assert kinds["num"] == meta["continuous"] and kinds["cnt"] == meta["count"] and kinds["bin"] == meta["binary"]
    assert pre.named_transformers_["cnt"].handle_unknown == "error" and pre.named_transformers_["bin"].handle_unknown == "error"
    assert list(pre.named_transformers_["cnt"].drop) == [meta["reference_levels"][f] for f in meta["count"]]


def test_dependencies_match_the_fitted_environment(meta):
    assert sklearn.__version__ == meta["sklearn_version_at_fit"] == "1.9.1"
    pins = {}
    for line in (APP / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, ver = line.strip().split("==")
            pins[name] = ver
    assert pins["scikit-learn"] == sklearn.__version__
    assert pins["joblib"] == joblib.__version__ and pins["numpy"] == np.__version__ and pins["pandas"] == pd.__version__


def test_metadata_ranges_and_levels_cover_the_schema(meta):
    assert set(meta["ranges"]) == set(meta["continuous"]) and set(meta["levels"]) == set(meta["count"] + meta["binary"])
    for f in meta["continuous"]:
        assert meta["ranges"][f]["min"] < meta["ranges"][f]["max"]


def test_app_code_has_no_absolute_local_paths():
    for py in [q for q in APP.glob("*.py") if not q.name.startswith("verify_")]:      # the verify scripts contain a deliberate fake path
        txt = py.read_text(encoding="utf-8")
        assert not re.search(r"\b[A-Za-z]:[\\/]|/Users/|/home/", txt), f"absolute path in {py.name}"


# ------------------------------------------------------------------ raw data and earlier artifacts
@pytest.mark.parametrize("name,key", [("train", "raw_train"), ("validation", "raw_validation")])
def test_raw_train_and_validation_are_unchanged(name, key):
    assert sha256_of(ROOT / f"data/raw/{name}.csv") == recorded[key]


def test_raw_row_counts():
    assert len(pd.read_csv(ROOT / "data/raw/train.csv")) == 7225 and len(pd.read_csv(ROOT / "data/raw/validation.csv")) == 1275


def test_held_out_test_file_unchanged_without_reading_it():
    """The held-out file is NOT opened (project rule). Only its size is compared with the size recorded when Phase 7 opened it."""
    path = ROOT / "data/raw/test.csv"
    if not path.exists():
        pytest.skip("test.csv is not part of the public repository")
    assert os.stat(path).st_size == protocol["test_file_stat_at_start"]["raw_test"]["size"]


def test_phase4_comparison_artifact_unchanged():
    assert sha256_of(ROOT / "models/phase4/LR_B.joblib") == recorded["LR_B.joblib"]
