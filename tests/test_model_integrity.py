"""C. Model-integrity tests: production code never trains, refits, recalibrates or writes; the frozen parameters are the recorded ones."""
import ast
import hashlib
import json

import joblib
import numpy as np
import pytest

from conftest import APP, FROZEN_SHA256, MODEL, ROOT, TYP, sha256_of
from prediction import validate_inputs, predict

RUNTIME_MODULES = ["app.py", "prediction.py", "explainability.py", "scenarios.py", "report.py", "ui_helpers.py", "content.py"]
FORBIDDEN_CALLS = {"fit", "fit_transform", "partial_fit", "fit_predict", "dump", "to_csv", "to_pickle", "write_bytes", "write_text", "CalibratedClassifierCV",
                   "GridSearchCV", "RandomizedSearchCV", "cross_val_score"}


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            yield (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)), node.lineno


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_runtime_source_contains_no_training_or_writing_calls(module):
    bad = [(n, ln) for n, ln in _calls(APP / module) if n in FORBIDDEN_CALLS]
    assert not bad, f"{module}: {bad}"


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_runtime_source_does_not_open_files_for_writing(module):
    for node in ast.walk(ast.parse((APP / module).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
            assert not set("wax+") & set(str(mode)), f"{module}:{node.lineno}"


def test_runtime_source_does_not_import_training_machinery():
    for module in RUNTIME_MODULES:
        for node in ast.walk(ast.parse((APP / module).read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert "model_selection" not in n and "calibration" not in n and "LogisticRegression" not in n, f"{module}: {n}"


def _arm(monkeypatch, obj):
    """Make every fitting entry point on `obj` (and the objects inside it) raise."""
    def boom(*a, **k):
        raise AssertionError("a fit method was called in production code")
    for step in ([obj] + [s for _, s in obj.steps] + [t for _, t, _ in obj.named_steps["prep"].transformers_ if hasattr(t, "fit")]
                 + [obj.named_steps["prep"]]):
        for name in ("fit", "fit_transform", "partial_fit"):
            if hasattr(type(step), name):
                monkeypatch.setattr(type(step), name, boom, raising=False)


def test_prediction_path_never_calls_fit(bundle, monkeypatch):
    from explainability import explain
    from scenarios import evaluate_scenarios
    import pandas as pd
    _arm(monkeypatch, bundle.pipeline)
    clean, errs = validate_inputs(TYP, bundle.metadata)
    assert not errs
    pred = predict(bundle, clean)
    exp = explain(bundle.pipeline, clean)
    evaluate_scenarios(bundle, pd.DataFrame([{"Scenario": "A", **TYP}]))
    assert 0 <= pred.probability <= 1 and exp.logit_reconstructed == pytest.approx(pred.logit, abs=1e-9)
    with pytest.raises(AssertionError):                    # the guard itself works
        bundle.pipeline.fit(None, None)


def test_streamlit_run_never_calls_fit_and_does_not_change_the_artifact(monkeypatch):
    """Full app run (cold load, predict, scenarios, report) with every fit method armed, hashing the artifact before and after."""
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    import prediction
    before = sha256_of(MODEL)
    st.cache_resource.clear()
    bundle = prediction.load_bundle()
    _arm(monkeypatch, bundle.pipeline)
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()
    for f, v in TYP.items():
        w = at.number_input(key=f) if f in bundle.metadata["continuous"] else at.selectbox(key=f) if f in bundle.metadata["count"] else at.radio(key=f)
        w.set_value(v)
    [b for b in at.button if b.label == "Predict Placement Probability"][0].click().run()
    assert not at.exception and any('<div class="hero-prob">' in m.value for m in at.markdown)
    assert sha256_of(MODEL) == before == FROZEN_SHA256


def test_frozen_parameters_equal_the_recorded_snapshot(bundle, golden):
    lr = bundle.pipeline.named_steps["model"]
    pre = bundle.pipeline.named_steps["prep"]
    assert np.array_equal(lr.coef_[0], np.array(golden["coef"])) and float(lr.intercept_[0]) == golden["intercept"]
    assert list(lr.classes_) == golden["classes"]
    sc = pre.named_transformers_["num"]
    assert np.array_equal(sc.mean_, np.array(golden["scaler_mean"])) and np.array_equal(sc.scale_, np.array(golden["scaler_scale"]))
    assert list(pre.get_feature_names_out()) == golden["output_feature_names"]


def test_running_predictions_does_not_mutate_parameters(bundle, golden):
    lr = bundle.pipeline.named_steps["model"]
    before = (lr.coef_.copy(), lr.intercept_.copy(), bundle.pipeline.named_steps["prep"].named_transformers_["num"].mean_.copy())
    for row in golden["rows"]:
        predict(bundle, validate_inputs(row["profile"], bundle.metadata)[0])
    assert np.array_equal(before[0], lr.coef_) and np.array_equal(before[1], lr.intercept_)
    assert np.array_equal(before[2], bundle.pipeline.named_steps["prep"].named_transformers_["num"].mean_)


def test_model_artifacts_are_not_writable_targets_of_any_runtime_module():
    for module in RUNTIME_MODULES:
        txt = (APP / module).read_text(encoding="utf-8")
        assert "joblib.dump" not in txt and "pickle.dump" not in txt


def test_phase7_sibling_artifacts_and_configs_untouched():
    cfg = json.loads((ROOT / "models/phase7/final_model_configs.json").read_text(encoding="utf-8"))
    assert "LR_B" in json.dumps(cfg)
    assert sha256_of(MODEL) == FROZEN_SHA256
