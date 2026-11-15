"""B. Inference tests: validation, schema construction, probability extraction and regression parity with the recorded snapshot."""
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import MODEL
from explainability import explain
from prediction import BINARY, CONTINUOUS, COUNT, FEATURES, ModelBundle, predict, to_frame, validate_inputs


def direct_proba(raw):
    df = pd.DataFrame([raw], columns=FEATURES)
    for c in COUNT:
        df[c] = df[c].astype(int)
    pipe = joblib.load(MODEL)
    return float(pipe.predict_proba(df)[0, 1]), float(pipe.decision_function(df)[0])


def run(bundle, raw):
    clean, errs = validate_inputs(raw, bundle.metadata)
    assert not errs, errs
    return predict(bundle, clean)


def test_valid_representative_input(bundle, typ):
    pred = run(bundle, typ)
    assert 0.0 <= pred.probability <= 1.0 and pred.probability == pytest.approx(0.373, abs=1e-3)
    assert pred.placed_at_operating is True and pred.placed_at_reference is False      # inside the 0.35-0.50 disagreement zone


def test_application_equals_direct_frozen_pipeline(bundle, typ):
    p, z = direct_proba(typ)
    pred = run(bundle, typ)
    assert pred.probability == p and pred.logit == z


def test_probability_is_p_placed(bundle, typ):
    both = bundle.pipeline.predict_proba(to_frame(validate_inputs(typ, bundle.metadata)[0]))[0]
    assert run(bundle, typ).probability == both[list(bundle.pipeline.classes_).index(1)]
    assert both.sum() == pytest.approx(1.0, abs=1e-12) and run(bundle, typ).probability == pytest.approx(1 - both[0], abs=1e-12)


def test_golden_snapshot_parity(bundle, golden):
    """Recorded outputs of the frozen artifact on 12 fixed synthetic profiles (tests/fixtures/golden_snapshot.json)."""
    assert golden["artifact_sha256"] == bundle.metadata["artifact_sha256"]
    assert len(golden["rows"]) == 12
    for row in golden["rows"]:
        pred = run(bundle, row["profile"])
        assert pred.probability == pytest.approx(row["probability"], abs=1e-12), row["name"]
        assert pred.logit == pytest.approx(row["log_odds"], abs=1e-12), row["name"]


def test_repeated_identical_inputs_are_deterministic(bundle, typ):
    assert len({run(bundle, dict(typ)).probability for _ in range(25)}) == 1


def test_probability_range_and_logistic_link(bundle, golden):
    for row in golden["rows"]:
        pred = run(bundle, row["profile"])
        assert 0.0 <= pred.probability <= 1.0
        assert pred.probability == pytest.approx(1 / (1 + math.exp(-pred.logit)), abs=1e-12)


@pytest.mark.parametrize("field", CONTINUOUS)
def test_numeric_boundaries_inclusive(bundle, typ, field):
    lo, hi = bundle.metadata["ranges"][field]["min"], bundle.metadata["ranges"][field]["max"]
    for v in (lo, hi):
        assert 0 <= run(bundle, dict(typ, **{field: v})).probability <= 1
    for v in (lo - 0.01, hi + 0.01):
        clean, errs = validate_inputs(dict(typ, **{field: v}), bundle.metadata)
        assert errs and "outside the range" in errs[0] and field not in clean


@pytest.mark.parametrize("field", COUNT + BINARY)
def test_every_documented_categorical_level_predicts(bundle, typ, field):
    for lv in bundle.metadata["levels"][field]["levels"]:
        assert 0 <= run(bundle, dict(typ, **{field: lv})).probability <= 1


@pytest.mark.parametrize("field", FEATURES)
@pytest.mark.parametrize("bad", [None, float("nan"), "", "   "])
def test_missing_values_rejected(bundle, typ, field, bad):
    clean, errs = validate_inputs(dict(typ, **{field: bad}), bundle.metadata)
    assert errs and field not in clean


@pytest.mark.parametrize("field", FEATURES)
def test_missing_column_rejected(bundle, typ, field):
    clean, errs = validate_inputs({k: v for k, v in typ.items() if k != field}, bundle.metadata)
    assert errs and field not in clean


def test_extra_column_rejected_not_ignored(bundle, typ):
    _, errs = validate_inputs(dict(typ, Gender="F"), bundle.metadata)
    assert any("Unexpected input field" in e for e in errs)


@pytest.mark.parametrize("field,bad", [("CGPA", "7.7"), ("CGPA", [7.7]), ("CGPA", True), ("Projects", "2"), ("Projects", 2.5), ("Projects", 9),
                                         ("Internships", True), ("PlacementTraining", 1), ("PlacementTraining", "yes"),
                                         ("ExtracurricularActivities", "Maybe"), ("HSC_Marks", "abc")])
def test_wrong_types_and_unknown_categories_rejected(bundle, typ, field, bad):
    clean, errs = validate_inputs(dict(typ, **{field: bad}), bundle.metadata)
    assert errs and field not in clean


@pytest.mark.parametrize("field", CONTINUOUS + COUNT)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_infinite_values_rejected(bundle, typ, field, bad):
    _, errs = validate_inputs(dict(typ, **{field: bad}), bundle.metadata)
    assert errs


def test_frame_has_exact_schema_order_and_dtypes(bundle, typ):
    clean, errs = validate_inputs(dict(reversed(list(typ.items()))), bundle.metadata)     # input order must not matter
    assert not errs
    df = to_frame(clean)
    assert list(df.columns) == FEATURES and len(df) == 1
    assert all(pd.api.types.is_integer_dtype(df[c]) for c in COUNT)
    assert run(bundle, dict(reversed(list(typ.items())))).probability == run(bundle, typ).probability


def test_pipeline_itself_refuses_unknown_levels(bundle, typ):
    """Defence in depth: even if validation were bypassed, the frozen OneHotEncoder(handle_unknown='error') raises rather than guessing."""
    df = to_frame(dict(typ))
    df.loc[0, "Projects"] = 7
    with pytest.raises(ValueError):
        bundle.pipeline.predict_proba(df)


def test_explanation_reconstructs_the_frozen_logit(bundle, golden):
    for row in golden["rows"]:
        clean, _ = validate_inputs(row["profile"], bundle.metadata)
        ex = explain(bundle.pipeline, clean)
        assert ex.logit_reconstructed == pytest.approx(row["log_odds"], abs=1e-9)
        assert set(ex.contributions["feature"]) == set(FEATURES)


def test_scenario_path_uses_the_same_prediction_function(bundle, typ):
    from scenarios import evaluate_scenarios
    tbl = pd.DataFrame([{"Scenario": "A", **typ}, {"Scenario": "B", **dict(typ, CGPA=9.0)}])
    a, b = evaluate_scenarios(bundle, tbl)
    assert a.pred.probability == run(bundle, typ).probability and b.pred.probability == run(bundle, dict(typ, CGPA=9.0)).probability


class _Boom:
    classes_ = [0, 1]

    def predict_proba(self, X):
        raise RuntimeError("boom")

    def decision_function(self, X):
        raise RuntimeError("boom")


class _BadProba:
    classes_ = [0, 1]

    def predict_proba(self, X):
        return np.array([[0.2, 1.4]])

    def decision_function(self, X):
        return np.array([0.1])


class _Flipped:
    classes_ = [1, 0]

    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])

    def decision_function(self, X):
        return np.array([0.0])


def test_no_hardcoded_probability_when_inference_fails(bundle, typ):
    clean, _ = validate_inputs(typ, bundle.metadata)
    with pytest.raises(RuntimeError):
        predict(ModelBundle(pipeline=_Boom(), metadata=bundle.metadata), clean)


def test_invalid_probability_is_rejected(bundle, typ):
    clean, _ = validate_inputs(typ, bundle.metadata)
    with pytest.raises(RuntimeError):
        predict(ModelBundle(pipeline=_BadProba(), metadata=bundle.metadata), clean)


def test_positive_class_is_selected_by_label_not_position(bundle, typ):
    clean, _ = validate_inputs(typ, bundle.metadata)
    assert predict(ModelBundle(pipeline=_Flipped(), metadata=bundle.metadata), clean).probability == 0.7
