"""E. Streamlit tests (headless AppTest): start-up, controls, valid submission, ruler, explainability, invalid input, load failure, stale results."""
import pytest
from streamlit.testing.v1 import AppTest

import streamlit as st

from conftest import APP, TYP
from prediction import FEATURES, validate_inputs, predict, load_bundle

HERO = '<div class="hero-prob">'
CONT = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
CNT = ["Internships", "Projects", "Workshops/Certifications"]
WEAK = {"CGPA": 6.6, "Internships": 0, "Projects": 1, "Workshops/Certifications": 3, "AptitudeTestScore": 62, "SoftSkillsRating": 3.2,
        "ExtracurricularActivities": "No", "PlacementTraining": "No", "SSC_Marks": 57, "HSC_Marks": 60}


def new_app():
    return AppTest.from_file(str(APP / "app.py"), default_timeout=90).run()


def widget(at, f):
    return at.number_input(key=f) if f in CONT else at.selectbox(key=f) if f in CNT else at.radio(key=f)


def fill(at, prof):
    for f in FEATURES:
        widget(at, f).set_value(prof[f])


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def html(at):
    return " ".join(m.value for m in at.markdown)


def shown(at):
    return HERO in html(at)


def expected_pct(prof):
    b = load_bundle()
    return predict(b, validate_inputs(prof, b.metadata)[0]).probability * 100


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield


def test_initialisation_has_all_ten_inputs_empty_and_no_result():
    at = new_app()
    assert not at.exception
    assert len(at.number_input) == 5 and len(at.selectbox) == 3 and len(at.radio) == 2
    assert [w.key for w in list(at.number_input) + list(at.selectbox) + list(at.radio)].sort() == sorted(FEATURES).sort()
    assert all(widget(at, f).value is None for f in FEATURES)
    assert not shown(at)
    assert [b.label for b in at.button][:3] == ["Predict Placement Probability", "Load typical profile", "Clear profile"]


def test_no_slider_and_only_schema_fields():
    at = new_app()
    assert len(at.slider) == 0
    assert {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.radio)} == set(FEATURES)


def test_valid_submission_shows_the_exact_model_probability_and_ruler():
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    assert not at.exception
    p = expected_pct(TYP)
    h = html(at)
    assert f"{HERO}{p:.1f}%</div>" in h                                       # the displayed number is the frozen model's P(Placed)
    assert f'style="width:{p:.3f}%"' in h                                      # ruler fill is the same probability
    assert 'class="gauge-zone" style="left:35.0%;width:15.0%"' in h            # disagreement zone 0.35-0.50
    assert 'left:35.0%">0.35' in h and 'left:50.0%">0.50' in h
    assert "not a guarantee" in h
    assert "Project threshold 0.35" in h and "Conventional threshold 0.50" in h


def test_ruler_css_navy_fill_and_animation_at_most_450ms():
    from ui_helpers import CSS
    assert "background: #0F4C75" in CSS.split(".gauge-fill")[1].split("}")[0].replace("{{", "{").replace("}}", "}") or "0F4C75" in CSS.upper()
    assert "animation: gauge-grow 450ms" in CSS and "prefers-reduced-motion" in CSS


def test_explainability_is_present_and_association_not_causation():
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    h = html(at)
    assert "Key factors influencing this estimate" in h and "Factors that raise the model" in h
    assert [t.label for t in at.tabs] == ["Why this estimate", "Compare scenarios", "Profile and report"]
    assert "cause" in h.lower() or "association" in h.lower()
    assert "guarantee" in h.lower()


@pytest.mark.parametrize("field,bad", [("CGPA", 9.5), ("CGPA", 3.0), ("AptitudeTestScore", 95), ("HSC_Marks", 40), ("SoftSkillsRating", 9.0)])
def test_out_of_range_blocks_the_model(field, bad):
    at = new_app()
    fill(at, dict(TYP, **{field: bad}))
    button(at, "Predict Placement Probability").click().run()
    assert not shown(at) and any("outside the range" in e.value for e in at.error)


def test_empty_submission_is_rejected_listing_every_field():
    at = new_app()
    button(at, "Predict Placement Probability").click().run()
    assert not shown(at) and at.error[0].value.count("a value is required") == 10


def test_model_load_failure_fails_safe_without_a_fake_prediction(monkeypatch, tmp_path):
    import prediction
    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "missing_model.joblib")
    at = new_app()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)
    assert not shown(at) and len(at.number_input) == 0                         # the app stops: no form that could return a number
    assert str(tmp_path) not in " ".join(e.value for e in at.error)            # no file-system paths leaked


def test_tampered_or_wrong_artifact_is_refused(monkeypatch, tmp_path):
    import prediction
    fake = tmp_path / "LR_B_final.joblib"
    fake.write_bytes((APP.parent / "models/phase4/LR_A.joblib").read_bytes())
    monkeypatch.setattr(prediction, "MODEL_PATH", fake)
    at = new_app()
    assert any("fingerprint" in e.value for e in at.error) and not shown(at)


def test_stale_result_is_removed_on_every_single_input_change():
    for f in FEATURES:
        at = new_app()
        fill(at, TYP)
        button(at, "Predict Placement Probability").click().run()
        assert shown(at)
        widget(at, f).set_value(WEAK[f]).run()
        assert not shown(at), f
        assert "changed since the last prediction" in html(at)


def test_recalculate_uses_current_values_and_repeats_correctly():
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    button(at, "Predict Placement Probability").click().run()                    # repeated identical submit
    assert html(at).count(HERO) == 1 and f"{HERO}{expected_pct(TYP):.1f}%" in html(at)
    fill(at, WEAK)
    button(at, "Predict Placement Probability").click().run()
    assert html(at).count(HERO) == 1 and f"{HERO}{expected_pct(WEAK):.1f}%" in html(at)
    assert f"{HERO}{expected_pct(TYP):.1f}%" not in html(at)


def test_error_after_a_result_leaves_no_old_result_visible():
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    widget(at, "CGPA").set_value(9.9).run()
    button(at, "Predict Placement Probability").click().run()
    assert not shown(at) and at.error


def test_internal_failure_shows_generic_error_and_no_number(monkeypatch):
    import prediction
    at = new_app()
    fill(at, TYP)
    monkeypatch.setattr(prediction, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret detail 7.7")))
    button(at, "Predict Placement Probability").click().run()
    assert not shown(at) and any("internal error" in e.value for e in at.error)
    assert "secret detail" not in " ".join(e.value for e in at.error)


def test_clear_and_load_typical_reset_state():
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    button(at, "Clear profile").click().run()
    assert not shown(at) and all(widget(at, f).value is None for f in FEATURES)
    button(at, "Load typical profile").click().run()
    assert all(widget(at, f).value is not None for f in FEATURES) and not shown(at)


def test_scenario_table_does_not_replay_old_edits_after_the_profile_changes():
    """Edit a scenario row, change the profile, predict again: the scenarios must be rebuilt from the NEW profile."""
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    assert any(df.value.shape[0] >= 2 for df in at.dataframe)
    fill(at, WEAK)
    button(at, "Predict Placement Probability").click().run()
    assert not at.exception
    tables = [df.value for df in at.dataframe if "Probability" in df.value.columns]
    assert tables and tables[0].iloc[0]["Scenario"].startswith("Your profile")
    assert f"{expected_pct(WEAK):.1f}" in tables[0].iloc[0]["Probability"]
    # the editable copies start from the NEW profile (CGPA 6.6), not from the previous one (7.7)
    editors = [df.value for df in at.dataframe if "CGPA" in df.value.columns and "Scenario" in df.value.columns]
    assert editors and set(editors[0]["CGPA"]) == {WEAK["CGPA"]} and len(editors[0]) == 2


def test_scenario_editor_state_is_discarded_when_the_result_becomes_stale():
    """Streamlit drops a widget's state when it is not rendered; the editor lives inside the result, so any scenario edits die with the result."""
    at = new_app()
    fill(at, TYP)
    button(at, "Predict Placement Probability").click().run()
    assert "scenario_table" in at.session_state
    widget(at, "HSC_Marks").set_value(61).run()
    assert not shown(at) and "scenario_table" not in at.session_state
    button(at, "Clear profile").click().run()
    assert "scenario_table" not in at.session_state
