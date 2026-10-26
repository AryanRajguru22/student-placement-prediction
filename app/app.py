"""Student Placement Probability Predictor (Phase 8A). Run from the project root:  streamlit run app/app.py

The app only consumes the frozen Phase 7 Logistic Regression B artifact and its metadata. It trains nothing, tunes nothing and never reads the
held-out test split or any test prediction file. Entered profiles live only in the browser session's memory: they are never written to disk or logged.
"""
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # make the sibling modules importable however the app is launched

import pandas as pd
import streamlit as st

from explainability import explain
from prediction import BINARY, CONTINUOUS, COUNT, FEATURES, LABELS, ModelLoadError, load_bundle, predict, validate_inputs
from ui_helpers import calculation_table, disclaimer, factor_bars, header, inject_css, result_card, section

log = logging.getLogger("placement_app")

st.set_page_config(page_title="Student Placement Probability Predictor", layout="centered")
inject_css()
header()


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except ModelLoadError as exc:                                 # message is already free of file-system paths
    st.error(f"The prediction model could not be loaded. {exc} See the Troubleshooting section of app/README.md.")
    st.stop()
except Exception:                                             # noqa: BLE001 - never show a stack trace to the user
    log.exception("unexpected error while loading the model")
    st.error("The prediction model could not be loaded because of an unexpected error. See the Troubleshooting section of app/README.md.")
    st.stop()
meta = bundle.metadata
R, L = meta["ranges"], meta["levels"]


def range_help(f: str) -> str:
    return f"Supported range {R[f]['min']:g} to {R[f]['max']:g} (the range observed in the data the model was developed on)."


def num_input(f: str, step: float, fmt: str):
    # No min/max on the widget on purpose: Streamlit would silently fall back to the previous value for an out-of-range entry.
    # Every value is passed to validate_inputs(), which rejects missing or out-of-range input with an explicit message.
    # The step is always a float so Streamlit treats the field as a float input. An integer step makes it an integer input that truncates a typed
    # decimal (78.5 -> 78) before the app sees it. validate_inputs() accepts any finite value in range for these fields, so decimals are kept as typed.
    med = int(R[f]["median"]) if R[f]["integer_valued"] else R[f]["median"]
    v = st.number_input(LABELS[f], value=None, step=float(step), format=fmt, placeholder=f"e.g. {med:g}", help=range_help(f), key=f)
    lo, hi = R[f]["min"], R[f]["max"]
    if v is not None and not lo <= v <= hi:
        st.markdown(f'<div class="field-err" role="alert">Out of range: enter a value from {lo:g} to {hi:g}.</div>', unsafe_allow_html=True)
    else:
        st.caption(f"Supported range: {lo:g} to {hi:g}")
    return v


def level_input(f: str):
    lv = L[f]["levels"]
    n = L[f]["n_per_level"]
    note = [str(k) for k in lv if n[str(k)] < meta["rare_level_threshold_n"]]
    sel = st.selectbox(LABELS[f], lv, index=None, placeholder="Select a level", key=f, help="Valid levels observed in the development data.")
    if note:
        st.caption(f"Level {', '.join(note)} is rare in the development data ({', '.join(str(n[k]) for k in note)} of {meta['n_fit']:,} students); estimates for it are unreliable.")
    return sel


def load_typical_profile():
    """Callback: fill every field with the development-data median / most common level (a starting point, not a recommendation)."""
    for f in CONTINUOUS:
        m = R[f]["median"]
        st.session_state[f] = float(m)
    for f in COUNT + BINARY:
        st.session_state[f] = L[f]["default"]
    st.session_state.pop("last", None)
    st.session_state["stale"] = False


def clear_profile():
    """Callback: empty every field and discard any earlier result."""
    for f in FEATURES:
        st.session_state[f] = None
    st.session_state.pop("last", None)
    st.session_state["stale"] = False


section("Student profile")
st.caption("All ten fields are required. Enter the student's values, or load the typical profile (development-data medians / most common levels) and edit it.")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="group-title">Academic</div>', unsafe_allow_html=True)
        cgpa = num_input("CGPA", 0.1, "%g")
        ssc = num_input("SSC_Marks", 1, "%g")
        hsc = num_input("HSC_Marks", 1, "%g")
    with c2:
        st.markdown('<div class="group-title">Technical</div>', unsafe_allow_html=True)
        internships = level_input("Internships")
        projects = level_input("Projects")
        workshops = level_input("Workshops/Certifications")
        aptitude = num_input("AptitudeTestScore", 1, "%g")
    with c3:
        st.markdown('<div class="group-title">Employability</div>', unsafe_allow_html=True)
        soft = num_input("SoftSkillsRating", 0.1, "%g")
        extra = st.radio(LABELS["ExtracurricularActivities"], ["No", "Yes"], index=None, horizontal=True, key="ExtracurricularActivities")
        training = st.radio(LABELS["PlacementTraining"], ["No", "Yes"], index=None, horizontal=True, key="PlacementTraining")
    b1, b2, b3, _ = st.columns([1.3, 1.1, 1, 0.6])
    with b1:
        submitted = st.button("Predict Placement Probability", type="primary", width="stretch")
    with b2:
        st.button("Load typical profile", on_click=load_typical_profile, width="stretch")
    with b3:
        st.button("Clear profile", on_click=clear_profile, width="stretch")

raw = {"CGPA": cgpa, "Internships": internships, "Projects": projects, "Workshops/Certifications": workshops, "AptitudeTestScore": aptitude,
       "SoftSkillsRating": soft, "ExtracurricularActivities": extra, "PlacementTraining": training, "SSC_Marks": ssc, "HSC_Marks": hsc}

if submitted:
    st.session_state["stale"] = False
    clean, errors = validate_inputs(raw, meta)
    if errors:
        st.session_state.pop("last", None)
        st.error("Please correct the following before a prediction can be made:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        try:
            st.session_state["last"] = {"raw": dict(raw), "clean": clean, "pred": predict(bundle, clean), "exp": explain(bundle.pipeline, clean)}
        except Exception as exc:                                  # noqa: BLE001 - never show a stack trace or raw exception text to the user
            st.session_state.pop("last", None)
            # type name + stack frames only: exception text could echo input values, and student inputs are never logged
            log.error("prediction failed (%s)\n%s", type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)))
            st.error("The prediction could not be computed because of an internal error. No result is shown. Reload the page and try again; "
                     "if it persists, see the Troubleshooting section of app/README.md.")

last = st.session_state.get("last")
if last and last["raw"] != raw:                                   # an input was edited after the prediction: the old result no longer describes this profile
    st.session_state.pop("last", None)
    st.session_state["stale"] = True
    last = None
if st.session_state.get("stale"):
    st.markdown('<div class="stale" role="status">The profile has changed since the last prediction, so the earlier result has been removed. '
                'Press <b>Predict Placement Probability</b> to estimate for the current inputs.</div>', unsafe_allow_html=True)

if last:
    pred, exp, clean = last["pred"], last["exp"], last["clean"]
    section("Result")
    result_card(pred)
    for f in COUNT:
        n = L[f]["n_per_level"][str(clean[f])]
        if n < meta["rare_level_threshold_n"]:
            st.warning(f"{LABELS[f]} = {clean[f]} is rare in the development data ({n} of {meta['n_fit']:,} students); the model's estimate for this level is unreliable.")
    section("Key factors influencing this estimate")
    factor_bars(exp)
    with st.expander("Show the full calculation"):
        st.caption("log-odds = intercept + sum of (transformed feature value x model coefficient); probability = 1 / (1 + e^-log-odds).")
        st.dataframe(calculation_table(exp, pred).style.format({"Log-odds contribution": "{:+.3f}", "Odds multiplier": "{:.2f}"}, na_rep=""), hide_index=True, width="stretch")
    with st.expander("Inputs used for this estimate"):
        st.table({LABELS[f]: [f"{clean[f]:g}" if isinstance(clean[f], float) else str(clean[f])] for f in FEATURES})
    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    disclaimer()

section("Model information")
rep, ds = meta["phase7_reported"], meta["dataset"]
thr = meta["thresholds"]
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"- **Model:** {meta['model_name']} (L2 regularised, C = {meta['config']['C']:g}, no class weighting). Logistic regression, gradient boosting and random forest "
                "performed practically the same on the primary metrics; logistic regression was retained for its simplicity and interpretability, not because it was uniquely better.\n"
                f"- **Training data:** train + validation ({meta['n_fit']:,} students)\n"
                f"- **Thresholds:** {thr['operating']:.2f} (project threshold) and {thr['reference']:.2f} (conventional comparison)\n"
                f"- **Probabilities:** raw model output, no recalibration")
with c2:
    st.markdown(f"- **Dataset:** [{ds['source']}]({ds['url']}), licence {ds['license']}. {ds['n_raw']:,} supplied rows; {ds['n_modeling']:,} remain after removing surplus exact duplicates "
                f"and excluding conflicting-label groups (split {ds['n_train']:,} / {ds['n_validation']:,} / {ds['n_test']:,} for train / validation / held-out test).\n"
                f"- **Frozen artifact fingerprint (SHA-256):** `{meta['artifact_sha256'][:16]}...`")
st.markdown(f"**Held-out evaluation ({rep['n_test']:,} students, one-time Phase 7 evaluation).** ROC-AUC {rep['roc_auc'][0]:.3f} (95% CI {rep['roc_auc'][1]:.3f}-{rep['roc_auc'][2]:.3f}), "
            f"log loss {rep['log_loss'][0]:.3f}, Brier score {rep['brier'][0]:.3f}, expected calibration error {rep['ece']:.3f}, calibration slope {rep['calibration_slope']:.2f}. "
            "Classification at the two thresholds:")
tm = rep["threshold_metrics"]
table = pd.DataFrame([{"Threshold": f"{k}" + (" (project)" if float(k) == thr["operating"] else " (conventional)"), "Accuracy": v["accuracy"], "Precision": v["precision"],
                       "Recall": v["recall"], "F1": v["f1"]} for k, v in sorted(tm.items(), key=lambda kv: float(kv[0]))]).set_index("Threshold")
st.table(table.style.format("{:.3f}"))
st.caption("These historical figures were computed once in Phase 7 and are not recomputed by this app. The held-out test data were not used for model development, "
           "selection, tuning, calibration or threshold selection.")
with st.expander("About the thresholds and the limits of this tool"):
    st.markdown(
        f"- The **{thr['operating']:.2f} project threshold** was selected in Phase 6 by a rule fixed in advance (highest validation F1). It favours finding students "
        f"who are placed (higher recall) at the cost of more false alarms (lower precision) than the conventional {thr['reference']:.2f} threshold. Neither threshold is claimed to be the best: "
        "the right cut-off depends on the relative cost of false positives and false negatives. The probability itself is the primary output.\n"
        "- The interpretation bands (0-20%, 20-40%, ...) are simple descriptive labels, not confidence intervals, and do not coincide with the classification thresholds.\n"
        "- The model learned **associations** from one observational dataset. The observed structure is consistent with a rule-based or constructed dataset, although this is not "
        "established by the dataset source, and strong performance on this dataset does not establish equivalent performance on real-world university placement populations. "
        "Changing a characteristic in the form does not tell you what would happen if a real student changed it.\n"
        "- A probability describes a group of similar students, not a certainty for an individual: roughly 1 in 5 students was misclassified in held-out evaluation.\n"
        "- **Privacy:** entered values are used only to compute the result in this browser session. The app does not store or log them and does not send them to any other service.")
