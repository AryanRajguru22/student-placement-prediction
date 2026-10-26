"""Prediction layer of the placement-probability app.

Consumes the FROZEN Phase 7 artifact (models/phase7/LR_B_final.joblib: the Phase 4 LR_B pipeline, refit on train + validation) as-is. Nothing is
trained, tuned or recalibrated here, and no preprocessing is re-implemented: the pipeline's own ColumnTransformer does it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

log = logging.getLogger("placement_app")       # developer diagnostics go to the server console; student inputs are never logged

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "phase7" / "LR_B_final.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"

FEATURES = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating",
            "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
CONTINUOUS = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
COUNT = ["Internships", "Projects", "Workshops/Certifications"]
BINARY = ["ExtracurricularActivities", "PlacementTraining"]
LABELS = {"CGPA": "CGPA", "Internships": "Internships", "Projects": "Projects", "Workshops/Certifications": "Workshops / Certifications",
          "AptitudeTestScore": "Aptitude Test Score", "SoftSkillsRating": "Soft Skills Rating", "ExtracurricularActivities": "Extracurricular Activities",
          "PlacementTraining": "Placement Training", "SSC_Marks": "SSC Marks", "HSC_Marks": "HSC Marks"}

# probability interpretation bands (descriptive labels for the model's probability; NOT confidence intervals)
BANDS = [(0.2, "Very low estimated probability"), (0.4, "Low estimated probability"), (0.6, "Moderate estimated probability"),
         (0.8, "High estimated probability"), (1.0000001, "Very high estimated probability")]


class ModelLoadError(RuntimeError):
    """The frozen model or its metadata could not be loaded / verified. The message is safe to show to users (no file-system paths);
    the underlying cause is chained (``__cause__``) and logged for developers."""


@dataclass(frozen=True)
class ModelBundle:
    pipeline: object
    metadata: dict


@dataclass(frozen=True)
class Prediction:
    probability: float            # P(Placed), unrounded
    logit: float                  # log-odds (decision function)
    operating_threshold: float
    reference_threshold: float
    placed_at_operating: bool     # p >= operating threshold
    placed_at_reference: bool     # p >= reference threshold
    band: str


def load_bundle() -> ModelBundle:
    """Load the frozen pipeline and verify it is the artifact the metadata was built from (SHA-256)."""
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        wanted_sha = metadata["artifact_sha256"]
    except Exception as exc:                                     # noqa: BLE001 - reported to the UI in sanitised form
        log.exception("could not read the model metadata")
        raise ModelLoadError("The model metadata file (app/model_metadata.json) is missing or unreadable.") from exc
    try:
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    except Exception as exc:                                     # noqa: BLE001
        log.exception("could not read the frozen model artifact")
        raise ModelLoadError("The frozen model file (models/phase7/LR_B_final.joblib) is missing or unreadable.") from exc
    if digest != wanted_sha:
        raise ModelLoadError("The model file does not match the frozen Phase 7 fingerprint recorded in app/model_metadata.json; refusing to use it.")
    fitted_with = metadata.get("sklearn_version_at_fit")
    if fitted_with and sklearn.__version__ != fitted_with:
        raise ModelLoadError(f"The frozen model was fitted with scikit-learn {fitted_with} but {sklearn.__version__} is installed. "
                             f"Install the pinned version: pip install scikit-learn=={fitted_with}")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as exc:                                     # noqa: BLE001
        log.exception("could not deserialise the frozen model")
        raise ModelLoadError("The frozen model could not be loaded. Check that the dependencies in requirements are installed.") from exc
    if list(getattr(pipeline, "feature_names_in_", [])) != FEATURES:
        raise ModelLoadError("The loaded pipeline does not expect the documented ten input features.")
    return ModelBundle(pipeline=pipeline, metadata=metadata)


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def validate_inputs(raw: dict, metadata: dict) -> tuple[dict, list[str]]:
    """Validate one student's inputs against the ranges/levels the model was developed on. Nothing is clipped or coerced silently:
    any missing, non-numeric, non-finite, out-of-range or unknown-level value is reported and no prediction is made."""
    errors: list[str] = []
    clean: dict = {}
    for f in FEATURES:
        label = LABELS[f]
        v = raw.get(f)
        if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()):
            errors.append(f"{label}: a value is required.")
            continue
        if f in CONTINUOUS:
            if not _is_number(v) or not math.isfinite(float(v)):
                errors.append(f"{label}: '{v}' is not a valid number.")
                continue
            lo, hi = metadata["ranges"][f]["min"], metadata["ranges"][f]["max"]
            if not lo <= float(v) <= hi:
                errors.append(f"{label}: {v} is outside the range the model was developed on ({lo:g} to {hi:g}); no prediction is made for out-of-range values.")
                continue
            clean[f] = float(v)
        elif f in COUNT:
            allowed = metadata["levels"][f]["levels"]
            if not _is_number(v) or not math.isfinite(float(v)) or float(v) != int(float(v)) or int(float(v)) not in allowed:
                errors.append(f"{label}: '{v}' is not a valid level (allowed: {', '.join(map(str, allowed))}).")
                continue
            clean[f] = int(float(v))
        else:
            allowed = metadata["levels"][f]["levels"]
            if not isinstance(v, str) or v not in allowed:
                errors.append(f"{label}: '{v}' is not a valid choice (allowed: {', '.join(allowed)}).")
                continue
            clean[f] = v
    unknown = sorted(set(raw) - set(FEATURES))
    if unknown:
        errors.append(f"Unexpected input field(s): {', '.join(unknown)}.")
    return clean, errors


def to_frame(clean: dict) -> pd.DataFrame:
    """Single-row dataframe with the exact column names/dtypes the Phase 4 pipeline was fitted on (counts as int)."""
    row = {f: clean[f] for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    for c in COUNT:
        df[c] = df[c].astype(int)
    return df


def probability_band(p: float) -> str:
    for upper, label in BANDS:
        if p < upper:
            return label
    return BANDS[-1][1]


def classify(p: float, threshold: float) -> bool:
    """The rule used throughout Phases 4-7: predict Placed when p >= threshold."""
    return p >= threshold


def predict(bundle: ModelBundle, clean: dict) -> Prediction:
    df = to_frame(clean)
    p = float(bundle.pipeline.predict_proba(df)[0, 1])
    z = float(bundle.pipeline.decision_function(df)[0])
    thr = bundle.metadata["thresholds"]
    if not (0.0 <= p <= 1.0) or not math.isfinite(z):
        raise RuntimeError("The model returned an invalid probability.")
    return Prediction(probability=p, logit=z, operating_threshold=thr["operating"], reference_threshold=thr["reference"],
                      placed_at_operating=classify(p, thr["operating"]), placed_at_reference=classify(p, thr["reference"]), band=probability_band(p))
