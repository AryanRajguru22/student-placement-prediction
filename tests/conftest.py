"""Shared fixtures for the release test-suite. Run from the project root:   python -m pytest tests -q

Nothing here trains or refits anything, and the held-out test split is never opened: an audit hook raises if a file named like it is opened.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
MODEL = ROOT / "models" / "phase7" / "LR_B_final.joblib"
sys.path.insert(0, str(APP))

_FORBIDDEN = ("test" + ".csv", "08_test" + "_predictions")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if any(tok in os.path.basename(os.fsdecode(args[0])).lower() for tok in _FORBIDDEN):
            raise PermissionError("tests must not open the held-out test files")


sys.addaudithook(_audit)

# every test that touches the artifact starts from the same recorded fingerprint
FROZEN_SHA256 = "58e8b04e9666b13be47e0cb5c4850707312976776ab5bdfbaa436874ca3cb0eb"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def sha_before_session():
    return sha256_of(MODEL)


@pytest.fixture(scope="session")
def bundle():
    from prediction import load_bundle
    return load_bundle()


@pytest.fixture(scope="session")
def meta(bundle):
    return bundle.metadata


@pytest.fixture(scope="session")
def golden():
    return json.loads((ROOT / "tests" / "fixtures" / "golden_snapshot.json").read_text(encoding="utf-8"))


TYP = {"CGPA": 7.7, "Internships": 1, "Projects": 2, "Workshops/Certifications": 1, "AptitudeTestScore": 80, "SoftSkillsRating": 4.4,
       "ExtracurricularActivities": "Yes", "PlacementTraining": "Yes", "SSC_Marks": 70, "HSC_Marks": 73}


@pytest.fixture()
def typ():
    return dict(TYP)


@pytest.fixture(scope="session", autouse=True)
def _artifact_unchanged_after_session(sha_before_session):
    yield
    assert sha256_of(MODEL) == sha_before_session == FROZEN_SHA256, "the frozen artifact changed during the test session"
