"""Hypothetical scenario comparison: the same frozen model run on several valid input profiles. No modelling logic of its own."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from prediction import FEATURES, Prediction, predict, validate_inputs

MAX_SCENARIOS = 6


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    clean: dict | None
    pred: Prediction | None
    errors: list[str]


def _py(v):
    """numpy scalar -> plain Python value; missing -> None (so the validator reports it)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


def evaluate_scenarios(bundle, table: pd.DataFrame) -> list[ScenarioResult]:
    """Validate and predict each row of the scenario table. Invalid rows are reported with their errors and never predicted or silently repaired."""
    out: list[ScenarioResult] = []
    for i, row in enumerate(table.to_dict("records")):
        name = str(_py(row.get("Scenario")) or f"Scenario {i + 1}")[:40]
        raw = {f: _py(row.get(f)) for f in FEATURES}
        clean, errors = validate_inputs(raw, bundle.metadata)
        if errors:
            out.append(ScenarioResult(name, None, None, errors))
        else:
            out.append(ScenarioResult(name, clean, predict(bundle, clean), []))
    return out


def changed_fields(base: dict, other: dict) -> list[str]:
    return [f for f in FEATURES if base[f] != other[f]]
