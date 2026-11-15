"""Downloadable prediction report: one self-contained HTML page (print it to PDF from the browser). Every number comes from the same Prediction /
Explanation objects the screen shows; every dynamic string is HTML-escaped. Nothing is written to disk or sent anywhere: the bytes go straight to the browser."""
from __future__ import annotations

from datetime import datetime
from html import escape

from content import FIELD_HELP, LIMITATIONS_SHORT
from explainability import Explanation
from prediction import FEATURES, LABELS, Prediction
from ui_helpers import BAND_EDGES, prob_text

CSS = """
body{font-family:Segoe UI,Helvetica,Arial,sans-serif;color:#1D2733;max-width:820px;margin:2rem auto;padding:0 1.2rem;line-height:1.5}
h1{color:#0F4C75;font-size:1.6rem;margin-bottom:.1rem} h2{font-size:1.1rem;margin:1.6rem 0 .4rem;border-bottom:1px solid #D5DCE4;padding-bottom:.2rem}
table{border-collapse:collapse;width:100%;font-size:.92rem} th,td{border:1px solid #D5DCE4;padding:.35rem .55rem;text-align:left} th{background:#F6F8FB}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.big{font-size:2.6rem;font-weight:700;color:#0F4C75;margin:.2rem 0} .muted{color:#52606D;font-size:.88rem}
.box{border-left:4px solid #7B8794;background:#F6F8FB;padding:.6rem .9rem;font-size:.92rem}
@media print{body{margin:0}}
"""


def _val(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def build_report_html(clean: dict, pred: Prediction, exp: Explanation, meta: dict, generated: datetime | None = None) -> str:
    generated = generated or datetime.now().astimezone()
    stamp = generated.strftime("%Y-%m-%d %H:%M %Z").strip()
    op, ref = pred.operating_threshold, pred.reference_threshold
    shown = prob_text(pred.probability, (op, ref) + BAND_EDGES)
    cls = lambda yes: "Placed" if yes else "Not Placed"
    profile = "".join(f"<tr><td>{escape(LABELS[f])}</td><td class='num'>{escape(_val(clean[f]))}</td><td class='muted'>{escape(FIELD_HELP[f])}</td></tr>" for f in FEATURES)
    df = exp.contributions.sort_values("contribution", key=lambda s: -s.abs())
    contrib = "".join(f"<tr><td>{escape(r.label)}</td><td>{escape(r.detail)}</td><td class='num'>{r.contribution:+.3f}</td><td class='num'>{r.odds_multiplier:.2f}</td></tr>"
                      for r in df.itertuples())
    rep = meta["phase7_reported"]
    tm = rep["threshold_metrics"]
    hist = "".join(f"<tr><td>{escape(k)}</td>" + "".join(f"<td class='num'>{tm[k][m]:.3f}</td>" for m in ("accuracy", "precision", "recall", "f1")) + "</tr>"
                   for k in sorted(tm, key=float))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Placement probability report</title><style>{CSS}</style></head><body>
<h1>Student Placement Probability Report</h1>
<div class="muted">Generated {escape(stamp)} by the Student Placement Probability Predictor ({escape(meta['model_name'])}, frozen Phase 7 model, SHA-256 {escape(meta['artifact_sha256'][:16])}&hellip;)</div>

<h2>Result</h2>
<div class="big">{escape(shown)}</div>
<div class="muted">Estimated probability of placement: a model-based estimate for students with these characteristics, not a guarantee.</div>
<table style="margin-top:.8rem"><tr><th>Cut-off</th><th>Rule</th><th>Classification</th></tr>
<tr><td>{op:.2f} (project threshold)</td><td>Placed if probability &ge; {op:.2f}</td><td><b>{cls(pred.placed_at_operating)}</b></td></tr>
<tr><td>{ref:.2f} (conventional threshold)</td><td>Placed if probability &ge; {ref:.2f}</td><td><b>{cls(pred.placed_at_reference)}</b></td></tr></table>
<p class="muted">Probability band: <b>{escape(pred.band)}</b>. The band is a descriptive label, not a confidence interval. Neither threshold is claimed to be the best; the probability is the primary output.</p>

<h2>Profile entered</h2>
<table><tr><th>Field</th><th>Value</th><th>Meaning</th></tr>{profile}</table>

<h2>What contributed to this estimate</h2>
<p class="muted">Log-odds = baseline + sum of the contributions below. Contributions are relative to a baseline student (development-data average scores, reference categories) and are
<b>associations</b> the model learned from one dataset, not causal effects. Baseline (intercept) = {exp.intercept:+.3f}; total log-odds = {exp.logit_reconstructed:+.3f}.</p>
<table><tr><th>Factor</th><th>Your value (baseline)</th><th>Log-odds contribution</th><th>Odds multiplier</th></tr>{contrib}</table>

<h2>How to interpret this</h2>
<p>The model estimates how often students with similar characteristics were placed in its development data. It cannot know your individual circumstances, and a
high or low probability does not guarantee or rule out any outcome. It is not a recruitment decision tool. Changing a value in this app shows what the model would
estimate for a different hypothetical profile, not what would happen if a real student made that change.</p>

<h2>Limitations</h2>
<div class="box">{escape(LIMITATIONS_SHORT)}</div>
<p class="muted">Historical held-out evaluation of this model (n = {rep['n_test']:,}, one-time Phase 7 evaluation, not recomputed here): ROC-AUC {rep['roc_auc'][0]:.3f},
log loss {rep['log_loss'][0]:.3f}, Brier score {rep['brier'][0]:.3f}. These describe the development dataset, not real-world placement populations.</p>
<table><tr><th>Threshold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{hist}</table>
<p class="muted">Data source: {escape(meta['dataset']['source'])} (licence {escape(meta['dataset']['license'])}). This report was generated in your browser session and was not stored by the app.</p>
</body></html>"""
