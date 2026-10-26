"""Presentation helpers (HTML/CSS fragments and small formatting functions) for the Streamlit app. No modelling logic lives here."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from explainability import Explanation
from prediction import BANDS, Prediction

BAND_EDGES = tuple(upper for upper, _ in BANDS[:-1])      # 0.2, 0.4, 0.6, 0.8: the displayed percentage must not round across a band edge either

NAVY, POS, NEG = "#0F4C75", "#0072B2", "#D55E00"          # POS/NEG (colour-blind-safe blue/orange) colour the factor bars only
POS_TEXT, NEG_TEXT = "#00537F", "#A04400"                 # darker shades for small text on white (contrast >= 4.5:1)
INK, MUTED, LINE = "#1D2733", "#52606D", "#D5DCE4"
SERIF = '"Newsreader", Georgia, "Times New Roman", serif'  # no web-font download: falls back to Georgia when Newsreader is not installed
NEGLIGIBLE = 0.005                                        # |log-odds contribution| below this displays as 0.00 and is listed as negligible

CSS = f"""
<style>
.block-container {{ max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }}
h1.app-title {{ font-family: {SERIF}; font-size: 2.3rem; font-weight: 600; letter-spacing: -0.01em; color: {NAVY}; margin: 0 0 .25rem 0; padding: 0; line-height: 1.2; }}
.app-sub {{ font-size: 1.05rem; color: {MUTED}; margin: 0 0 .4rem 0; }}
.rule {{ border: 0; border-top: 1px solid {LINE}; margin: 1rem 0 1.2rem 0; }}
h2.sec-title {{ font-family: {SERIF}; font-size: 1.45rem; font-weight: 600; color: {INK}; margin: 1.8rem 0 .5rem 0; padding: 0; line-height: 1.3; }}
.group-title {{ font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; color: {NAVY}; font-weight: 700; margin: 0 0 .5rem 0; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; }}
.stButton > button, .stFormSubmitButton > button {{ min-height: 2.75rem; }}
.hero {{ border: 1px solid {LINE}; border-radius: 10px; padding: 1.4rem 1.6rem 1.2rem 1.6rem; background: #fff; }}
.hero-label {{ font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; color: {MUTED}; font-weight: 600; }}
.hero-prob {{ font-family: {SERIF}; font-size: 4.4rem; line-height: 1.05; font-weight: 600; color: {NAVY}; margin: .15rem 0 .1rem 0; letter-spacing: -0.02em; font-variant-numeric: lining-nums; }}
.hero-note {{ font-size: .9rem; color: {MUTED}; }}
.ruler {{ margin: 2.1rem 0 .2rem 0; }}
.gauge {{ position: relative; height: 12px; border-radius: 6px; background: #E3E8EE; margin: 0 .5rem 2.1rem .5rem; }}
.gauge-fill {{ height: 100%; border-radius: 6px; background: {NAVY}; }}
.tick {{ position: absolute; top: -6px; width: 2px; height: 24px; background: {INK}; }}
.tick-lab {{ position: absolute; transform: translateX(-50%); font-size: .85rem; font-weight: 600; color: {INK}; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.tick-up {{ top: -30px; }}
.tick-dn {{ top: 21px; }}
.ruler-key {{ font-size: .82rem; color: {MUTED}; line-height: 1.45; }}
.facts {{ padding: .1rem .3rem; }}
.fact {{ padding: .6rem 0; border-bottom: 1px solid #E9EDF2; }}
.fact:last-child {{ border-bottom: 0; }}
.fact-k {{ font-size: .8rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }}
.fact-v {{ font-size: 1.28rem; font-weight: 650; color: {INK}; }}
.fact-s {{ font-size: .86rem; color: {MUTED}; }}
.pill {{ display: inline-block; padding: .08rem .7rem; border-radius: 999px; font-weight: 650; font-size: 1.05rem; }}
.pill-yes {{ background: #E3F0F9; color: #00537F; border: 1px solid #B7D7EC; }}
.pill-no {{ background: #FBEADF; color: #8A3800; border: 1px solid #F0C9AD; }}
.fb-head {{ font-size: .95rem; font-weight: 650; color: {INK}; margin: 1rem 0 .35rem 0; }}
.fb-row {{ display: grid; grid-template-columns: minmax(150px, 34%) 1fr 74px; align-items: center; gap: 12px; padding: 6px 0; border-bottom: 1px solid #EDF0F4; }}
.fb-name {{ font-size: .95rem; font-weight: 600; color: {INK}; }}
.fb-det {{ font-size: .82rem; color: {MUTED}; font-weight: 400; }}
.fb-track {{ position: relative; height: 16px; background: #EEF1F5; border-radius: 4px; }}
.fb-mid {{ position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: #7B8794; }}
.fb-bar {{ position: absolute; top: 2px; bottom: 2px; border-radius: 3px; }}
.fb-val {{ font-size: .92rem; font-variant-numeric: tabular-nums; text-align: right; font-weight: 650; }}
.fb-note {{ font-size: .84rem; color: {MUTED}; margin-top: .7rem; line-height: 1.5; }}
.field-err {{ font-size: .85rem; font-weight: 650; color: #B42318; margin-top: -.4rem; }}
.stale {{ border: 1px solid #B7D7EC; border-left: 4px solid {NAVY}; background: #fff; padding: .7rem 1rem; border-radius: 4px; font-size: .95rem; color: {INK}; margin-top: 1rem; }}
.disclaimer {{ border-left: 4px solid #7B8794; background: #fff; border-top: 1px solid {LINE}; border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE};
              padding: .8rem 1rem; border-radius: 4px; font-size: .92rem; color: {INK}; }}
@media (max-width: 640px) {{
  h1.app-title {{ font-size: 1.75rem; }}
  .hero {{ padding: 1.1rem 1rem 1rem 1rem; }}
  .hero-prob {{ font-size: 3.4rem; }}
  .fb-row {{ grid-template-columns: 1fr auto; row-gap: 6px; }}
  .fb-track {{ grid-column: 1 / -1; grid-row: 2; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown('<h1 class="app-title">Student Placement Probability Predictor</h1>'
                '<div class="app-sub">Predicting placement probability from academic, technical and employability characteristics.</div>'
                '<hr class="rule">', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<h2 class="sec-title">{escape(title)}</h2>', unsafe_allow_html=True)


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"                   # rounded for display only


def prob_text(p: float, thresholds: tuple[float, ...] = ()) -> str:
    """Percentage with one decimal, or as many decimals as needed so that the displayed number never looks like it is on the other side of a
    threshold (or band edge) than the (unrounded) probability that was classified, e.g. p = 0.3496 is shown as 34.96%, not 35.0%."""
    for d in (1, 2, 3, 4, 5, 6):
        if all((round(p * 100, d) >= t * 100) == (p >= t) for t in thresholds):
            return f"{p * 100:.{d}f}%"
    return f"{p * 100:.8f}%"


def _pill(yes: bool) -> str:
    return '<span class="pill pill-yes">Placed</span>' if yes else '<span class="pill pill-no">Not Placed</span>'


def result_html(pred: Prediction) -> tuple[str, str]:
    """(left, right) HTML of the result: probability card with the threshold ruler, and the classification facts. Pure function (testable)."""
    op, ref = pred.operating_threshold, pred.reference_threshold
    shown = prob_text(pred.probability, (op, ref) + BAND_EDGES)
    aria = (f"Estimated placement probability {shown} on a scale from 0 to 100 percent. "
            f"Project threshold at {op * 100:.0f} percent, conventional threshold at {ref * 100:.0f} percent.")
    ruler = (f'<div class="ruler" role="img" aria-label="{escape(aria)}"><div class="gauge">'
             f'<div class="gauge-fill" style="width:{pred.probability * 100:.3f}%"></div>'
             f'<div class="tick" style="left:{op * 100:.1f}%"></div><div class="tick-lab tick-up" style="left:{op * 100:.1f}%">{op:.2f}</div>'
             f'<div class="tick" style="left:{ref * 100:.1f}%"></div><div class="tick-lab tick-dn" style="left:{ref * 100:.1f}%">{ref:.2f}</div></div>'
             f'<div class="ruler-key">Filled bar: estimated probability. Tick marks: {op:.2f} project threshold (label above the bar) '
             f'and {ref:.2f} conventional threshold (label below).</div></div>')
    left = (f'<div class="hero-label">Estimated probability of placement</div><div class="hero-prob">{shown}</div>'
            f'<div class="hero-note">A model-based estimate for students with these characteristics, not a guarantee.</div>{ruler}')
    right = (f'<div class="fact"><div class="fact-k">Project threshold {op:.2f}</div><div class="fact-v">{_pill(pred.placed_at_operating)}</div>'
             f'<div class="fact-s">Placed if probability &ge; {op:.2f}</div></div>'
             f'<div class="fact"><div class="fact-k">Conventional threshold {ref:.2f}</div><div class="fact-v">{_pill(pred.placed_at_reference)}</div>'
             f'<div class="fact-s">Placed if probability &ge; {ref:.2f}</div></div>'
             f'<div class="fact"><div class="fact-k">Probability band</div><div class="fact-v">{escape(pred.band)}</div>'
             f'<div class="fact-s">A descriptive label, not a confidence interval. The threshold changes the label, never the probability; '
             f'neither threshold is claimed to be the best.</div></div>')
    return left, right


def result_card(pred: Prediction) -> None:
    left, right = result_html(pred)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(f'<div class="hero">{left}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="facts">{right}</div>', unsafe_allow_html=True)


def _signed(c: float) -> str:
    return "0.00" if abs(c) < NEGLIGIBLE else f"{c:+.2f}"


def _bar_row(label: str, detail: str, c: float, scale: float) -> str:
    w = min(abs(c) / scale, 1.0) * 50          # percent of the full track; each half-track is 50%
    side = f"left:50%;width:{w:.2f}%;background:{POS}" if c >= 0 else f"right:50%;width:{w:.2f}%;background:{NEG}"
    colour = POS_TEXT if c >= 0 else NEG_TEXT
    return (f'<div class="fb-row"><div class="fb-name">{escape(label)}<div class="fb-det">{escape(detail)}</div></div>'
            f'<div class="fb-track"><div class="fb-mid"></div><div class="fb-bar" style="{side}"></div></div>'
            f'<div class="fb-val" style="color:{colour}">{_signed(c)}</div></div>')


def factor_html(exp: Explanation) -> str:
    df = exp.contributions
    zero = df[df.contribution.abs() < NEGLIGIBLE]
    pos = df[df.contribution >= NEGLIGIBLE].sort_values("contribution", ascending=False)
    neg = df[df.contribution <= -NEGLIGIBLE].sort_values("contribution")
    scale = float(df.contribution.abs().max()) or 1.0
    html = ""
    if len(pos):
        html += '<div class="fb-head">Factors that raise the model\'s estimate</div>' + "".join(_bar_row(r.label, r.detail, r.contribution, scale) for r in pos.itertuples())
    if len(neg):
        html += '<div class="fb-head">Factors that lower the model\'s estimate</div>' + "".join(_bar_row(r.label, r.detail, r.contribution, scale) for r in neg.itertuples())
    if len(zero):
        html += ('<div class="fb-head">At or very near the baseline (contribution below 0.005 in magnitude)</div><div class="fb-det">'
                 + "; ".join(escape(f"{r.label} = {r.detail}") for r in zero.itertuples()) + "</div>")
    html += ('<div class="fb-note">Values are each factor\'s contribution to the model\'s log-odds relative to a baseline student '
             '(training-average scores, reference categories); blue raises and orange lowers the estimate. '
             'Bar length is relative to the largest contribution within this prediction. '
             'These are associations learned by the model, not causal effects.</div>')
    return html


def factor_bars(exp: Explanation) -> None:
    st.markdown(factor_html(exp), unsafe_allow_html=True)


def calculation_table(exp: Explanation, pred: Prediction) -> pd.DataFrame:
    rows = [{"Step": "Baseline (intercept: reference student)", "Input": "", "Log-odds contribution": exp.intercept, "Odds multiplier": float("nan")}]
    for r in exp.contributions.itertuples():
        rows.append({"Step": r.label, "Input": r.detail, "Log-odds contribution": r.contribution, "Odds multiplier": r.odds_multiplier})
    rows.append({"Step": "Total log-odds", "Input": "", "Log-odds contribution": exp.logit_reconstructed, "Odds multiplier": float("nan")})
    rows.append({"Step": "Probability = 1 / (1 + e^-total)", "Input": "", "Log-odds contribution": float("nan"), "Odds multiplier": float("nan")})
    out = pd.DataFrame(rows)
    out.loc[out.index[-1], "Input"] = prob_text(pred.probability, (pred.operating_threshold, pred.reference_threshold) + BAND_EDGES)
    return out


def disclaimer() -> None:
    st.markdown('<div class="disclaimer"><b>Disclaimer.</b> This probability is an estimate produced by the trained model for students with the supplied '
                'characteristics. It is not a guarantee of placement and should not be interpreted as a causal assessment.</div>', unsafe_allow_html=True)
