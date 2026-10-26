"""Phase 7 documentation / figure correction (post-evaluation audit). DOCUMENTATION ONLY - no model is trained, no metric is recomputed,
no cell is executed and no test file is read.

What it does (run once from the project root):
  1. applies wording corrections to src/build_notebook_08.py and to the *executed* notebooks/08_final_test_evaluation.ipynb
     (markdown, code-cell source text and the stored text output of the final-audit cell);
  2. regenerates ONLY results/figures/modeling/phase7_test_reliability.png from the saved results/metrics/modeling/08_test_calibration_bins.csv
     and 08_test_calibration_metrics.csv (the stale "orange bar" note is replaced by a note that reflects the data), and swaps the stored
     figure output of the reliability cell in the notebook.
Every replacement must match exactly once, or already be applied (the script is re-runnable; the figure is always regenerated from the saved CSVs).
Revision 2: the figure note was moved to the empty upper-left of the right-hand panel (it touched bar 5 at the bottom).
"""
import base64
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(".")
BUILDER, NB_PATH = ROOT / "src/build_notebook_08.py", ROOT / "notebooks/08_final_test_evaluation.ipynb"
MET, FIG = ROOT / "results/metrics/modeling", ROOT / "results/figures/modeling"

HISTORY_MD = """## Test-set access history (disclosed)
Test data were not used for model development, selection, tuning, calibration, or threshold selection. Earlier exploratory notebooks (Phases 1-3) accessed the eventual test rows for descriptive analysis (data understanding and split construction, EDA, statistics); these rows were not used to fit or evaluate predictive models. Before this notebook ran, two orientation shell commands (`head -2 data/processed/test.csv` and `wc -l data/processed/*.csv`) read the processed test file outside the audit hook; neither was used for any model decision, tuning, threshold, calibration or fitting. The Phase 4-6 audit hooks describe those notebooks only. In Phase 7 the test set is intentionally accessed **once**, under the audit gate below, after the modelling protocol has been frozen.

"""

# (old, new) applied to the builder source and to notebook cell source text
SRC_REPS = [
    ("the real test.csv is never touched.", "the real test.csv is not read."),
    ("## Test-access controls (enforced in code)\n", HISTORY_MD + "## Test-access controls (enforced in code)\n"),
    ('raw and processed test files unchanged (size and mtime), raw test file never opened.")',
     'raw and processed test files unchanged (size and mtime); the raw test file was not opened by this notebook.")'),
    ("inside the gated cell; the raw test file was never opened; access marker", "inside the gated cell; the raw test file was not opened by this notebook; access marker"),
    ('print("\\nTest set was used only for the pre-registered final evaluation and was not used for model selection, threshold tuning, calibration selection, or feature selection.")',
     'print("\\nTest data were not used for model development, selection, tuning, calibration, or threshold selection. Phases 1-3 accessed the eventual test rows descriptively; Phase 7 performed the final gated test evaluation only after the protocol was frozen.")'),
    ('ax.text(0.99, 0.02, "orange bar: mean prediction outside the Wilson 95% interval", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=NEUTRAL)',
     'note = ("orange bar: mean prediction outside the Wilson 95% interval" if bins.gap_outside_wilson_ci.any()\n'
     '        else f"all {NB} bin mean predictions lie inside the Wilson 95% interval of the observed rate")\n'
     'ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=NEUTRAL)',
     'ax.text(0.015, 0.985, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NEUTRAL)'),
    # revision 2: note position
    ('ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=NEUTRAL)',
     'ax.text(0.015, 0.985, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NEUTRAL)'),
]
# stored text output of the final-audit cell
OUT_REPS = [
    ("raw and processed test files unchanged (size and mtime), raw test file never opened.", "raw and processed test files unchanged (size and mtime); the raw test file was not opened by this notebook."),
    ("inside the gated cell; the raw test file was never opened; access marker", "inside the gated cell; the raw test file was not opened by this notebook; access marker"),
    ("Test set was used only for the pre-registered final evaluation and was not used for model selection, threshold tuning, calibration selection, or feature selection.",
     "Test data were not used for model development, selection, tuning, calibration, or threshold selection. Phases 1-3 accessed the eventual test rows descriptively; Phase 7 performed the final gated test evaluation only after the protocol was frozen."),
]

def apply_reps(text, reps, where):
    # sequential; each replacement (old, new[, later_state...]) is applied once, or skipped if already applied
    for r in reps:
        old, new, alts = r[0], r[1], r[2:]
        if new in text or any(a in text for a in alts): continue
        assert text.count(old) == 1, f"{where}: expected exactly one match, found {text.count(old)}: {old[:70]!r}"
        text = text.replace(old, new)
    return text

# ---------------- 1. builder ----------------
b = apply_reps(BUILDER.read_text(encoding="utf-8"), SRC_REPS, "builder")
BUILDER.write_text(b, encoding="utf-8")

# ---------------- 2. regenerate ONLY the reliability figure from saved CSVs ----------------
NB_BINS = 10
bins = pd.read_csv(MET / "08_test_calibration_bins.csv"); lr = pd.read_csv(MET / "08_test_calibration_metrics.csv").set_index("model").loc["LR_B"]
bins["gap_outside_wilson_ci"] = (bins.mean_pred < bins.ci_lo) | (bins.mean_pred > bins.ci_hi)
N_TE = int(bins.n.sum()); assert N_TE == 1487 and len(bins) == NB_BINS
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
INK, NEUTRAL, LR_BLUE, ORANGE = "#1d2733", "#6b7785", "#0072B2", "#D55E00"
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]; ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.2, label="Perfect calibration")
ax.errorbar(bins.mean_pred, bins.obs_rate, yerr=[bins.obs_rate - bins.ci_lo, bins.ci_hi - bins.obs_rate], fmt="o-", color=LR_BLUE, ecolor=LR_BLUE, elinewidth=1.4, capsize=3, ms=6, lw=1.5,
            markeredgecolor="white", label="LR_B, test (observed rate, Wilson 95%)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.set_xlabel("Mean predicted probability (bin)"); ax.set_ylabel("Observed placement rate (bin)")
ax.set_title("Reliability diagram, test set (10 equal-count bins)"); ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.text(0.97, 0.05, f"intercept {lr['cal_intercept']:+.2f}   slope {lr['cal_slope']:.2f}\nCITL {lr['citl']:+.2f}   ECE {lr['ece_quantile']:.3f}\nBrier {lr['brier']:.3f}   n = {N_TE:,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c9ced4"))
ax = axes[1]
ax.bar(bins.bin, -bins.calibration_gap, color=np.where(bins.gap_outside_wilson_ci, ORANGE, LR_BLUE), width=0.7)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(bins.bin); ax.set_xlabel("Bin (1 = lowest predicted probability)"); ax.set_ylabel("Observed minus mean predicted")
ax.set_title("Per-bin calibration gap (above 0: observed higher than predicted)"); ax.grid(axis="x", visible=False)
note = ("orange bar: mean prediction outside the Wilson 95% interval" if bins.gap_outside_wilson_ci.any()
        else f"all {NB_BINS} bin mean predictions lie inside the Wilson 95% interval of the observed rate")
ax.text(0.015, 0.985, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NEUTRAL)
fig.tight_layout(); fig.savefig(FIG / "phase7_test_reliability.png", dpi=200, bbox_inches="tight"); plt.close(fig)
png_b64 = base64.b64encode((FIG / "phase7_test_reliability.png").read_bytes()).decode("ascii")

# ---------------- 3. executed notebook: source text, final-audit output text, reliability figure output (no cell is executed) ----------------
nb = nbformat.read(NB_PATH, as_version=4)
for c in nb.cells:
    if c.cell_type == "markdown" and c.source.startswith("<!--PHASE7"): continue
    for r in SRC_REPS:
        old, new, alts = r[0], r[1], r[2:]
        if new in c.source or any(a in c.source for a in alts): continue
        if old in c.source: c.source = c.source.replace(old, new)
all_src = "\n".join(c.source for c in nb.cells if not c.source.startswith("<!--PHASE7"))
assert all_src.count("## Test-set access history (disclosed)") == 1 and 'ha="left", va="top", fontsize=8.5' in all_src and "Test set was used only for" not in all_src
used_out = [0] * len(OUT_REPS); fig_swapped = 0
for c in nb.cells:
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            t = "".join(o.text) if isinstance(o.text, list) else o.text
            for i, (old, new) in enumerate(OUT_REPS):
                if old in t: t = t.replace(old, new); used_out[i] += 1
            o.text = t
    if "phase7_test_reliability" in c.source:
        for o in c.outputs:
            if o.output_type == "display_data" and "image/png" in o.data: o.data["image/png"] = png_b64; fig_swapped += 1
assert all(u in (0, 1) for u in used_out), used_out
assert fig_swapped == 1, fig_swapped
nbformat.write(nb, NB_PATH)
print("builder + executed notebook patched (no cell executed); reliability figure regenerated from saved CSVs and swapped into the notebook output")
