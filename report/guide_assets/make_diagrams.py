import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = "D:/Student-Placement-Prediction/student-placement-prediction/report/guide_assets/"
NAVY, LIGHT, ORANGE, GREY = "#12355b", "#e8eef6", "#c2570c", "#5b6670"
def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)
def arrow(ax, p, q, color=NAVY):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color))

fig, ax = plt.subplots(figsize=(8.4, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
W = 2.2
box(ax, 0.0, 3.1, W, 1.5, "Original dataset\n10,000 rows x 11 cols\n(supplied as 7,225 /\n1,275 / 1,500)", bold=True)
box(ax, 2.6, 3.1, W, 1.5, "Collapse same-label\nduplicate profiles\nto one record\n-72 rows (66 groups)", fc="#fff3e8", ec=ORANGE)
box(ax, 5.2, 3.1, W, 1.5, "Exclude conflicting-\nlabel groups (same 10\nfeatures, different label)\n8 groups = -16 rows", fc="#fff3e8", ec=ORANGE)
box(ax, 7.8, 3.1, W, 1.5, "Cleaned dataset\n9,912 rows\n4,119 Placed /\n5,793 NotPlaced", bold=True)
for a in (2.2, 4.8, 7.4): arrow(ax, (a, 3.85), (a + 0.4, 3.85))
ax.text(0.1, 2.4, "stratified split, seed 42, zero profile overlap", ha="left", fontsize=8, color=GREY)
box(ax, 0.0, 0.3, 3.0, 1.2, "Train 6,938 (70%)\nfits the model", fc="#e6f2e6", ec="#2d6a2d", fs=8.5)
box(ax, 3.4, 0.3, 3.2, 1.2, "Validation 1,487 (15%)\nchooses model and threshold", fc="#e6f2e6", ec="#2d6a2d", fs=8.5)
box(ax, 7.0, 0.3, 3.0, 1.2, "Test 1,487 (15%)\none final evaluation", fc="#e6f2e6", ec="#2d6a2d", fs=8.5)
for x in (1.5, 5.0, 8.5): arrow(ax, (8.9, 3.1), (x, 1.5), color=GREY)
fig.savefig(OUT + "g01_cleaning_flow.png", dpi=170, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.4, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.text(0.1, 6.75, "TRAINING TIME (notebooks 01-08, built by src/build_notebook_NN.py)", fontsize=8.5, fontweight="bold", color=NAVY)
box(ax, 0.1, 5.4, 2.1, 1.0, "data/raw/*.csv\n(10,000 rows)", fs=7.5)
box(ax, 2.6, 5.4, 2.3, 1.0, "02: clean + split\ndata/processed/\ntrain, val, test", fs=7.5)
box(ax, 5.3, 5.4, 2.1, 1.0, "05-07: fit, compare,\ncalibrate, threshold\nmodels/phase4/", fs=7.5)
box(ax, 7.8, 5.4, 2.1, 1.0, "08: freeze, refit on\ntrain+val, test once\nmodels/phase7/", fs=7.5)
for a, b in [(2.2, 2.6), (4.9, 5.3), (7.4, 7.8)]: arrow(ax, (a, 5.9), (b, 5.9))
box(ax, 7.8, 3.9, 2.1, 1.0, "LR_B_final.joblib\n(preprocessing + model)", fc="#fff3e8", ec=ORANGE, fs=7.5, bold=True)
arrow(ax, (8.85, 5.4), (8.85, 4.9))
ax.text(0.1, 3.7, "INFERENCE TIME (app/, run with: streamlit run app/app.py)", fontsize=8.5, fontweight="bold", color=NAVY)
box(ax, 0.1, 2.0, 2.1, 1.0, "app.py, ui_helpers.py\nform, buttons,\nresult card, CSS", fs=7.5)
box(ax, 2.6, 2.0, 2.3, 1.0, "prediction.py\nvalidate_inputs(),\nto_frame(), predict()", fs=7.5)
box(ax, 5.3, 2.0, 2.1, 1.0, "Frozen Pipeline\nscale + one-hot\n-> logistic model", fs=7.5)
box(ax, 7.8, 2.0, 2.1, 1.0, "explainability.py\nper-feature log-odds\ncontributions", fs=7.5)
arrow(ax, (2.2, 2.5), (2.6, 2.5)); arrow(ax, (4.9, 2.5), (5.3, 2.5)); arrow(ax, (7.4, 2.5), (7.8, 2.5))
arrow(ax, (8.3, 3.9), (6.5, 3.0), color=ORANGE)
ax.text(9.9, 3.3, "loaded after SHA-256 check", fontsize=7.5, color=ORANGE, ha="right")
box(ax, 0.1, 0.2, 4.6, 0.9, "app/model_metadata.json: ranges, levels,\nthresholds, SHA-256, headline metrics", fs=7.5, fc="#f4f4f4", ec=GREY)
box(ax, 5.1, 0.2, 4.8, 0.9, "app/verify_app.py: 178 checks, no test split\nreport/, presentation/: documentation", fs=7.5, fc="#f4f4f4", ec=GREY)
fig.savefig(OUT + "g02_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("ok")
