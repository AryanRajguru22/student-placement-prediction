"""Draws the new diagrams for the Learning Masterbook and downsizes the few existing figures it reuses.
Presentation only: no data files are read and nothing in the model, data or app is touched.
Run from anywhere:  python report/masterbook_assets/make_figures.py
Output: report/masterbook_assets/img/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
OUT = HERE / "img"
OUT.mkdir(exist_ok=True)

NAVY, BLUE, ORANGE, GREY, SOFT, INK = "#12355b", "#0072B2", "#D55E00", "#6b7580", "#eef2f7", "#1d2733"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GREY, "text.color": INK, "axes.labelcolor": INK})


def box(ax, x, y, w, h, text, fc=SOFT, ec=NAVY, fs=8.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.3, color=color))


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. pipeline flow ---------------------------------------------------------------------------------------------------------------------------
def pipeline_flow():
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 10.3); ax.set_ylim(0, 5.3); ax.axis("off")
    box(ax, 0.1, 2.0, 1.7, 1.3, "10 raw inputs\n(one student)", fc="white", bold=True)
    rows = [(3.95, "StandardScaler\nCGPA, Aptitude, SoftSkills,\nSSC, HSC  →  5 columns"),
            (2.45, "OneHotEncoder (drop ref. level)\nInternships, Projects, Workshops\n→  2 + 3 + 3 = 8 columns"),
            (0.95, "OneHotEncoder (drop='first')\nExtracurricular, PlacementTraining\n→  2 columns (the 'Yes' flags)")]
    for y, t in rows:
        box(ax, 2.35, y - 0.1, 3.5, 1.15, t, fs=7.0)
        arrow(ax, 1.8, 2.65, 2.35, y + 0.45)
        arrow(ax, 5.85, y + 0.45, 6.4, 2.65)
    ax.text(4.05, 5.25, "ColumnTransformer  ('prep')", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.add_patch(Rectangle((2.2, 0.7), 3.8, 4.4, fc="none", ec=GREY, lw=1, ls="--"))
    box(ax, 6.2, 2.0, 1.5, 1.3, "15 numeric\ncolumns", fc="white", bold=True)
    arrow(ax, 7.7, 2.65, 8.2, 2.65)
    box(ax, 8.1, 3.05, 1.8, 1.0, "LogisticRegression\nz = b0 + Σ bj·xj", fs=7.8, fc="#fdf1e8", ec=ORANGE)
    box(ax, 8.1, 1.75, 1.8, 1.0, "sigmoid\np = 1/(1+e^-z)", fs=7.8, fc="#fdf1e8", ec=ORANGE)
    box(ax, 8.1, 0.45, 1.8, 1.0, "p ≥ 0.35 ?\np ≥ 0.50 ?", fs=7.8, fc="#fdf1e8", ec=ORANGE)
    arrow(ax, 9.05, 3.05, 9.05, 2.75, ORANGE); arrow(ax, 9.05, 1.75, 9.05, 1.45, ORANGE)
    ax.text(8.1, 4.25, "'model' step", fontsize=8, fontweight="bold", color=ORANGE)
    save(fig, "f_pipeline.png")


# 2. sigmoid -------------------------------------------------------------------------------------------------------------------------------
def sigmoid():
    z = np.linspace(-6, 6, 400); p = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(z, p, color=NAVY, lw=2.2)
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":"); ax.axhline(0.35, color=ORANGE, lw=0.8, ls="--"); ax.axvline(0, color=GREY, lw=0.8, ls=":")
    z0 = -0.5195; p0 = 1 / (1 + np.exp(-z0))
    ax.plot([z0], [p0], "o", color=ORANGE, ms=8, mec="white", mew=1.5, zorder=5)
    ax.annotate("typical profile\nz = −0.52, p = 0.373", (z0, p0), xytext=(-5.7, 0.62), fontsize=8, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.text(5.9, 0.515, "p = 0.50  (z = 0)", ha="right", fontsize=7.5, color=GREY)
    ax.text(5.9, 0.365, "p = 0.35  (z = −0.619)", ha="right", fontsize=7.5, color=ORANGE)
    ax.set_xlabel("score z  (log-odds)"); ax.set_ylabel("probability p = σ(z)")
    ax.set_ylim(-0.02, 1.02); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "f_sigmoid.png")


# 3. split ---------------------------------------------------------------------------------------------------------------------------------
def split():
    fig, ax = plt.subplots(figsize=(7.4, 2.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0, 3.75, "9,912 cleaned rows  →  stratified, random_state = 42", fontsize=8.5, fontweight="bold", color=NAVY)
    segs = [(0, 70, "TRAIN  6,938\n(70%)", "#cfe3f3"), (70, 15, "VALIDATION\n1,487 (15%)", "#fbe3d1"), (85, 15, "TEST\n1,487 (15%)", "#e1e1e1")]
    for x, w, t, c in segs:
        ax.add_patch(Rectangle((x, 2.3), w, 1.1, fc=c, ec="white", lw=2)); ax.text(x + w / 2, 2.85, t, ha="center", va="center", fontsize=8.3)
    ax.text(35, 1.95, "fit scaler, encoder, coefficients;\ncross-validation lives inside this part", ha="center", va="top", fontsize=7.4, color=GREY)
    ax.text(77.5, 1.95, "compare models,\nchoose threshold", ha="center", va="top", fontsize=7.4, color=GREY)
    ax.text(92.5, 1.95, "one final\nevaluation", ha="center", va="top", fontsize=7.4, color=GREY)
    ax.add_patch(Rectangle((0, 0.1), 85, 0.5, fc="#d9d3ee", ec="white", lw=2))
    ax.text(42.5, 0.35, "FINAL REFIT: train + validation = 8,425 rows (same settings, nothing re-chosen)", ha="center", va="center", fontsize=7.6)
    save(fig, "f_split.png")


# 4. threshold line --------------------------------------------------------------------------------------------------------------------------
def thresholds():
    fig, ax = plt.subplots(figsize=(7.2, 2.3)); ax.set_xlim(0, 1); ax.set_ylim(0, 3); ax.axis("off")
    ax.add_patch(Rectangle((0, 1.2), 0.35, 0.5, fc="#e1e1e1", ec="white")); ax.add_patch(Rectangle((0.35, 1.2), 0.15, 0.5, fc="#ffe0c9", ec="white"))
    ax.add_patch(Rectangle((0.5, 1.2), 0.5, 0.5, fc="#cfe3f3", ec="white"))
    ax.text(0.175, 1.45, "Not Placed\n(both rules)", ha="center", va="center", fontsize=7.8)
    ax.text(0.425, 1.45, "Placed @0.35\nNot Placed @0.50", ha="center", va="center", fontsize=7.0)
    ax.text(0.75, 1.45, "Placed (both rules)", ha="center", va="center", fontsize=7.8)
    for x, c in [(0.35, ORANGE), (0.5, NAVY)]:
        ax.plot([x, x], [1.1, 1.85], color=c, lw=2); ax.text(x, 1.95, f"{x:.2f}", ha="center", fontsize=9, fontweight="bold", color=c)
    ax.plot([0.373], [0.85], "^", color=INK, ms=9); ax.text(0.373, 0.35, "37.3%  (the model's probability)", ha="center", fontsize=8)
    ax.text(0.0, 2.55, "ONE probability, TWO decision rules", fontsize=9, fontweight="bold", color=NAVY)
    for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0): ax.text(x, 1.0, "", fontsize=6)
    save(fig, "f_thresholds.png")


# 5. ensembles ----------------------------------------------------------------------------------------------------------------------------
def ensembles():
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax in axs: ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax = axs[0]; ax.set_title("Random Forest: independent trees", fontsize=9.2, color=NAVY, fontweight="bold")
    box(ax, 3.4, 5.0, 3.2, 0.8, "training data", fc="white", fs=8)
    for i, x in enumerate((0.3, 3.7, 7.1)):
        arrow(ax, 5, 5.0, x + 1.3, 4.1); box(ax, x, 3.3, 2.6, 0.8, f"sample {i+1}", fs=7.4, fc="white")
        arrow(ax, x + 1.3, 3.3, x + 1.3, 2.6); box(ax, x, 1.8, 2.6, 0.8, f"tree {i+1}\n(random cols)", fs=7.2)
        arrow(ax, x + 1.3, 1.8, 5, 1.0)
    box(ax, 2.7, 0.2, 4.6, 0.8, "average the probabilities", fs=8, fc="#fdf1e8", ec=ORANGE)
    ax = axs[1]; ax.set_title("Gradient Boosting: sequential trees", fontsize=9.2, color=NAVY, fontweight="bold")
    xs = [0.1, 2.7, 5.3]
    for i, x in enumerate(xs):
        box(ax, x, 3.0, 2.2, 1.1, f"small tree {i+1}\n(depth 2)", fs=7.4)
        if i < 2: arrow(ax, x + 2.0, 3.55, xs[i + 1], 3.55); ax.text(x + 2.35, 4.15, "fix\nerrors", ha="center", fontsize=6.8, color=ORANGE)
    ax.text(8.35, 3.55, "… 300\nstages", ha="center", va="center", fontsize=8)
    box(ax, 0.2, 1.2, 9.6, 0.9, "score = start + 0.05·tree1 + 0.05·tree2 + …", fs=7.6, fc="#fdf1e8", ec=ORANGE)
    ax.text(5, 5.2, "each new tree is fitted to what the\nprevious trees still get wrong", ha="center", fontsize=7.6, color=GREY)
    save(fig, "f_ensembles.png")


# 6. log-loss curve ----------------------------------------------------------------------------------------------------------------------
def logloss():
    p = np.linspace(0.01, 0.99, 300)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(p, -np.log(p), color=BLUE, lw=2.2, label="true label y = 1:  loss = −log(p)")
    ax.plot(p, -np.log(1 - p), color=ORANGE, lw=2.2, label="true label y = 0:  loss = −log(1−p)")
    ax.set_ylim(0, 4.5); ax.set_xlabel("predicted probability of Placed, p"); ax.set_ylabel("loss for one student")
    ax.legend(frameon=False, fontsize=8, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("confident and wrong:\nloss explodes", (0.04, 3.2), xytext=(0.12, 3.5), fontsize=7.6, color=GREY)
    save(fig, "f_logloss.png")


# 7. ROC toy + confusion grid -----------------------------------------------------------------------------------------------------------
def roc_confusion():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    pos, neg = [0.9, 0.6, 0.4], [0.7, 0.3, 0.2]
    ts = sorted(set(pos + neg + [1.1, -0.1]), reverse=True)
    tpr = [np.mean([s >= t for s in pos]) for t in ts]; fpr = [np.mean([s >= t for s in neg]) for t in ts]
    ax.plot(fpr, tpr, "-o", color=BLUE, lw=2, ms=4); ax.plot([0, 1], [0, 1], ":", color=GREY)
    ax.fill_between(fpr, tpr, step=None, color=BLUE, alpha=0.1)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate (recall)"); ax.set_title("Toy ROC curve (AUC = 7/9 = 0.778)", fontsize=8.6, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1]; ax.set_xlim(0, 4.4); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Confusion matrix, threshold 0.50 (test set)", fontsize=8.6, color=NAVY)
    cells = [(1.6, 1.5, "TN\n735", "#cfe3f3"), (3.0, 1.5, "FP\n134", "#fbe3d1"), (1.6, 0.2, "FN\n145", "#fbe3d1"), (3.0, 0.2, "TP\n473", "#cfe3f3")]
    for x, y, t, c in cells:
        ax.add_patch(Rectangle((x, y), 1.3, 1.2, fc=c, ec="white", lw=3)); ax.text(x + 0.65, y + 0.6, t, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(2.25, 3.0, "predicted\nNot Placed", ha="center", fontsize=7.4); ax.text(3.65, 3.0, "predicted\nPlaced", ha="center", fontsize=7.4)
    ax.text(1.5, 2.1, "actual\nNot Placed", ha="right", va="center", fontsize=7.4); ax.text(1.5, 0.8, "actual\nPlaced", ha="right", va="center", fontsize=7.4)
    save(fig, "f_roc_confusion.png")


# 8. cross-validation ---------------------------------------------------------------------------------------------------------------------
def cv():
    fig, ax = plt.subplots(figsize=(6.4, 2.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    for r in range(5):
        for c in range(5):
            ax.add_patch(Rectangle((1.6 + c * 1.35, 4.3 - r * 0.8), 1.3, 0.65, fc="#fbe3d1" if c == r else "#cfe3f3", ec="white", lw=1.5))
        ax.text(1.4, 4.62 - r * 0.8, f"fit {r+1}", ha="right", va="center", fontsize=7.5)
    ax.text(8.5, 2.6, "orange = held-out fold\n(scored)\nblue = fitted on 4 folds", fontsize=7.6, va="center")
    ax.text(1.6, 5.0, "5-fold stratified CV inside the 6,938 training rows", fontsize=8, fontweight="bold", color=NAVY)
    save(fig, "f_cv.png")


# existing figures, downsized -------------------------------------------------------------------------------------------------------------------
def reuse():
    src = {"g_cleaning.png": REPORT / "guide_assets/g01_cleaning_flow.png", "g_architecture.png": REPORT / "guide_assets/g02_architecture.png",
           "r_coefficients.png": REPORT / "figures/fig11_logistic_coefficients.png", "r_conf050.png": REPORT / "figures/fig16a_test_confusion_050.png",
           "r_conf035.png": REPORT / "figures/fig16b_test_confusion_035.png", "r_reliability.png": REPORT / "figures/fig17_test_reliability.png",
           "r_threshold_metrics.png": REPORT / "figures/fig10_threshold_metrics.png", "r_target.png": REPORT / "figures/fig01_target_distribution.png",
           "r_validation_models.png": REPORT / "figures/fig06_validation_model_comparison.png"}
    for name, p in src.items():
        im = Image.open(p).convert("RGB")
        w = 1000 if "conf0" not in name else 520
        if im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im.save(OUT / name, optimize=True)


if __name__ == "__main__":
    for f in (pipeline_flow, sigmoid, split, thresholds, ensembles, logloss, roc_confusion, cv, reuse): f()
    print(sorted(p.name for p in OUT.iterdir()))
