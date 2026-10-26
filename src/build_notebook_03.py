"""Builds notebooks/03_eda.ipynb (Phase 2 - exploratory data analysis).
Findings prose is read from src/eda_findings.md so it can be written against the executed numbers."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 03 - Exploratory Data Analysis (Phase 2)
**Goal:** understand how student characteristics relate to `PlacementStatus` *before* any modelling.

**Data used (only):** `data/processed/modeling_dataset.csv` and the split files `train.csv`, `validation.csv`, `test.csv`
produced in notebook 02 (de-duplicated, conflicting-label profiles excluded, stratified 70/15/15, seed 42).
The Phase-1 flagged file and the raw supplied splits are *not* used.

**Ground rules:** no models are trained; no features are engineered or saved; outliers are not removed.
Any binning below is **exploratory, for visualisation only**, lives in temporary variables and is never added to the dataset.
All statements are descriptions of *association in this dataset*, not causal claims.
""")

code(r"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC, FIGDIR, MET = ROOT / "data" / "processed", ROOT / "results" / "figures" / "eda", ROOT / "results" / "metrics"
FIGDIR.mkdir(parents=True, exist_ok=True); MET.mkdir(parents=True, exist_ok=True)

# ---- consistent visual style: one fixed colour per class, recessive grid, thin marks ----
PLACED_C, NOT_C, NEUTRAL = "#2166ac", "#e08214", "#6b7785"
PAL = {"Placed": PLACED_C, "NotPlaced": NOT_C}
ORDER = ["NotPlaced", "Placed"]
W = 15
sns.set_theme(style="whitegrid", rc={"grid.color": "#e3e6ea", "axes.edgecolor": "#9aa3ad", "axes.titlesize": 12,
                                      "axes.titleweight": "semibold", "axes.labelsize": 10.5, "figure.dpi": 110})
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 50)

def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight"); plt.show()

SUMMARY = []                       # rows for results/metrics/03_eda_summary.csv
def add(section, item, value, note=""):
    SUMMARY.append({"section": section, "item": item, "value": value, "note": note})

TARGET = "PlacementStatus"
CONT = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]   # many-valued scales
COUNT = ["Internships", "Projects", "Workshops/Certifications"]                    # 3-4 level counts
NUM = CONT + COUNT
CATB = ["ExtracurricularActivities", "PlacementTraining"]
FEATURES = NUM + CATB
""")

md("## 0. Load data and integrity checks")
code(r"""
modeling = pd.read_csv(PROC / "modeling_dataset.csv")
splits = {s: pd.read_csv(PROC / f"{s}.csv") for s in ["train", "validation", "test"]}
df = modeling

# --- required checks ---
assert df.isna().sum().sum() == 0 and all(d.isna().sum().sum() == 0 for d in splits.values()), "missing values"
assert set(df[TARGET]) == {"Placed", "NotPlaced"}, "unexpected target classes"
assert not df.duplicated().sum(), "exact duplicate rows exist"
assert not df.duplicated(subset=FEATURES).sum(), "duplicate feature profiles exist"
assert list(df.columns) == sorted(FEATURES, key=list(pd.read_csv(PROC / "train.csv").columns).index) + [TARGET]
profiles = {s: set(map(tuple, d[FEATURES].values)) for s, d in splits.items()}
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not (profiles[a] & profiles[b]), f"feature-profile leakage {a}/{b}"
assert sum(map(len, splits.values())) == len(df) and set().union(*profiles.values()) == set(map(tuple, df[FEATURES].values))
print("OK - modelling dataset:", df.shape, "| splits:", {s: len(d) for s, d in splits.items()})
print("Checks passed: no missing values, 2 target classes, no duplicate rows/profiles, zero cross-split profile overlap.")

# numeric target view kept OUTSIDE the dataframe (no new columns are created)
y = (df[TARGET] == "Placed").astype(int)

def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def rate_table(by):
    # placement rate per group with n and Wilson 95% CI (descriptive uncertainty)
    t = y.groupby(by, observed=True).agg(n="size", placed="sum")
    t["rate"] = t.placed / t.n
    t["ci_lo"], t["ci_hi"] = wilson(t.placed, t.n)
    return t

OVERALL = y.mean()

def rate_bar(ax, by, title, xlabel=None, rot=0):
    t = rate_table(by)
    x = np.arange(len(t))
    ax.bar(x, t.rate * 100, color=PLACED_C, width=0.62)
    ax.errorbar(x, t.rate * 100, yerr=[(t.rate - t.ci_lo) * 100, (t.ci_hi - t.rate) * 100], fmt="none", ecolor="#1d2733", lw=1.1, capsize=3)
    for xi, (r, hi, n) in enumerate(zip(t.rate, t.ci_hi, t.n)):
        ax.text(xi, hi * 100 + 2, f"{r*100:.1f}%\nn={n:,}", ha="center", va="bottom", fontsize=8.5, color="#1d2733")
    ax.axhline(OVERALL * 100, color=NEUTRAL, ls="--", lw=1)
    ax.text(0.02, 0.97, f"- - overall rate {OVERALL*100:.1f}%", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NEUTRAL)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in t.index], rotation=rot, ha="right" if rot else "center")
    ax.set_ylim(0, 118); ax.set_ylabel("Placement rate (%)"); ax.set_xlabel(xlabel or ""); ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return t

def cohens_d(a, b):
    na, nb_ = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb_ - 1) * b.var(ddof=1)) / (na + nb_ - 2))
    return (a.mean() - b.mean()) / sp
""")

md(r"""
## 1. Dataset overview
Predictors fall into three groups: **5 scale-type numeric** variables (CGPA, aptitude, soft skills, SSC, HSC marks - many
distinct values, but bounded and recorded in coarse steps), **3 count-type numeric** variables (3-4 levels each, treated as
ordered categories when plotting) and **2 binary categorical** variables.
""")
code(r"""
overview = pd.DataFrame({
    "Item": ["Rows", "Predictors", "Numeric predictors", "  scale-type (many levels)", "  count-type (<=4 levels)",
             "Categorical predictors (binary)", "Target", "Placed", "NotPlaced"],
    "Value": [f"{len(df):,}", len(FEATURES), len(NUM), ", ".join(CONT), ", ".join(COUNT), ", ".join(CATB), TARGET,
              f"{(y==1).sum():,} ({y.mean()*100:.2f}%)", f"{(y==0).sum():,} ({(1-y.mean())*100:.2f}%)"]})
add("overview", "rows", len(df)); add("overview", "predictors", len(FEATURES))
add("overview", "placed_pct", round(y.mean() * 100, 2)); add("overview", "notplaced_pct", round((1 - y.mean()) * 100, 2))
overview
""")
code(r"""
desc = df[NUM].agg(["mean", "median", "std", "min", "max", "skew"]).T
desc["n_unique"] = df[NUM].nunique()
print("Descriptive statistics (skewness is most informative for the scale-type variables):")
desc.round(3)
""")
code(r"""
print("Categorical predictors:")
pd.DataFrame({c: df[c].value_counts() for c in CATB}).T.assign(**{"Yes %": lambda t: (t["Yes"] / len(df) * 100).round(1)})
""")

md(r"""
## 2. Target distribution
""")
code(r"""
fig, ax = plt.subplots(figsize=(6.5, 4.6))
vc = df[TARGET].value_counts().reindex(ORDER)
bars = ax.bar(ORDER, vc.values, color=[PAL[c] for c in ORDER], width=0.55)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 80, f"{v:,}\n({v/len(df)*100:.2f}%)", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, vc.max() * 1.18); ax.set_ylabel("Students"); ax.set_xlabel("PlacementStatus")
ax.set_title("Target distribution (modelling dataset, n = %s)" % f"{len(df):,}"); ax.grid(axis="x", visible=False)
save(fig, "eda_01_target_distribution")
ratio = vc["NotPlaced"] / vc["Placed"]
print(f"NotPlaced {vc['NotPlaced']:,} ({vc['NotPlaced']/len(df)*100:.2f}%)  vs  Placed {vc['Placed']:,} ({vc['Placed']/len(df)*100:.2f}%)  -> ratio {ratio:.2f}:1")
add("target", "imbalance_ratio_notplaced_to_placed", round(ratio, 3))
""")
md(r"""
**Class balance:** about 58.4% NotPlaced vs 41.6% Placed (roughly 1.4 : 1). This is a moderate imbalance, not a severe one;
accuracy alone would still be a poor headline metric later (always predicting `NotPlaced` already scores ~58%), so
threshold-free and probability-based metrics will be preferred in modelling.
""")

md(r"""
## 3. Numerical feature distributions
Scale-type variables get histograms with a KDE overlay (bin width matched to the recording step so integer/0.1-step gaps do
not create artificial spikes). Count-type variables are shown as bar charts of level frequencies - a KDE or an outlier
rule is not meaningful for 3-4 level counts.
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(W, 8.6))
kw = {"CGPA": dict(binwidth=0.1), "SoftSkillsRating": dict(binwidth=0.1),
      "AptitudeTestScore": dict(discrete=True), "SSC_Marks": dict(discrete=True), "HSC_Marks": dict(discrete=True)}
for ax, c in zip(axes.ravel(), CONT):
    sns.histplot(df[c], kde=True, color=NEUTRAL, edgecolor="white", linewidth=0.4, ax=ax, **kw[c])
    ax.axvline(df[c].median(), color="#1d2733", ls="--", lw=1)
    ax.text(df[c].median(), ax.get_ylim()[1] * 0.97, f" median {df[c].median():g}", fontsize=8.5, va="top")
    ax.set_title(f"{c}  (skew {df[c].skew():.2f})"); ax.set_xlabel(c); ax.set_ylabel("Students")
ax = axes.ravel()[-1]; ax.axis("off")
ax.text(0, 0.5, "Count-type variables (Internships,\nProjects, Workshops/Certifications)\nare shown in the next figure.\n\nAll scale-type variables are bounded\nwith no long tails: no values are flagged\nas errors or removed.", fontsize=10.5, va="center", color="#1d2733")
fig.suptitle("Distributions of scale-type numeric predictors", y=1.0, fontsize=13, fontweight="semibold")
save(fig, "eda_02_scale_distributions")

fig, axes = plt.subplots(1, 3, figsize=(W, 4.6))
for ax, c in zip(axes, COUNT):
    vc_ = df[c].value_counts().sort_index()
    ax.bar(vc_.index.astype(str), vc_.values, color=NEUTRAL, width=0.62)
    for i, v in enumerate(vc_.values):
        ax.text(i, v + vc_.max() * 0.015, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, vc_.max() * 1.22); ax.set_title(c); ax.set_xlabel("Level"); ax.set_ylabel("Students"); ax.grid(axis="x", visible=False)
fig.suptitle("Frequency of count-type predictors", y=1.02, fontsize=13, fontweight="semibold")
save(fig, "eda_02b_count_distributions")
""")
code(r"""
dist = df[NUM].agg(["count", "mean", "median", "std", "min", "max", "skew", "kurt"]).T
dist["n_unique"] = df[NUM].nunique()
dist["modal_share_%"] = [df[c].value_counts(normalize=True).iloc[0] * 100 for c in NUM]
dist.round(3)
""")

md(r"""
## 4. Placement vs numerical features
Violin plots (with inner box) for the scale-type variables; effect size is summarised as Cohen's *d* (Placed minus NotPlaced,
in pooled-SD units). *d* here is a **descriptive** effect size; formal testing is deferred to Phase 3.
""")
code(r"""
rows = []
for c in NUM:
    a, b = df.loc[y == 1, c], df.loc[y == 0, c]
    rows.append({"feature": c, "mean_placed": a.mean(), "mean_notplaced": b.mean(), "median_placed": a.median(),
                 "median_notplaced": b.median(), "sd_placed": a.std(), "sd_notplaced": b.std(), "cohens_d": cohens_d(a, b)})
    add("numeric_vs_target", f"{c}: cohens_d", round(cohens_d(a, b), 3), "Placed minus NotPlaced, pooled SD")
    add("numeric_vs_target", f"{c}: mean_placed", round(a.mean(), 3)); add("numeric_vs_target", f"{c}: mean_notplaced", round(b.mean(), 3))
cmp = pd.DataFrame(rows).set_index("feature")
cmp.round(3)
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(W, 9))
for ax, c in zip(axes.ravel(), CONT):
    sns.violinplot(data=df, x=TARGET, y=c, order=ORDER, hue=TARGET, hue_order=ORDER, palette=PAL, inner="box", cut=0,
                   linewidth=1, saturation=1, legend=False, ax=ax)
    for i, g in enumerate(ORDER):
        m = df.loc[df[TARGET] == g, c].mean()
        ax.scatter(i, m, marker="D", s=34, color="white", edgecolor="#1d2733", zorder=5)
    ax.set_title(f"{c}  (d = {cmp.loc[c, 'cohens_d']:+.2f})"); ax.set_xlabel(""); ax.set_ylabel(c)
ax = axes.ravel()[-1]
cm = cmp.cohens_d.sort_values()
ax.barh(cm.index, cm.values, color=NEUTRAL, height=0.6)
for i, v in enumerate(cm.values): ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlim(0, cm.max() * 1.18); ax.set_xlabel("Cohen's d (Placed - NotPlaced)"); ax.set_title("Standardised mean difference, all 8 numeric"); ax.grid(axis="y", visible=False)
fig.suptitle("Scale-type predictors by PlacementStatus  (diamond = mean, inner box = IQR/median)", y=1.0, fontsize=13, fontweight="semibold")
save(fig, "eda_03_scale_vs_placement")
""")
md(r"""
### Count-type predictors: placement rate per level
Whiskers are Wilson 95% intervals - they show how much sampling noise is attached to each rate (very wide when *n* is small).
No monotonic pattern is assumed.
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(W, 5))
count_tabs = {}
for ax, c in zip(axes, COUNT):
    count_tabs[c] = rate_bar(ax, df[c], c, xlabel=f"{c} (count)")
fig.suptitle("Placement rate by count-type predictor", y=1.02, fontsize=13, fontweight="semibold")
save(fig, "eda_04_count_vs_placement")
for c, t in count_tabs.items():
    print(f"--- {c} ---"); print((t.assign(rate=t.rate * 100, ci_lo=t.ci_lo * 100, ci_hi=t.ci_hi * 100)).round(1).to_string(), "\n")
    for lvl, r in t.iterrows():
        add("count_rate", f"{c}={lvl}: placement_rate_%", round(r.rate * 100, 2), f"n={int(r.n)}")
""")

md(r"""
### Investigating the non-monotonic patterns
Placement rate does **not** rise steadily with Internships or Workshops (e.g. Workshops = 1 is *below* Workshops = 0, and 3 is
below 2). Three descriptive checks look for an explanation:
1. Do students at different levels differ in their *other* characteristics (a composition effect)?
2. Do the patterns hold inside combinations of count variables?
3. Does the Workshops pattern persist within bands of aptitude (an exploratory split, used only here)?
""")
code(r"""
prof = df.groupby("Workshops/Certifications")[["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks", "Internships", "Projects"]].mean()
prof["Extracurricular=Yes %"] = df.groupby("Workshops/Certifications")["ExtracurricularActivities"].apply(lambda s: (s == "Yes").mean() * 100)
prof["PlacementTraining=Yes %"] = df.groupby("Workshops/Certifications")["PlacementTraining"].apply(lambda s: (s == "Yes").mean() * 100)
prof["n"] = df["Workshops/Certifications"].value_counts().sort_index()
prof["placement_rate_%"] = count_tabs["Workshops/Certifications"].rate * 100
print("Average profile of students at each Workshops/Certifications level:")
display(prof.round(2))
prof2 = df.groupby("Projects")[["CGPA", "AptitudeTestScore", "SSC_Marks", "HSC_Marks"]].mean().round(2)
prof2["n"] = df["Projects"].value_counts().sort_index(); prof2["placement_rate_%"] = (count_tabs["Projects"].rate * 100).round(1)
print("Average profile of students at each Projects level:"); display(prof2)
""")
code(r"""
def rate_heat(ax, rows, cols, title):
    r = y.groupby([df[rows], df[cols]]).mean().unstack(); n = y.groupby([df[rows], df[cols]]).size().unstack()
    ann = r.copy().astype(object)
    for i in r.index:
        for j in r.columns:
            nn = n.loc[i, j]; ann.loc[i, j] = "" if pd.isna(nn) else f"{r.loc[i, j]*100:.0f}%\nn={int(nn)}" + (" †" if nn < 30 else "")
    sns.heatmap(r * 100, annot=ann, fmt="", cmap="Blues", vmin=0, vmax=100, cbar_kws={"label": "Placement rate (%)"}, linewidths=1,
                linecolor="white", annot_kws={"fontsize": 8.5}, ax=ax)
    ax.set_title(title); ax.set_xlabel(cols); ax.set_ylabel(rows); ax.invert_yaxis()
    return r, n

fig, axes = plt.subplots(1, 3, figsize=(W + 1, 5.2))
rate_heat(axes[0], "Projects", "Workshops/Certifications", "Projects x Workshops")
rate_heat(axes[1], "Internships", "Projects", "Internships x Projects")
# exploratory aptitude bands (visualisation only; temporary variable, never stored in df)
apt_band = pd.qcut(df["AptitudeTestScore"], 3, labels=["Low aptitude", "Mid aptitude", "High aptitude"], duplicates="drop")
assert apt_band.nunique() == 3
t = y.groupby([df["Workshops/Certifications"], apt_band], observed=True).agg(["mean", "size"]).reset_index()
shade = {"Low aptitude": "#9ecae1", "Mid aptitude": "#4292c6", "High aptitude": "#08306b"}
for band in shade:
    s = t[t.AptitudeTestScore == band] if "AptitudeTestScore" in t else None
    axes[2].plot(s["Workshops/Certifications"], s["mean"] * 100, marker="o", color=shade[band], lw=2, label=band)
    axes[2].text(s["Workshops/Certifications"].iloc[-1] + 0.06, s["mean"].iloc[-1] * 100, band.split()[0], fontsize=8.5, va="center")
axes[2].set_xticks([0, 1, 2, 3]); axes[2].set_xlim(-0.2, 3.6); axes[2].set_ylim(0, 100)
axes[2].set_xlabel("Workshops/Certifications"); axes[2].set_ylabel("Placement rate (%)")
axes[2].set_title("Workshops vs placement within aptitude bands\n(exploratory terciles, visualisation only)"); axes[2].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Non-monotonic count patterns: composition and stratified views († = n < 30, treat with caution)", y=1.03, fontsize=13, fontweight="semibold")
save(fig, "eda_04b_nonmonotonic_investigation")
strat = t.pivot(index="Workshops/Certifications", columns="AptitudeTestScore", values="mean").mul(100).round(1)
strat_n = t.pivot(index="Workshops/Certifications", columns="AptitudeTestScore", values="size")
print("Placement rate (%) of Workshops level within aptitude tercile:"); display(strat); print("n per cell:"); display(strat_n)
for lvl in strat.index:
    for band in strat.columns:
        add("workshops_within_aptitude_band", f"workshops={lvl}, {band}: rate_%", strat.loc[lvl, band], f"n={int(strat_n.loc[lvl, band])}")
""")
md(r"""
**Reading these three views (descriptive only).** The students with 2 Workshops/Certifications also have noticeably higher
average CGPA, aptitude and project counts and are far more likely to report extracurriculars and placement training than
students with 0-1 workshops (first table). So the Workshops level is entangled with an overall "stronger profile" pattern,
and its raw placement rate partly reflects who sits at each level. The within-aptitude view shows whether the dip at level 1
survives once aptitude is held roughly constant. Cells marked † (e.g. Projects = 0, n = 30) rest on very few students and should
not be interpreted individually. Whether these relationships hold up statistically is a Phase 3 question.
""")

md("## 5. Categorical features")
code(r"""
for c in CATB:
    t = rate_table(df[c])
    print(f"--- {c} ---"); print(t.assign(share_pct=t.n / len(df) * 100, rate=t.rate * 100, ci_lo=t.ci_lo * 100, ci_hi=t.ci_hi * 100).round(1).to_string(), "\n")
    for lvl, r in t.iterrows():
        add("categorical_rate", f"{c}={lvl}: placement_rate_%", round(r.rate * 100, 2), f"n={int(r.n)}")
combo = df["ExtracurricularActivities"].map({"Yes": "Extra: Yes", "No": "Extra: No"}) + " / " + df["PlacementTraining"].map({"Yes": "Train: Yes", "No": "Train: No"})
combo = pd.Categorical(combo, ["Extra: No / Train: No", "Extra: No / Train: Yes", "Extra: Yes / Train: No", "Extra: Yes / Train: Yes"])
fig, axes = plt.subplots(1, 3, figsize=(W, 5), gridspec_kw={"width_ratios": [1, 1, 1.9]})
rate_bar(axes[0], df["ExtracurricularActivities"], "ExtracurricularActivities", rot=0)
rate_bar(axes[1], df["PlacementTraining"], "PlacementTraining", rot=0)
tc = rate_bar(axes[2], pd.Series(combo, index=df.index), "Combination of both", rot=0)
axes[2].set_xticklabels([l.get_text().replace(" / ", "\n") for l in axes[2].get_xticklabels()], fontsize=9)
fig.suptitle("Placement rate by categorical predictors (whiskers: Wilson 95% CI; dashed: overall rate)", y=1.02, fontsize=13, fontweight="semibold")
save(fig, "eda_05_categorical_vs_placement")
for lvl, r in tc.iterrows(): add("categorical_rate", f"combo {lvl}: placement_rate_%", round(r.rate * 100, 2), f"n={int(r.n)}")
print("Association between the two binary predictors themselves:")
display(pd.crosstab(df["ExtracurricularActivities"], df["PlacementTraining"], normalize="index").mul(100).round(1))
""")

md(r"""
## 6. Correlation analysis
Pearson correlations are computed first, with Spearman rank correlations alongside as a check.
**Correlation measures association, not causation**: a positive correlation between a feature and placement does not show that
the feature causes placement - both may reflect shared underlying factors (e.g. overall academic strength).
**Limitation of Pearson:** it measures *linear* association and assumes interval-scaled variables. Several predictors here
are discrete or ordinal (3-4 levels) and Phase 1 / section 4 showed non-monotonic relationships, so Pearson can understate or
misrepresent them (a U- or step-shaped relationship can have a near-zero *r*). The correlation with a 0/1 target is the
point-biserial correlation, which is simply Pearson on the encoded target.
""")
code(r"""
corr = df[NUM].corr(method="pearson")
fig, ax = plt.subplots(figsize=(8.6, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, linewidths=1, linecolor="white",
            cbar_kws={"label": "Pearson r", "shrink": 0.8}, ax=ax)
ax.set_title("Pearson correlation among numeric predictors"); ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
save(fig, "eda_06_correlation_matrix")
pr = df[NUM].corrwith(y, method="pearson"); sp = df[NUM].corrwith(y, method="spearman")
ct = pd.DataFrame({"pearson_r (point-biserial)": pr, "spearman_rho": sp}).sort_values("pearson_r (point-biserial)")
ct.round(3)
""")
code(r"""
fig, ax = plt.subplots(figsize=(8.6, 5.2))
yy = np.arange(len(ct)); h = 0.38
ax.barh(yy + h / 2, ct.iloc[:, 0], height=h, color=PLACED_C, label="Pearson r")
ax.barh(yy - h / 2, ct.iloc[:, 1], height=h, color=NEUTRAL, label="Spearman ρ")
for i, (a, b) in enumerate(zip(ct.iloc[:, 0], ct.iloc[:, 1])):
    ax.text(a + 0.008, i + h / 2, f"{a:.2f}", va="center", fontsize=8.5); ax.text(b + 0.008, i - h / 2, f"{b:.2f}", va="center", fontsize=8.5)
ax.set_yticks(yy); ax.set_yticklabels(ct.index); ax.set_xlim(0, max(ct.max()) * 1.15); ax.legend(frameon=False, loc="lower right")
ax.set_xlabel("Correlation with Placed (1) vs NotPlaced (0)"); ax.set_title("Association of numeric predictors with placement"); ax.grid(axis="y", visible=False)
save(fig, "eda_06b_correlation_with_target")
for c in NUM:
    add("correlation_with_target", f"{c}: pearson_r", round(pr[c], 3)); add("correlation_with_target", f"{c}: spearman_rho", round(sp[c], 3))
upper = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(ascending=False)
print("Strongest predictor-predictor correlations:"); print(upper.head(6).round(3).to_string())
add("correlation_predictors", "max_abs_pairwise_pearson", round(upper.abs().max(), 3), upper.abs().idxmax().__str__())
""")

md(r"""
## 7. Multivariate views
A few targeted plots rather than a full pair-plot. Scores are recorded in coarse steps, so points overlap heavily; a small
jitter (display only) and transparency are used, and a binned placement-rate map (exploratory bins) shows the joint pattern
more reliably than raw dots.
""")
code(r"""
rng = np.random.default_rng(42)
def jit(s, scale): return s + rng.uniform(-scale, scale, len(s))
pairs = [("CGPA", "AptitudeTestScore", 0.03, 0.4), ("CGPA", "HSC_Marks", 0.03, 0.4), ("AptitudeTestScore", "SoftSkillsRating", 0.4, 0.03)]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
for ax, (xc, yc, jx, jy) in zip(axes.ravel(), pairs):
    for g in ORDER:
        m = df[TARGET] == g
        ax.scatter(jit(df.loc[m, xc], jx), jit(df.loc[m, yc], jy), s=9, alpha=0.30, color=PAL[g], label=g, linewidths=0)
    ax.set_xlabel(xc); ax.set_ylabel(yc); ax.set_title(f"{xc} vs {yc}")
    leg = ax.legend(frameon=False, markerscale=2.5, loc="lower right"); [h.set_alpha(1) for h in leg.legend_handles]
# exploratory binned rate map (temporary bins)
cg = pd.qcut(df["CGPA"], 4, duplicates="drop"); ap = pd.qcut(df["AptitudeTestScore"], 4, duplicates="drop")
r = y.groupby([ap, cg], observed=True).mean().unstack(); n = y.groupby([ap, cg], observed=True).size().unstack()
ann = (r * 100).round(0).astype(int).astype(str) + "%\nn=" + n.astype(str)
r.index = [str(i) for i in r.index]; r.columns = [str(i) for i in r.columns]; ann.index, ann.columns = r.index, r.columns
sns.heatmap(r * 100, annot=ann, fmt="", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white", annot_kws={"fontsize": 8.5},
            cbar_kws={"label": "Placement rate (%)"}, ax=axes[1, 1])
axes[1, 1].invert_yaxis(); axes[1, 1].set_xlabel("CGPA quartile bin (exploratory)"); axes[1, 1].set_ylabel("Aptitude quartile bin (exploratory)")
axes[1, 1].set_title("Placement rate by CGPA x Aptitude bins"); axes[1, 1].tick_params(axis="x", rotation=20); axes[1, 1].tick_params(axis="y", rotation=0)
fig.suptitle("Joint behaviour of the strongest numeric predictors", y=1.0, fontsize=13, fontweight="semibold")
save(fig, "eda_07_multivariate")
print("Binned rates (%):"); display((r * 100).round(1))
add("multivariate", "rate_lowest_cgpa_x_lowest_aptitude_%", round(r.iloc[0, 0] * 100, 1)); add("multivariate", "rate_highest_cgpa_x_highest_aptitude_%", round(r.iloc[-1, -1] * 100, 1))
""")

md(r"""
## 8. Placement-rate ranking
**Exploratory bins notice:** the quantile bins used below exist only for these charts. They are held in temporary variables,
are not saved, and are **not** model features; the dataframe columns are asserted unchanged at the end of the notebook.
""")
code(r"""
parts = []
for c in ["Internships", "Projects", "Workshops/Certifications", "ExtracurricularActivities", "PlacementTraining"]:
    t = rate_table(df[c]).reset_index().rename(columns={c: "level"}); t.insert(0, "variable", c); t["level"] = t["level"].astype(str); parts.append(t)
rank = pd.concat(parts).sort_values("rate", ascending=False).reset_index(drop=True)
rank["label"] = rank.variable.str.replace("Workshops/Certifications", "Workshops") + " = " + rank.level + "  (n=" + rank.n.astype(str) + ")"
tbl = rank[["variable", "level", "n", "placed", "rate", "ci_lo", "ci_hi"]].copy()
for c in ["rate", "ci_lo", "ci_hi"]: tbl[c] = (tbl[c] * 100).round(1)
tbl.index = tbl.index + 1; tbl.index.name = "rank"
display(tbl)
fig, ax = plt.subplots(figsize=(W * 0.62, 7.4))
yy = np.arange(len(rank))[::-1]
ax.barh(yy, rank.rate * 100, color=np.where(rank.n < 100, "#9ecae1", PLACED_C), height=0.62)
ax.errorbar(rank.rate * 100, yy, xerr=[(rank.rate - rank.ci_lo) * 100, (rank.ci_hi - rank.rate) * 100], fmt="none", ecolor="#1d2733", lw=1, capsize=2.5)
for yi, r_, hi in zip(yy, rank.rate, rank.ci_hi): ax.text(hi * 100 + 1.2, yi, f"{r_*100:.1f}%", va="center", fontsize=8.5)
ax.axvline(OVERALL * 100, color=NEUTRAL, ls="--", lw=1); ax.text(OVERALL * 100 + 0.8, -0.95, f"overall {OVERALL*100:.1f}%", fontsize=8, color=NEUTRAL)
ax.set_yticks(yy); ax.set_yticklabels(rank.label, fontsize=9); ax.set_xlim(0, 108); ax.set_xlabel("Placement rate (%)  - whiskers: Wilson 95% CI")
ax.set_title("Placement rate by category, ranked  (light bars: n < 100)"); ax.grid(axis="y", visible=False)
save(fig, "eda_08_rate_ranking")
""")
code(r"""
# Exploratory quantile bins for continuous-scale variables (VISUALISATION ONLY - never added to df)
fig, axes = plt.subplots(2, 3, figsize=(W, 9))
for ax, c in zip(axes.ravel(), CONT):
    b = pd.qcut(df[c], 5, duplicates="drop")
    t = rate_table(b); lab = [f"{i.left:g}-{i.right:g}" for i in t.index]; x = np.arange(len(t))
    ax.errorbar(x, t.rate * 100, yerr=[(t.rate - t.ci_lo) * 100, (t.ci_hi - t.rate) * 100], fmt="o-", color=PLACED_C, ecolor="#1d2733", capsize=3, lw=2, ms=6)
    for xi, (r_, n_) in enumerate(zip(t.rate, t.n)): ax.text(xi, r_ * 100 - 7, f"n={n_:,}", ha="center", fontsize=8, color="#1d2733")
    ax.axhline(OVERALL * 100, color=NEUTRAL, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=20, ha="right"); ax.set_ylim(0, 100)
    ax.set_xlabel(f"{c} (exploratory quantile bins)"); ax.set_ylabel("Placement rate (%)"); ax.set_title(c)
    for i, r_ in enumerate(t.rate): add("binned_rate_exploratory", f"{c} bin {lab[i]}: rate_%", round(r_ * 100, 1), f"n={int(t.n.iloc[i])}")
ax = axes.ravel()[-1]; ax.axis("off")
ax.text(0, 0.5, "Bins are quantile-based and used only\nto draw these charts. They are NOT\nmodel features; models will use the\noriginal numeric columns.\n\nDashed line = overall placement rate.", fontsize=10.5, va="center")
fig.suptitle("Placement rate across exploratory bins of scale-type predictors", y=1.0, fontsize=13, fontweight="semibold")
save(fig, "eda_08b_binned_rates")
""")

md(r"""
## 9. Train / validation / test sanity check
Purpose: confirm the stratified split is representative. Differences of a few percent between splits of this size are expected
from sampling alone; the KS statistic (max gap between empirical CDFs) and percentage-point gaps quantify any shift.
""")
code(r"""
chk = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "Internships", "Projects"]
names = ["train", "validation", "test"]
rows = []
for s, d in splits.items():
    r_ = {"split": s, "n": len(d), "Placed_%": (d[TARGET] == "Placed").mean() * 100}
    for c in chk: r_[f"{c} mean"] = d[c].mean(); r_[f"{c} sd"] = d[c].std()
    rows.append(r_)
st = pd.DataFrame(rows).set_index("split"); display(st.round(3))
for s in names:
    add("split_check", f"{s}: n", len(splits[s])); add("split_check", f"{s}: placed_%", round(st.loc[s, "Placed_%"], 2))
    for c in chk: add("split_check", f"{s}: {c} mean", round(st.loc[s, f"{c} mean"], 3))

res = []
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    for c in ["CGPA", "AptitudeTestScore", "SoftSkillsRating"]:
        k = stats.ks_2samp(splits[a][c], splits[b][c]); res.append((f"{a} vs {b}", c, "KS statistic", k.statistic, k.pvalue))
    for c in ["Internships", "Projects", TARGET]:
        pa = splits[a][c].value_counts(normalize=True); pb = splits[b][c].value_counts(normalize=True)
        gap = (pa.sub(pb, fill_value=0).abs().max()) * 100
        ct_ = pd.concat([splits[a][c].value_counts(), splits[b][c].value_counts()], axis=1).fillna(0)
        res.append((f"{a} vs {b}", c, "max |pp gap|", gap, stats.chi2_contingency(ct_)[1]))
shift = pd.DataFrame(res, columns=["pair", "variable", "measure", "value", "p_value"])
shift_wide = shift.pivot(index=["variable", "measure"], columns="pair", values="value").round(3)
display(shift_wide)
print("max KS statistic:", round(shift[shift.measure == "KS statistic"].value.max(), 3), "| max percentage-point gap:", round(shift[shift.measure == "max |pp gap|"].value.max(), 2))
print("Smallest p-value across the 18 comparisons:", round(shift.p_value.min(), 3), "(with 18 looks, one or two p < .05 would arise by chance alone)")
add("split_check", "max_KS_statistic", round(shift[shift.measure == "KS statistic"].value.max(), 4))
add("split_check", "max_pct_point_gap_discrete", round(shift[shift.measure == "max |pp gap|"].value.max(), 3))
add("split_check", "min_p_value_of_18", round(shift.p_value.min(), 4))
""")
code(r"""
SPC = {"train": "#4d4d4d", "validation": "#9a6fb0", "test": "#2a9d8f"}
fig, axes = plt.subplots(2, 3, figsize=(W, 8.6))
for ax, c in zip(axes[0], ["CGPA", "AptitudeTestScore", "SoftSkillsRating"]):
    for s in names:
        v = np.sort(splits[s][c].values); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=SPC[s], lw=1.6, label=s)
    ax.set_xlabel(c); ax.set_ylabel("Cumulative share"); ax.set_title(f"{c}: empirical CDF by split"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
def grouped(ax, series_fn, title, xlabel):
    levels = sorted(series_fn(splits["train"]).unique()); w = 0.26
    for k, s in enumerate(names):
        p = series_fn(splits[s]).value_counts(normalize=True).reindex(levels).fillna(0) * 100
        ax.bar(np.arange(len(levels)) + (k - 1) * w, p.values, width=w, color=SPC[s], label=s)
    ax.set_xticks(np.arange(len(levels))); ax.set_xticklabels(levels); ax.set_xlabel(xlabel); ax.set_ylabel("% of split"); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="x", visible=False)
grouped(axes[1, 0], lambda d: d["Internships"], "Internships: share by split", "Internships")
grouped(axes[1, 1], lambda d: d["Projects"], "Projects: share by split", "Projects")
grouped(axes[1, 2], lambda d: d[TARGET], "PlacementStatus: share by split", "PlacementStatus")
fig.suptitle("Train / validation / test representativeness", y=1.0, fontsize=13, fontweight="semibold")
save(fig, "eda_09_split_sanity")
""")

md(r"""
## 10. Preparation for formal statistical testing (Phase 3)
No inferential tests are run here. The table lists the relationships that look worth testing, the candidate test, and why that
test suits the variable types. Observed numbers are filled in from the analysis above.
""")
code(r"""
T = {k: rate_table(df[k]) for k in ["Internships", "Projects", "Workshops/Certifications", "ExtracurricularActivities", "PlacementTraining"]}
pct = lambda t, lvl: f"{t.loc[lvl, 'rate']*100:.1f}%"
d_ = cmp.cohens_d
plan = pd.DataFrame([
 ("CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks vs placement",
  f"Placed students score higher on all five; Cohen's d from {d_[CONT].min():.2f} ({d_[CONT].idxmin()}) to {d_[CONT].max():.2f} ({d_[CONT].idxmax()})",
  "Mann-Whitney U (primary) + Welch t-test (sensitivity); effect size: Cohen's d and rank-biserial / AUC",
  "Scores are bounded, mildly skewed and recorded in coarse steps with many ties, so a rank-based test avoids normality assumptions; Welch handles unequal variances. With n~10k, p-values will be tiny, so effect size is what matters."),
 ("ExtracurricularActivities vs placement",
  f"Placement rate {pct(T['ExtracurricularActivities'],'Yes')} (Yes) vs {pct(T['ExtracurricularActivities'],'No')} (No)",
  "Chi-square test of independence (2x2); effect size: odds ratio with CI and phi",
  "Two binary variables; expected cell counts are large, so the chi-square approximation is valid. Odds ratio gives an interpretable effect size."),
 ("PlacementTraining vs placement",
  f"Placement rate {pct(T['PlacementTraining'],'Yes')} (Yes) vs {pct(T['PlacementTraining'],'No')} (No)",
  "Chi-square test of independence (2x2); odds ratio with CI",
  "Same reasoning as above. Also check its overlap with ExtracurricularActivities (the two co-vary)."),
 ("Internships vs placement (non-monotonic)",
  f"Rates {pct(T['Internships'],0)} / {pct(T['Internships'],1)} / {pct(T['Internships'],2)} for 0 / 1 / 2: flat between 0 and 1, jump at 2",
  "Chi-square (3x2) for overall association; two-proportion z-test for 0 vs 1; do NOT rely on a linear trend test (Cochran-Armitage)",
  "A trend test presumes ordered, monotonic change, which the data do not show. The 0-vs-1 contrast directly tests the flat segment."),
 ("Projects vs placement (non-monotonic, sparse level 0)",
  f"Rates {pct(T['Projects'],0)} / {pct(T['Projects'],1)} / {pct(T['Projects'],2)} / {pct(T['Projects'],3)} for 0-3; level 0 has only n={int(T['Projects'].loc[0,'n'])}",
  "Chi-square on levels 1-3; Fisher's exact (or exact CI) for the Projects = 0 group",
  "Level 0 is too small for chi-square validity (expected counts) and its rate is highly uncertain; exact methods are appropriate."),
 ("Workshops/Certifications vs placement (non-monotonic)",
  f"Rates {pct(T['Workshops/Certifications'],0)} / {pct(T['Workshops/Certifications'],1)} / {pct(T['Workshops/Certifications'],2)} / {pct(T['Workshops/Certifications'],3)} for 0-3; level 1 below level 0",
  "Chi-square (4x2) overall; pairwise two-proportion tests with multiple-comparison correction (Holm)",
  "Pairwise comparisons identify where the pattern departs from monotonic; correction is needed since several pairs are compared."),
 ("Is the Workshops/Projects pattern confounded by overall profile strength?",
  "Higher-workshop groups also have higher CGPA/aptitude and more extracurricular/training; patterns can differ within aptitude bands",
  "Cochran-Mantel-Haenszel test stratified by aptitude band; later a multivariable logistic regression (inference form)",
  "Stratified testing asks whether the association persists after holding another variable roughly fixed; a necessary check before interpreting a raw rate."),
 ("Multicollinearity among scores",
  f"Strongest predictor-predictor Pearson r = {upper.abs().max():.2f} ({upper.abs().idxmax()[0]} / {upper.abs().idxmax()[1]})",
  "Spearman correlations and variance inflation factors",
  "Rank correlation is robust for bounded, discrete scores; VIF quantifies redundancy relevant to linear models."),
], columns=["Feature / relationship", "Observed pattern", "Candidate statistical test", "Reason"])
pd.set_option("display.max_colwidth", 200)
plan.to_csv(MET / "03_phase3_test_plan.csv", index=False)
plan
""")

md("## 11. EDA findings")
findings = Path("src/eda_findings.md").read_text(encoding="utf-8") if Path("src/eda_findings.md").exists() else "_Findings pending._"
md(findings)

md("## 12. Save the numeric summary and final checks")
code(r"""
summary = pd.DataFrame(SUMMARY)
summary.to_csv(MET / "03_eda_summary.csv", index=False)
print("Saved", MET / "03_eda_summary.csv", summary.shape)
# no engineered/binned feature leaked into the data; no model was trained
assert list(df.columns) == list(pd.read_csv(PROC / "modeling_dataset.csv").columns) and df.shape == (9912, 11)
assert not any(k in globals() for k in ("model", "clf", "LogisticRegression"))
print("Figures written:", len(list(FIGDIR.glob('*.png'))), "->", FIGDIR)
summary.head(15)
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/03_eda.ipynb")
print("wrote notebooks/03_eda.ipynb")


