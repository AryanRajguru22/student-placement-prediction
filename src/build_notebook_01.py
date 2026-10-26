"""Builds notebooks/01_data_understanding.ipynb (run once, then execute the notebook)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 - Data Understanding
**Project:** Predicting Student Placement Probability Using Academic, Technical and Employability Features

**Phase scope:** data understanding only. No feature engineering, no modelling, no rows removed,
raw files are never modified. A processed copy (with audit flags only) is written to `data/processed/`.

**Source:** [Krooz/Campus_Recruitment_CSV](https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV)
(`raw_datasets/train.csv`, `validation.csv`, `test.csv`; licence CC0-1.0).
""")

code("""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIG, MET = ROOT / "results" / "figures", ROOT / "results" / "metrics"
for p in (PROC, FIG, MET):
    p.mkdir(parents=True, exist_ok=True)
TARGET = "PlacementStatus"
""")

md("## 1. Load the raw data\nThe Hugging Face repo ships three pre-made splits. They are loaded as-is and stacked only for inspection; a `split` column records provenance.")
code("""
splits = ["train", "validation", "test"]
frames = {}
for s in splits:
    f = RAW / f"{s}.csv"
    print(f"{f.name:16s} sha256={hashlib.sha256(f.read_bytes()).hexdigest()[:16]}...")
    frames[s] = pd.read_csv(f)
df = pd.concat([d.assign(split=s) for s, d in frames.items()], ignore_index=True)
print("\\nSplit shapes:", {s: d.shape for s, d in frames.items()})
print("Identical columns across splits:", all(list(d.columns) == list(frames["train"].columns) for d in frames.values()))
""")

md("## 2. Shape, columns and data types")
code("""
print("Combined shape:", df.shape)
data_cols = [c for c in df.columns if c != "split"]
info = pd.DataFrame({"dtype": df[data_cols].dtypes.astype(str), "non_null": df[data_cols].notna().sum(),
                     "n_unique": df[data_cols].nunique()})
info
""")
code("df.head(10)")

md("## 3. Missing values")
code("""
miss = pd.DataFrame({"missing": df[data_cols].isna().sum(), "pct": (df[data_cols].isna().mean() * 100).round(3)})
print("Total missing cells:", int(miss.missing.sum()))
miss
""")
code("""
# Hidden missingness: blank / whitespace-only strings or placeholder tokens
obj = df[data_cols].select_dtypes(exclude="number")
placeholders = {"", "na", "n/a", "nan", "null", "none", "-", "?", "unknown"}
hidden = {c: int(obj[c].astype(str).str.strip().str.lower().isin(placeholders).sum()) for c in obj}
print("Placeholder-like strings per object column:", hidden)
""")

md("## 4. Duplicate rows\nDuplicates are only *counted and flagged*. Nothing is dropped.")
code("""
dup_all = df.duplicated(subset=data_cols, keep=False)      # every member of a duplicate group
dup_extra = df.duplicated(subset=data_cols, keep="first")  # surplus copies
print(f"Rows belonging to a duplicate group : {dup_all.sum()}")
print(f"Surplus copies (would be dropped)   : {dup_extra.sum()}  ({dup_extra.mean()*100:.2f}%)")
print("\\nSurplus copies by split:\\n", df[dup_extra].split.value_counts().to_string())
print("\\nDuplicates within each split:", {s: int(d.duplicated().sum()) for s, d in frames.items()})

# Cross-split leakage: identical records appearing in more than one split
keys = df[data_cols].astype(str).agg("|".join, axis=1)
n_splits = df.assign(k=keys).groupby("k").split.nunique()
print("\\nDistinct records present in >1 split (potential train/test leakage):", int((n_splits > 1).sum()))
""")
code("""
# Do duplicate groups ever carry conflicting labels? (compare on features only)
feat_cols = [c for c in data_cols if c != TARGET]
g = df.groupby(feat_cols).agg(n=(TARGET, "size"), labels=(TARGET, "nunique")).reset_index()
print("Feature-identical groups with >1 row:", int((g.n > 1).sum()))
print("...of which have conflicting PlacementStatus labels:", int(((g.n > 1) & (g.labels > 1)).sum()))
""")

md("## 5. Categorical columns - unique values")
code("""
cat_cols = df[data_cols].select_dtypes(exclude="number").columns.tolist()
for c in cat_cols:
    print(f"--- {c} ---")
    print(df[c].value_counts(dropna=False).to_string(), "\\n")
# Low-cardinality numeric columns that behave like categories/counts
for c in ["Internships", "Projects", "Workshops/Certifications"]:
    print(f"--- {c} (numeric count) ---")
    print(df[c].value_counts(dropna=False).sort_index().to_string(), "\\n")
""")

md("## 6. Descriptive statistics")
code("""
num_cols = df[data_cols].select_dtypes(include="number").columns.tolist()
desc = df[num_cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
desc["skew"] = df[num_cols].skew(); desc["kurtosis"] = df[num_cols].kurt()
desc.round(3)
""")
code("df[cat_cols].describe().T")
code("""
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]):
    sns.histplot(df[c], bins=40, kde=True, ax=ax); ax.set_title(c)
sns.boxplot(data=df[["CGPA", "SoftSkillsRating"]], ax=axes[1, 2]); axes[1, 2].set_title("CGPA vs SoftSkills (boxplot)")
plt.tight_layout(); plt.savefig(FIG / "01_numeric_distributions.png", dpi=150); plt.show()
""")

md("## 7. Target distribution (`PlacementStatus`)")
code("""
tc = df[TARGET].value_counts(dropna=False)
print(pd.DataFrame({"count": tc, "pct": (tc / len(df) * 100).round(2)}))
print("\\nBy split (row %):")
print((pd.crosstab(df.split, df[TARGET], normalize="index") * 100).round(2))
fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.countplot(data=df, x=TARGET, ax=ax); ax.set_title("Target distribution")
for p in ax.patches: ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIG / "01_target_distribution.png", dpi=150); plt.show()
""")

md("## 8. Impossible or suspicious values\nRange checks use only what the dataset card and the data itself justify (marks/scores are percentages-like, CGPA is on a 10-point-style scale). Observed ranges are printed first so the thresholds below are not assumed blindly.")
code("""
print(df[num_cols].agg(["min", "max"]).T)
""")
code("""
checks = {
    "CGPA outside [0, 10]":                  ~df.CGPA.between(0, 10),
    "SSC_Marks outside [0, 100]":            ~df.SSC_Marks.between(0, 100),
    "HSC_Marks outside [0, 100]":            ~df.HSC_Marks.between(0, 100),
    "AptitudeTestScore outside [0, 100]":    ~df.AptitudeTestScore.between(0, 100),
    "SoftSkillsRating outside [0, 5]":       ~df.SoftSkillsRating.between(0, 5),
    "Negative Internships/Projects/Workshops": (df[["Internships", "Projects", "Workshops/Certifications"]] < 0).any(axis=1),
    "Non-integer count columns":             (df[["Internships", "Projects", "Workshops/Certifications"]] % 1 != 0).any(axis=1),
}
rep = pd.Series({k: int(v.sum()) for k, v in checks.items()}, name="n_rows")
rep
""")
code("""
# Statistical outliers (IQR rule, 1.5x) - informational, NOT errors.\n# NB: for 3-4 valued count columns (e.g. Internships, IQR=0) the rule is meaningless and flags every non-modal value.
rows = []
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows.append((c, lo, hi, int(((df[c] < lo) | (df[c] > hi)).sum())))
pd.DataFrame(rows, columns=["column", "lower_fence", "upper_fence", "n_outliers"]).set_index("column").round(2)
""")
code("""
# Plausibility / logical consistency checks (flags only)
consistency = {
    "CGPA == 0 or very low (<5)":                 df.CGPA < 5,
    "SSC or HSC below 40":                        (df.SSC_Marks < 40) | (df.HSC_Marks < 40),
    "Many internships (>=4)":                     df.Internships >= 4,
    "Many projects (>=6)":                        df.Projects >= 6,
}
print(pd.Series({k: int(v.sum()) for k, v in consistency.items()}))
print("\\nValue counts of SoftSkillsRating granularity (decimals):",
      (df.SoftSkillsRating * 10 % 1 != 0).sum(), "rows have >1 decimal place")
""")

md("## 9. Relationships snapshot (descriptive only, not modelling)\nLooking for constant features, leakage-like signals and redundancy.")
code("""
y = (df[TARGET] == "Placed").astype(int)  # temporary numeric view, not stored as a feature
enc = df[num_cols].copy()
for c in ["ExtracurricularActivities", "PlacementTraining"]:
    enc[c] = (df[c] == "Yes").astype(int)
enc["target_placed"] = y
corr = enc.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Correlation (Pearson; binary cats as 0/1)")
plt.tight_layout(); plt.savefig(FIG / "01_correlation_heatmap.png", dpi=150); plt.show()
corr["target_placed"].drop("target_placed").sort_values(key=abs, ascending=False).round(3)
""")
code("""
# Placement rate by categorical / count features
for c in ["ExtracurricularActivities", "PlacementTraining", "Internships", "Projects", "Workshops/Certifications"]:
    t = df.groupby(c)[TARGET].agg(n="size", placed_rate=lambda s: round((s == "Placed").mean(), 3))
    print(f"--- {c} ---"); print(t.to_string(), "\\n")
# Are train/val/test drawn from similar distributions?
for c in ["CGPA", "AptitudeTestScore", "SSC_Marks", "HSC_Marks"]:
    a, b = frames["train"][c], frames["test"][c]
    print(f"KS train vs test  {c:18s} stat={stats.ks_2samp(a, b).statistic:.3f}  p={stats.ks_2samp(a, b).pvalue:.3f}")
""")
code("""
# Class separation per numeric feature
cmp = df.groupby(TARGET)[num_cols].mean().T
cmp["abs_diff_in_sd"] = ((cmp.get("Placed") - cmp.get("NotPlaced")).abs() / df[num_cols].std()).round(3)
cmp.round(3)
""")

md("## 10. Data-quality issue register\nPopulated programmatically from the checks above; interpretation is in the markdown cell after it.")
code("""
issues = []
issues.append(("Missing values", int(df[data_cols].isna().sum().sum())))
issues.append(("Surplus duplicate rows", int(dup_extra.sum())))
issues.append(("Rows in any duplicate group", int(dup_all.sum())))
issues.append(("Records appearing in >1 split", int((n_splits > 1).sum())))
issues += [(k, int(v.sum())) for k, v in checks.items()]
issues.append(("Target imbalance (minority %)", round(float(tc.min() / tc.sum() * 100), 2)))
issue_df = pd.DataFrame(issues, columns=["check", "value"])
issue_df.to_csv(MET / "01_data_quality_summary.csv", index=False)
issue_df
""")

md("## 11. Save a processed copy (no rows removed, no values changed)\nThe processed file is the stacked data plus audit columns: `split`, `is_duplicate_extra` (surplus copy flag) and `is_in_dup_group`. Downstream phases decide what to do with the flags.")
code("""
out = df.copy()
out["is_duplicate_extra"] = dup_extra.values
out["is_in_dup_group"] = dup_all.values
out.to_csv(PROC / "campus_recruitment_combined_flagged.csv", index=False)
desc.round(4).to_csv(MET / "01_descriptive_statistics.csv")
print("Saved:", out.shape, "->", PROC / "campus_recruitment_combined_flagged.csv")
assert len(out) == sum(len(d) for d in frames.values()), "row count changed!"
# raw files untouched check
for s in splits:
    assert pd.read_csv(RAW / f"{s}.csv").equals(frames[s])
print("Raw files unchanged; no rows removed.")
""")

md("""
## 12. Summary of findings
| Item | Finding |
|---|---|
| Shape | 10,000 rows x 11 columns (train 7,225 / validation 1,275 / test 1,500 pre-made splits) |
| Types | 8 numeric (CGPA, SoftSkillsRating float; others int), 3 text (ExtracurricularActivities, PlacementTraining, PlacementStatus) |
| Missing | 0 missing cells; no placeholder strings |
| Duplicates | 72 surplus exact copies (0.72%); 138 rows sit in duplicate groups; 28 distinct records appear in more than one split (leakage risk); 8 feature-identical groups carry conflicting labels |
| Target | NotPlaced 5,803 (58.03%) / Placed 4,197 (41.97%); mild imbalance, identical across splits |
| Range violations | None: every column lies inside its plausible range |
| Observed ranges | CGPA 6.5-9.1, Aptitude 60-90, SoftSkills 3.0-4.8, SSC 55-90, HSC 57-88: all hard-bounded (no tails), so unusually clean |
| Odd patterns | `Projects=0` has only 30 rows; placement rate is non-monotonic in Internships (0:33% vs 1:33% vs 2:70%) and Workshops (0:28%, 1:21%, 2:73%, 3:56%) |
| IQR outliers | Only an artefact for `Internships` (IQR=0); no genuine outliers elsewhere |

**Caveat:** the dataset card does not state whether the data is real or synthetic; the bounded ranges and the very strong,
non-monotonic effects are consistent with synthetic generation, but this is an observation, not a conclusion.
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = "notebooks/01_data_understanding.ipynb"
nbf.write(nb, out)
print("wrote", out)

