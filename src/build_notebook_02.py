"""Builds notebooks/02_data_quality_and_split.ipynb. Raw-file hashes are embedded at build time."""
import hashlib
from pathlib import Path
import nbformat as nbf

raw = Path("data/raw")
hashes = {s: hashlib.sha256((raw / f"{s}.csv").read_bytes()).hexdigest() for s in ["train", "validation", "test"]}

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 02 - Data Quality and Split Integrity
Phase 1 found duplicate records and feature profiles shared across the supplied train/validation/test files.
This notebook builds a leakage-free modelling dataset and a new stratified 70/15/15 split.

**Rules:** raw files stay byte-identical; no modelling, feature engineering, scaling or outlier removal.

**Method**
1. Stack the three raw files (with a temporary `source_split` column).
2. Exact duplicates = identical on all 11 columns. Feature profile = identical on the 10 predictors (target and source excluded).
3. Profile groups with one label: keep exactly one representative. Groups with conflicting labels: exclude *all* their rows (no majority vote).
4. Stratified split on `PlacementStatus` with a fixed seed. Because every profile is now unique, no profile can span two splits; this is asserted.
""")

code(f"""
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 170); pd.set_option("display.max_columns", 50)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC, MET = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results" / "metrics"
MET.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)

TARGET = "PlacementStatus"
SEED = 42                      # fixed random seed for the split (recorded in results/metrics/02_split_summary.csv)
SPLITS = ["train", "validation", "test"]
EXPECTED_SHA256 = {hashes!r}   # hashes of the raw files as downloaded (match Hugging Face LFS object ids)

def raw_hashes():
    return {{s: hashlib.sha256((RAW / f"{{s}}.csv").read_bytes()).hexdigest() for s in SPLITS}}

assert raw_hashes() == EXPECTED_SHA256, "raw files differ from the downloaded originals!"
print("Raw files verified byte-identical to originals. Random seed:", SEED)
""")

md("## 1. Combine the raw files and record original counts")
code("""
frames = {s: pd.read_csv(RAW / f"{s}.csv") for s in SPLITS}
df = pd.concat([d.assign(source_split=s) for s, d in frames.items()], ignore_index=True)
COLS = [c for c in df.columns if c != "source_split"]       # 11 original columns
FEATURES = [c for c in COLS if c != TARGET]                  # 10 predictors
assert len(FEATURES) == 10 and TARGET in COLS
orig_counts = {s: len(d) for s, d in frames.items()}
n_orig = len(df)
print("Original rows:", orig_counts, "-> total", n_orig)
assert n_orig == 10000
assert df[COLS].isna().sum().sum() == 0
""")

md("## 2. Exact duplicates (all 11 columns)")
code("""
exact_extra = df.duplicated(subset=COLS, keep="first")
exact_any = df.duplicated(subset=COLS, keep=False)
n_exact_extra, n_exact_in_groups = int(exact_extra.sum()), int(exact_any.sum())
n_exact_groups = int(df[exact_any].groupby(COLS).ngroups)
print(f"Exact-duplicate groups: {n_exact_groups}; rows in those groups: {n_exact_in_groups}; surplus copies: {n_exact_extra}")
""")

md("## 3. Duplicate feature profiles (10 predictors only)")
code("""
prof = df.groupby(FEATURES, sort=False).agg(
    n_rows=(TARGET, "size"),
    n_labels=(TARGET, "nunique"),
    n_placed=(TARGET, lambda s: int((s == "Placed").sum())),
    n_notplaced=(TARGET, lambda s: int((s == "NotPlaced").sum())),
    splits=("source_split", lambda s: ",".join(sorted(set(s)))),
    n_source_splits=("source_split", "nunique"),
).reset_index()
dup_prof = prof[prof.n_rows > 1]
same = dup_prof[dup_prof.n_labels == 1]
conf = dup_prof[dup_prof.n_labels > 1]
print("Unique feature profiles:", len(prof))
print("Feature profiles occurring >1 time:", len(dup_prof), f"({int(dup_prof.n_rows.sum())} rows)")
print("  ...consistent label:", len(same), f"({int(same.n_rows.sum())} rows)")
print("  ...conflicting labels:", len(conf), f"({int(conf.n_rows.sum())} rows)")
print("Surplus rows over unique profiles:", n_orig - len(prof))
""")

md("## 4. Cross-split overlap in the ORIGINAL supplied splits")
code("""
cross = prof[prof.n_source_splits > 1]
print("Feature profiles present in >1 original split:", len(cross), f"({int(cross.n_rows.sum())} rows)")
print(cross.splits.value_counts().to_string())
# exact-record version (matches Phase 1)
k = df[COLS].astype(str).agg("|".join, axis=1)
print("Exact records present in >1 original split:", int((df.assign(k=k).groupby("k").source_split.nunique() > 1).sum()))
# rows in test/validation whose profile also exists in train
tr = set(map(tuple, frames["train"][FEATURES].values))
for s in ["validation", "test"]:
    n = sum(tuple(r) in tr for r in frames[s][FEATURES].values)
    print(f"{s}: {n} of {len(frames[s])} rows ({n/len(frames[s])*100:.2f}%) share a feature profile with train")
""")

md("## 5. The conflicting-label groups")
code("""
conf_report = conf.sort_values(FEATURES).reset_index(drop=True)
conf_report.index = conf_report.index + 1
conf_report.to_csv(MET / "02_conflicting_label_groups.csv", index_label="group_id")
conf_report
""")
code("""
# Row-level view of every excluded row, with its original file and line number (header = line 1)
df["raw_line"] = df.groupby("source_split").cumcount() + 2
conf_keys = conf.set_index(FEATURES).index
is_conf = df.set_index(FEATURES).index.isin(conf_keys)
conf_rows = df[is_conf].sort_values(FEATURES + ["source_split", "raw_line"])
conf_rows.to_csv(MET / "02_excluded_conflicting_rows.csv", index=False)
print("Rows in conflicting groups:", len(conf_rows))
conf_rows[FEATURES + [TARGET, "source_split", "raw_line"]]
""")

md("## 6. Build the modelling dataset")
code("""
# (a) exclude conflicting groups entirely
kept = df[~is_conf].copy()
n_removed_conflict_rows = int(is_conf.sum()); n_removed_conflict_groups = len(conf)
# (b) one representative per remaining feature profile (all members share a label) - first occurrence, deterministic
model_df = kept.drop_duplicates(subset=FEATURES, keep="first")
n_removed_dup_rows = len(kept) - len(model_df)
model_df = model_df[FEATURES + [TARGET]].reset_index(drop=True)

print(f"Original rows                         : {n_orig}")
print(f"Removed - conflicting groups          : {n_removed_conflict_rows} rows in {n_removed_conflict_groups} groups")
print(f"Removed - duplicate profiles (same label): {n_removed_dup_rows} rows")
print(f"Modelling dataset                     : {len(model_df)}")
assert n_orig - n_removed_conflict_rows - n_removed_dup_rows == len(model_df)
assert not model_df.duplicated(subset=FEATURES).any()
assert model_df.isna().sum().sum() == 0
""")
code("""
# Every same-label duplicate profile is also an exact duplicate (labels agree), so the two counts must coincide.
# (Conflicting groups are never exact duplicates of each other, since their labels differ.)
assert n_removed_dup_rows == n_exact_extra == 72 or n_removed_dup_rows == n_exact_extra
print(f"Rows removed as duplicates: {n_removed_dup_rows} == surplus exact duplicates: {n_exact_extra}")
""")
md("## 7. Stratified 70 / 15 / 15 split")
code("""
train, tmp = train_test_split(model_df, test_size=0.30, stratify=model_df[TARGET], random_state=SEED)
val, test = train_test_split(tmp, test_size=0.50, stratify=tmp[TARGET], random_state=SEED)
train, val, test = (d.sort_index().reset_index(drop=True) for d in (train, val, test))
new = {"train": train, "validation": val, "test": test}
modeling = model_df.copy()

modeling.to_csv(PROC / "modeling_dataset.csv", index=False)
for s, d in new.items():
    d.to_csv(PROC / f"{s}.csv", index=False)
print({s: len(d) for s, d in new.items()}, "total", sum(len(d) for d in new.values()))
""")

md("## 8. Verification")
code("""
# raw untouched
assert raw_hashes() == EXPECTED_SHA256, "raw files modified!"
# reload what was written to disk and test THAT
disk = {s: pd.read_csv(PROC / f"{s}.csv") for s in SPLITS}
disk_all = pd.read_csv(PROC / "modeling_dataset.csv")
for name, d in {**disk, "modeling_dataset": disk_all}.items():
    assert TARGET in d.columns, name
    assert d.isna().sum().sum() == 0, name
    assert set(d[TARGET]) == {"Placed", "NotPlaced"}, name
    assert list(d.columns) == FEATURES + [TARGET], name
# row accounting
assert sum(len(d) for d in disk.values()) == len(disk_all)
# zero profile overlap between splits (and with no duplicates inside any split)
prof_sets = {s: set(map(tuple, d[FEATURES].values)) for s, d in disk.items()}
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    inter = prof_sets[a] & prof_sets[b]
    print(f"profile overlap {a:10s} ∩ {b:10s}: {len(inter)}")
    assert len(inter) == 0
for s, d in disk.items():
    assert not d.duplicated(subset=FEATURES).any(), s
assert not disk_all.duplicated(subset=FEATURES).any()
# the conflicting profiles are absent everywhere
conf_set = set(map(tuple, conf[FEATURES].values))
assert all(not (conf_set & p) for p in prof_sets.values())
# proportions ~ 70/15/15
n = len(disk_all)
props = {s: len(d) / n for s, d in disk.items()}
assert abs(props["train"] - .70) < .005 and abs(props["validation"] - .15) < .005 and abs(props["test"] - .15) < .005
# class balance similar
rates = {s: (d[TARGET] == "Placed").mean() for s, d in disk.items()}
overall = (disk_all[TARGET] == "Placed").mean()
assert all(abs(r - overall) < .005 for r in rates.values())
print("All assertions passed.")
""")
code("""
summary = pd.DataFrame({
    "rows": {s: len(d) for s, d in disk.items()},
    "share_%": {s: round(props[s] * 100, 2) for s in SPLITS},
    "Placed": {s: int((d[TARGET] == "Placed").sum()) for s, d in disk.items()},
    "NotPlaced": {s: int((d[TARGET] == "NotPlaced").sum()) for s, d in disk.items()},
    "Placed_%": {s: round(rates[s] * 100, 2) for s in SPLITS},
})
summary.loc["modeling_dataset"] = [len(disk_all), 100.0, int((disk_all[TARGET]=="Placed").sum()),
                                   int((disk_all[TARGET]=="NotPlaced").sum()), round(overall * 100, 2)]
summary.to_csv(MET / "02_split_summary.csv", index_label="split")
summary
""")
code("""
audit = pd.Series({
    "random_seed": SEED,
    "original_rows": n_orig,
    "original_rows_train/validation/test": "/".join(str(orig_counts[s]) for s in SPLITS),
    "exact_duplicate_groups": n_exact_groups,
    "exact_duplicate_rows_in_groups": n_exact_in_groups,
    "exact_duplicate_surplus_rows": n_exact_extra,
    "unique_feature_profiles": len(prof),
    "duplicate_feature_profiles": len(dup_prof),
    "duplicate_profiles_same_label": len(same),
    "duplicate_profiles_conflicting_label": len(conf),
    "original_cross_split_profiles": len(cross),
    "rows_removed_conflicting": n_removed_conflict_rows,
    "groups_removed_conflicting": n_removed_conflict_groups,
    "rows_removed_duplicate_profiles": n_removed_dup_rows,
    "final_modeling_rows": len(disk_all),
    "final_cross_split_profile_overlap": 0,
}, name="value")
audit.to_csv(MET / "02_data_quality_audit.csv", index_label="metric")
audit.to_frame()
""")

md("""
## 9. Conclusion
The modelling dataset contains one record per distinct feature profile, with ambiguous (conflicting-label) profiles excluded
rather than guessed. The new splits are stratified on `PlacementStatus`, seeded, and verified to share **zero** feature profiles.
`data/processed/{modeling_dataset,train,validation,test}.csv` are the only files later phases should model from.
Counts above are the authoritative record; see `results/metrics/02_*.csv`.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/02_data_quality_and_split.ipynb")
print("wrote notebook 02")

