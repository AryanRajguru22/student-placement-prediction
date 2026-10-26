# Predicting Student Placement Probability Using Academic, Technical and Employability Features

## Quick start: run the placement predictor app
The finished product is a local Streamlit app (`app/`) that returns a model-based placement probability from ten student characteristics, using the frozen Phase 7 Logistic Regression B. Requires Python 3.11 or newer.

- **Windows:** double-click `run_app.bat` (first run creates `.venv` and installs `requirements-app.txt`).
- **Any platform:** `python -m venv .venv`, activate it, `pip install -r requirements-app.txt`, then `streamlit run app/app.py` from this folder. Open <http://localhost:8501>; stop with Ctrl+C.
- Check the app layer with `python app/verify_app.py` (178 checks; never opens the held-out test data).

Full app documentation, thresholds, limitations, privacy and troubleshooting: [`app/README.md`](app/README.md). The final report is `report/final_report.md` and the presentation `presentation/final_presentation.pptx`. The app runs locally only; it is not deployed. The phase-by-phase history below is unchanged ("Current phase: 1" in it describes Phase 1 when it was written; all phases are complete).

## Problem statement
Campus placement outcomes depend on a mix of academic record, technical exposure and employability signals. Given a
student's profile, can we estimate the **probability** that they will be placed?

## Objective
Build a predictive-analysis pipeline that outputs a calibrated placement probability from academic (CGPA, SSC/HSC marks),
technical (projects, internships, workshops/certifications, aptitude score) and employability
(soft skills, extracurriculars, placement training) features, and identify which factors matter most.

## Dataset
- Source: Hugging Face - [`Krooz/Campus_Recruitment_CSV`](https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV) (CC0-1.0)
- Files: `raw_datasets/train.csv`, `validation.csv`, `test.csv` (7,225 / 1,275 / 1,500 rows = 10,000), saved unmodified in `data/raw/`
- 10 features + target `PlacementStatus` (`Placed` / `NotPlaced`)

## Current phase: 1 - Data understanding (complete)
`notebooks/01_data_understanding.ipynb` inspects shape, types, missingness, duplicates, categorical values,
descriptive statistics, target balance and suspicious values. No features were engineered, no models trained,
no rows removed. Outputs: `data/processed/campus_recruitment_combined_flagged.csv` (all rows + audit flags),
`results/figures/`, `results/metrics/`.

## Planned methodology
1. **Data preparation** - decide duplicate / cross-split-leakage handling; re-split with stratification (original splits overlap).
2. **EDA** - feature-vs-target analysis, statistical tests (chi-square, t/Mann-Whitney), multicollinearity.
3. **Modelling** - logistic regression baseline, then tree-based models (random forest, gradient boosting) with stratified cross-validation.
4. **Evaluation** - ROC-AUC, PR-AUC, F1, Brier score, calibration curves, confusion matrices.
5. **Interpretation** - coefficients, permutation importance, partial dependence.
6. **Deployment** - small app in `app/` returning a placement probability; written report in `report/`.

## Structure
```
data/{raw,processed}  notebooks/  src/  models/  results/{figures,metrics}  app/  report/
```

## Reproduce
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb
```
(`src/build_notebook_01.py` regenerates the notebook source.)

## Phase 1.5 - Data Quality and Split Integrity (complete)
The three files supplied with the dataset were **not** used as the final train/validation/test sets. Phase 1 showed that
33 feature profiles (69 rows) occur in more than one of them, and 33 validation/test rows (17 + 16) share a profile with train, so
evaluating on the supplied splits would partly measure memorisation. `notebooks/02_data_quality_and_split.ipynb` therefore:

1. Leaves `data/raw/*.csv` byte-identical (SHA-256 asserted) and stacks them: 10,000 rows.
2. Collapses duplicate feature profiles (all 10 predictors) with the same label to one record (72 rows removed).
3. Excludes the 8 profile groups (16 rows) whose duplicates carry conflicting `PlacementStatus` labels, with no majority vote; they are listed in `results/metrics/02_conflicting_label_groups.csv`.
4. Builds `data/processed/modeling_dataset.csv` (9,912 rows) and a stratified 70/15/15 split with `random_state=42`
   (`train.csv` 6,938 / `validation.csv` 1,487 / `test.csv` 1,487), asserting zero feature-profile overlap between splits.

All later phases must use `data/processed/{train,validation,test}.csv`.

## Phase 2 - Exploratory Data Analysis (complete)
`notebooks/03_eda.ipynb` (built by `src/build_notebook_03.py`) analyses only `data/processed/modeling_dataset.csv` and the new
`train/validation/test.csv`; no models are trained and nothing is engineered. Method:
- **Integrity assertions** first: no missing values, two target classes, no duplicate rows/profiles, zero cross-split profile overlap.
- **Univariate:** target balance (58.44% NotPlaced / 41.56% Placed, ~1.4:1, moderate), histograms for scale-type variables, bar charts for 3-4 level count variables (no outlier rules applied to counts).
- **Bivariate:** violin plots + Cohen's *d* for scores; placement rate with Wilson 95% intervals for count and binary variables; no monotonic relationship assumed. The non-monotonic Internships/Projects/Workshops patterns are probed via composition tables, two-way heatmaps and an aptitude-stratified view.
- **Association:** Pearson and Spearman correlations (association, not causation; Pearson is linear-only and weak for discrete/non-monotonic variables).
- **Multivariate:** targeted scatter/rate-map plots instead of a pair-plot. Quantile bins are *exploratory, for visualisation only* and are never saved as features.
- **Split sanity check:** train/validation/test compared with summary statistics, ECDFs, KS statistics and proportion gaps.
- **Phase 3 plan:** `results/metrics/03_phase3_test_plan.csv` lists the relationships to test formally and why each test fits.

Outputs: `results/figures/eda/` (13 figures), `results/metrics/03_eda_summary.csv`.

## Phase 3 - Statistical Analysis (complete)
`notebooks/04_statistical_analysis.ipynb` (built by `src/build_notebook_04.py`) formally tests the EDA findings on the modelling dataset
(unchanged; SHA-256 verified). No models are trained. alpha = 0.05; every test states its question, H0/H1, rationale, statistic, p-value, effect size and CI.
- **Scores vs placement:** Mann-Whitney U (primary), Welch t (sensitivity), Cohen's d and rank-biserial effect sizes (bootstrap CI).
- **Binary predictors:** chi-square of independence (assumptions checked), Cramer's V, odds ratio with Woolf CI.
- **Count predictors (Internships, Workshops, Projects):** omnibus chi-square plus planned pairwise comparisons (z-test, or Fisher's exact for small groups), rate differences with Newcombe CIs. No trend test, because the patterns are non-monotonic.
- **Confounding check:** Cochran-Mantel-Haenszel (general and 2x2 contrasts, Breslow-Day) stratified by the EDA aptitude terciles; crude vs stratified odds ratios. Not a causal adjustment.
- **Redundancy:** Pearson/Spearman matrices and VIF (all VIF below 2; strongest pair aptitude / HSC, r = 0.56). No variables removed.
- **Multiple comparisons:** Holm within separate logical families (documented in section 10); raw and adjusted p-values both reported.
- **Policy:** fit on train, select on validation, evaluate on test once; key results are replicated on the training split alone.

Outputs: `results/metrics/statistical/04_*.csv` (numeric, categorical, pairwise, stratified, VIF, effect-size summary, plus supporting tables) and `results/figures/statistical/`.

## Phase 4 - Preprocessing and Baseline Modelling (complete)
`notebooks/05_preprocessing_and_baselines.ipynb` (built by `src/build_notebook_05.py`) fits baseline models on **train** and compares/selects on **validation** only.
`test.csv` is not opened by this notebook (an audit hook records every file opened and the notebook asserts this; scope: this notebook only, see the Phase 7 test-set access history); it is reserved for the one-time final evaluation in Phase 6. No threshold tuning (default 0.5).
- **Preprocessing (sklearn `Pipeline` + `ColumnTransformer`, fitted on train inside the pipeline):** Variant A keeps Internships/Projects/Workshops numeric; Variant B one-hot encodes them. Scale-type variables are standardised for logistic regression; tree models are unscaled.
- **Models:** L2 logistic regression (plus class-balanced comparison), decision tree, random forest, gradient boosting, each with a small explicit validation search (grids saved in `results/metrics/modeling/05_tuning_grids.csv`).
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, log loss and Brier score with bootstrap CIs; paired bootstrap for model and representation comparisons; train-vs-validation overfitting check; probability-distribution diagnostics.
- **Provisional selection (validation only):** logistic regression with categorical counts (validation ROC-AUC 0.872, log loss 0.441, Brier 0.141), chosen because it is statistically tied with gradient boosting and random forest on probability quality while being the simplest and most interpretable, with the smallest train-validation gap. Calibration, cross-validation and threshold analysis are left to Phase 5.
- **Note:** scikit-learn 1.9 deprecates `penalty="l2"`; the equivalent `l1_ratio=0` is used (verified to give identical coefficients).

Outputs: `models/phase4/*.joblib` and `model_configs.json`, `results/metrics/modeling/05_*.csv` (validation results, coefficients, feature importance, train-validation comparison, grids, predictions) and `results/figures/modeling/`.

## Phase 5 - Cross-Validation, Calibration and Model Stability (complete)
**The Phase 5 notebook did not access the test set** (scope: this notebook only; see the Phase 7 test-set access history). `data/processed/test.csv` (and the raw test file) was not opened by it: an audit hook in `notebooks/06_calibration_and_cross_validation.ipynb` raises an error on any attempt, and the notebook ends by listing every data file opened and asserting the test set was not among them.

**Objective.** Check whether the three leading Phase 4 models are stable under cross-validation, how well calibrated their probabilities are, and whether recalibration helps; confirm the provisional model choice.

**Methodology** (built by `src/build_notebook_06.py`):
- *Models:* LR_B, GB_B, RF_B, cloned unchanged from the Phase 4 pipelines (the decision tree is omitted: Phase 4 showed it is clearly weaker).
- *Cross-validation:* 5-fold stratified (shuffle, seed 42) on the **training set only**; ROC-AUC, log loss and Brier are primary, F1/accuracy secondary; paired fold differences with Nadeau-Bengio-corrected intervals.
- *Out-of-fold predictions:* one probability per training student from a model that never saw that student (`06_oof_predictions.csv`, stable `train_row_id`).
- *Calibration:* reliability diagrams (10 equal-count bins, Wilson intervals), calibration slope, intercept and calibration-in-the-large, Brier, log loss and ECE (with a simulated "perfect calibration" noise yardstick).
- *Recalibration:* nested, leakage-safe comparison of Original vs sigmoid (Platt on logit) vs isotonic using only training data; a pre-declared materiality rule decides whether a method helps.
- *Validation confirmation:* calibrators fitted on train only and evaluated once on validation (kept separate from the training-CV estimates).

**Key findings.**
- Cross-validation supports the Phase 4 tie: mean ROC-AUC 0.879 / 0.876 / 0.876 and log loss 0.431 / 0.434 / 0.433 (LR / GB / RF) differ by only about a quarter of the fold-to-fold SD; no corrected interval excludes zero.
- Raw probabilities are already reasonably calibrated for all three models (LR slope 0.99, GB 0.98, RF 1.08, mean prediction equals observed rate). Small local departures (up to about 5-6 points in some bins) remain.
- Sigmoid recalibration is negligible; isotonic does not help (log loss and Brier unchanged or worse, AUC slightly lower, over-confident extremes). No method met the materiality rule.
- Predicted probabilities are mostly confident (52-60% of students fall below 0.2 or above 0.8) and those extremes are borne out: the observed placement rate is within about 2 points of the mean prediction in both extreme bands. The 0.2-0.8 middle band is placed about 48% of the time, i.e. genuinely uncertain.
- A predicted probability describes a *group* of similar students, not a guarantee for an individual.

**Limitations.** Only five folds; overlapping fold training sets; OOF models use 80% of the training data; ECE depends on binning; the validation set is too small to detect small miscalibration; GB/RF hyper-parameters were chosen on validation in Phase 4; single, possibly rule-generated dataset.

**Provisional model status.** Logistic regression with categorical counts (LR_B) remains the *provisionally preferred* model (not final): statistically tied on predictive quality, best mean metrics, interpretable and acceptably calibrated without recalibration. Conclusions and the recommendation for Phase 6 are in `src/phase5_findings.md`.

Outputs: `results/metrics/modeling/06_*.csv` (CV folds and summary, paired differences, OOF predictions, calibration bins and metrics, recalibration comparison, probability-distribution summary, validation confirmation) and `results/figures/modeling/phase5_*.png`.

## Phase 6 - Threshold Analysis, Explainability and Feature-Group Ablation (complete)
**The Phase 6 notebook did not access the test set** (scope: this notebook only; see the Phase 7 test-set access history). `data/processed/test.csv` (and the raw test file) was not opened by it: an audit hook in `notebooks/07_threshold_explainability_ablation.ipynb` raises an error on any attempt, and the notebook ends by listing every data file opened and asserting the test set was not among them. The provisional model is unchanged: logistic regression with categorical counts, raw (uncalibrated) probabilities. No final model is declared.

**Methodology** (built by `src/build_notebook_07.py`):
- *Threshold analysis:* thresholds 0.20-0.80 in steps of 0.05 applied to the existing `LR_B` probabilities (no refitting), reported **separately** for the Phase 5 out-of-fold training predictions and for the validation set. The pre-declared operating criterion is "highest validation F1, ties to the value nearest 0.50", checked with a paired bootstrap against 0.50 and per-fold OOF differences; it is described as the *selected operating threshold under the F1-oriented criterion*, not as the best threshold.
- *Explainability:* coefficients and odds ratios of the fitted pipeline (per training SD for scores, versus the reference level for one-hot variables) with Wald 95% intervals from an unpenalised fit on the same design; permutation importance on validation (ROC-AUC and log loss, 20 repeats, seed 42); a comparison with the SD of each feature's contribution to the linear predictor.
- *Individual explanations:* six validation students chosen by fixed rules (very high, very low, boundary, most confident false positive, most confident false negative, median); contributions are `coefficient x transformed value`, relative to the training mean / reference level.
- *Feature-group ablation:* five logistic regressions (academic; + aptitude; + technical counts; employability only; full) with the same preprocessing and regularisation as `LR_B` (the full model reproduces `LR_B` exactly), evaluated by the Phase 5 5-fold CV on train and once on validation, with paired bootstrap differences.
- *Error analysis:* validation false positives / negatives and high-confidence (p >= 0.8 or <= 0.2) correct and incorrect predictions, with standardised feature differences and a calibrated-expectation ceiling check.

**Key findings.**
- Selected threshold **0.35** (validation F1 0.766), but F1 is flat from 0.30 to 0.55 and 0.50 is practically tied (bootstrap difference +0.010, 95% interval -0.007 to +0.028). Moving from 0.50 to 0.35 trades 60 fewer false negatives for 82 more false positives (recall 0.74 -> 0.84, precision 0.77 -> 0.71); the right threshold depends on the relative cost of the two errors, which the project has not defined.
- Placement training (OR 2.75), extracurricular activities (1.95) and aptitude (1.65 per SD) have the strongest associations; the count variables stay non-monotonic (Workshops 1 below 0; Projects 1 and 2 below 3); Projects = 0 has only 19 students and is not interpretable. Aptitude, placement training and extracurricular activities are also the top three by permutation importance; Internships is last. These are associations, not causal effects.
- Each feature group added predictive value (validation AUC: academic 0.823, + aptitude 0.844, + technical 0.856, full 0.872; employability alone 0.828), consistently across folds; the groups overlap, so increments are order-dependent.
- The observed error rate (19.8% at 0.50) matches the rate expected from the model's own probabilities (19.6%); 70% of errors lie in the uncertain 0.2-0.8 range. 88 students (5.9%) are wrong with high confidence; the most confident false positive has essentially the top profile, as do the conflicting-label profiles found in Phase 2.

**Limitations.** Validation already informed model selection, so validation results at the chosen threshold are optimistic; coefficient intervals are for the unpenalised fit and conditional on the chosen representation; permutation importance is shared among correlated predictors; ablation increments are order-dependent; the high-confidence error groups are small (28 and 60 students); the data structure is consistent with a rule-based or constructed dataset (not established), so results should not be generalised to real placement populations.

Outputs: `src/phase6_findings.md`; `results/metrics/modeling/07_*.csv` and `07_threshold_selection.json` (thresholds, coefficients, permutation importance, global comparison, individual explanations, ablation and increments, error analysis, error comparison and ceiling); `results/figures/modeling/phase6_*.png`.

## Phase 7 - Final Test Evaluation (complete)
`notebooks/08_final_test_evaluation.ipynb` (built by `src/build_notebook_08.py`; full write-up in `src/phase7_findings.md`) is the **one-time, pre-registered evaluation on `data/processed/test.csv`**. Test data were **not used for model development, selection, tuning, calibration, or threshold selection**. Earlier exploratory notebooks (Phases 1-3) accessed the eventual test rows for descriptive analysis; these rows were not used to fit or evaluate predictive models. During Phase 7, the test set was intentionally accessed once under the final-evaluation audit gate after the modelling protocol had been frozen. Earlier phase history above is unchanged. **Test-set access history:** besides the Phase 1-3 descriptive use, two orientation shell commands (`head -2 data/processed/test.csv`, which displayed the header and one data row, and `wc -l data/processed/*.csv`, line counts only) read the file outside the audit hook before the Phase 7 notebook ran; neither was used for any model decision, tuning, threshold selection, calibration or fitting. The statements in the Phase 4-6 sections above that the test file was "never opened" describe those notebooks' own audit hooks, not the whole project history.

**Frozen before the test set was opened** (`results/metrics/modeling/08_frozen_protocol.json`): primary model **Logistic Regression B** (L2, C = 1, no class weighting, categorical Internships/Projects/Workshops, raw probabilities, no recalibration); thresholds **0.50** (reference) and **0.35** (*selected operating threshold under the F1-oriented criterion*, not "the best"); `GB_B` and `RF_B` as descriptive secondary comparators.

**Final training procedure.** The final Logistic Regression B specification was frozen before test access. After all model, calibration, and threshold decisions were complete, the same frozen specification was refit using the combined training + validation data (8,425 students) and then evaluated once on the held-out test set. The Phase 4/5/6 *development* results (CV, validation) come from models fitted on train data only; the Phase 7 *final* test metrics come from the frozen specification refit on train + validation (cloned; parameters asserted identical; a train-only refit reproduced the Phase 4 validation probabilities). The test set was not used to make any subsequent modelling decision. They are saved in `models/phase7/`. Access controls: an audit hook allowing exactly one gated read of the processed test file and none of the raw one; a marker file (`08_test_access_log.json`) that blocks re-execution; the notebook was executed once on the real data after two dry runs that used `validation.csv` as a stand-in.

**Test results (n = 1,487; placed rate 41.6%), LR_B.**
| ROC-AUC (95% CI) | Log loss | Brier | Calibration slope / intercept | ECE |
|---|---|---|---|---|
| 0.886 (0.868-0.902) | 0.420 (0.394-0.449) | 0.134 (0.124-0.145) | 1.05 (0.95-1.15) / +0.04 (-0.10 to +0.18) | 0.016 (noise level under perfect calibration: 0.023) |

| Threshold | Accuracy | Precision | Recall | F1 | Specificity | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|---|
| 0.50 | 0.812 | 0.779 | 0.765 | 0.772 | 0.846 | 473 / 735 / 134 / 145 |
| 0.35 | 0.802 | 0.722 | 0.851 | 0.781 | 0.766 | 526 / 666 / 203 / 92 |

On the final test set, 0.35 produced higher recall and F1 than 0.50, while also producing lower precision and specificity. 0.35 was selected in Phase 6 by the predeclared validation F1-oriented rule; 0.50 is the conventional reference threshold. Neither is universally preferable: the choice depends on the relative cost of false negatives and false positives, which the project has not defined.

**Calibration.** Raw probabilities are acceptably calibrated on test (no reliability bin outside its Wilson interval; ECE below its noise yardstick). A small, consistent under-prediction at the top (observed 0.946 vs predicted 0.905 in the top decile; also seen in Phase 5) was **not** recalibrated and is reported as a limitation.

**Secondary models.** GB_B and RF_B are indistinguishable from LR_B on test (AUC 0.886 for all three; log loss 0.419-0.420; Brier 0.134-0.135; every paired bootstrap interval includes 0), continuing the Phase 4-5 tie. No model was re-selected; LR_B was not replaced.

**Development vs test.** CV AUC 0.879 / validation 0.872 / test 0.886; log loss 0.431 / 0.441 / 0.420; Brier 0.139 / 0.141 / 0.134 (`08_development_vs_test.csv`). Test performance was broadly consistent with the development estimates, with test metrics within approximately one CV standard deviation of the corresponding CV means. The higher test performance does not by itself establish improved generalisation and may reflect sampling variation and the larger train + validation refit. Calibration remains reasonable.

**Decision.** *For this dataset and experimental protocol, Logistic Regression B was retained as the final model.* This is not a claim of general superiority, guaranteed generalisation or causation.

**Limitations.** One dataset and one split; the observed structure is consistent with a rule-based or constructed dataset, although this is not established by the dataset source, and strong performance here does not establish equivalent performance on real-world placement populations; validation-selection optimism in development figures; threshold dependence; small top-end under-prediction; test rows were part of the Phase 1-3 descriptive analyses (disclosed in notebook 05), so test independence is not absolute; two orientation shell commands (`head -2`, `wc -l`) touched `test.csv` outside the audit hook before the notebook ran, and no result from them was used. See `src/phase7_findings.md`.

Outputs: `results/metrics/modeling/08_*.csv` and `08_frozen_protocol.json`, `08_test_access_log.json` (predictions, bootstrap CIs, calibration bins and metrics, probability distribution, thresholds, classification metrics, development vs test, final model comparison); `results/figures/modeling/phase7_*.png` (reliability, probability distribution, final model comparison, confusion matrices at 0.50 and 0.35); `models/phase7/`.

## Phase 8A - Interactive Prediction Application
`app/` contains a Streamlit app (`streamlit run app/app.py` from the project root; details in `app/README.md`) that returns a model-based placement probability for one student. It uses the **frozen Phase 7 `LR_B`** (`models/phase7/LR_B_final.joblib`: the Phase 4 preprocessing + logistic-regression pipeline, refit on train + validation) exactly as saved: nothing is retrained, tuned, recalibrated or re-implemented, and Phase 4-7 files and results are unchanged.

- **Inputs:** the ten predictors, all required; ranges and valid levels are those observed in the development data (and Phase 1). Missing, non-numeric, out-of-range or unknown values are rejected with a message, never clipped.
- **Output:** probability of Placed (one decimal; rounded for display only), the classification at the **0.35** project threshold (selected operating threshold under the F1-oriented criterion) and at the conventional **0.50** threshold, shown side by side with neither called best, and a descriptive probability-interpretation band (not a confidence interval).
- **Explanation:** per-feature log-odds contributions from the frozen coefficients (Phase 6 methodology), relative to a baseline student; contributions reconstruct the model's log-odds to machine precision and reproduce the saved Phase 6 explanations. They are associations, not causal effects.
- **Verification** (`python app/verify_app.py`, 178 checks after the final product-completion pass, which also removed `st.form` so that editing any input removes the earlier result, and added Clear profile; the Phase 8A baseline was 122 checks): the model loads and matches its recorded SHA-256; application probabilities equal direct `LR_B` predictions (difference 0 on seven manual profiles); thresholds 0.35 / 0.50 applied correctly including boundary values; explanation reconstruction error below 1e-9; input-validation cases; a headless end-to-end UI run. The app and its verification never read the held-out test split or test prediction files.
- **Limitations:** one observational dataset whose structure is consistent with a rule-based or constructed origin (not established); no claim of equivalent performance on real placement populations; a probability describes a group of similar students, not a certainty.
