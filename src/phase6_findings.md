> **Evidence labels.** *OOF* = Phase 5 out-of-fold training predictions (n = 6,938). *Validation* = the development validation split (n = 1,487; observed placement rate 41.6%), which already informed model selection in Phase 4, so validation results at a threshold chosen on validation are **optimistic**. *CV* = 5-fold stratified cross-validation on the training set (the Phase 5 folds). **`test.csv` was never opened** (a hook raises an error on any attempt; the audit above lists every data file read). Everything describes **associations learned by the fitted model** in observational, possibly rule-generated data; nothing here is a causal claim. The model is still the *provisional* raw (uncalibrated) `LR_B`; no final model is declared.

### A. Decision threshold
**Criterion (fixed in advance).** *Selected operating threshold under the F1-oriented criterion* = the threshold on the 0.20-0.80 grid (step 0.05) with the highest **validation** F1; ties go to the value closest to 0.50. Result: **0.35** (validation F1 0.766). It is not called "best": F1 weights precision and recall equally and ignores the costs of the two error types, which the project has not defined.

| Validation (n = 1,487) | Threshold | Accuracy | Precision | Recall | Specificity | F1 | FP / FN | Predicted placed |
|---|---|---|---|---|---|---|---|---|
| Lower | 0.30 | 0.774 | 0.681 | 0.859 | 0.713 | 0.760 | 249 / 87 | 52.5% |
| **Selected** | **0.35** | 0.787 | 0.705 | 0.838 | 0.750 | **0.766** | 217 / 100 | 49.4% |
| Default | 0.50 | 0.802 | 0.772 | 0.741 | 0.845 | 0.756 | 135 / 160 | 39.9% |
| Higher | 0.70 | 0.777 | 0.850 | 0.561 | 0.930 | 0.676 | 61 / 271 | 27.4% |

- **The choice between 0.35 and 0.50 matters little for F1.** Validation F1 is within 0.01 of its maximum for every threshold from 0.30 to 0.55 (0.760, 0.766, 0.764, 0.764, 0.756, 0.757). The paired-bootstrap difference F1(0.35) - F1(0.50) is +0.010 (95% interval -0.007 to +0.028), which includes 0, so **0.50 is practically tied with the selected threshold**. On the independent OOF data the F1-optimal threshold is 0.40 (F1 0.763); at 0.35 F1 is 0.762 against 0.756 at 0.50, and 0.35 is better in 4 of the 5 folds (mean +0.006). So the selection sits on a flat plateau (about 0.35-0.45), not on a sharp optimum.
- **What the threshold really trades.** Lowering it from 0.50 to 0.35 converts 60 false negatives into 82 extra false positives on validation: recall rises from 0.74 to 0.84, precision falls from 0.77 to 0.71, accuracy falls from 0.802 to 0.787 (accuracy itself is highest at 0.55-0.60, 0.807 / 0.802), and the model labels 49% of students "Placed" against a true rate of 42%. The sensibility check passed (precision and recall both >= 0.60; predicted rate within 10 points of the observed rate). Thresholds above 0.65 give up too much recall for the precision gained (F1 falls to 0.72 and below).
- **A threshold is a policy choice, not a model property.** If a false "likely to be placed" is costlier than a missed one (or vice versa), the right threshold moves, and 0.35, 0.40 and 0.50 are all defensible. The probability, not the label, remains the primary output.

### B. Logistic-regression coefficients (associations, not causal effects)
Odds ratios are per one training SD for the five standardised scores and relative to the stated reference for the one-hot variables. 95% intervals are Wald intervals from an **unpenalised** fit on the same design; the penalised and unpenalised coefficients agree to within 0.006 log-odds everywhere except `Projects = 0`, which the L2 penalty pulls toward 0 (0.50 vs 0.67) because it has only 19 training students.

| Feature (reference) | Odds ratio (95% interval) | Reading |
|---|---|---|
| PlacementTraining = Yes (No) | 2.75 (2.33-3.28) | largest single coefficient; associated with higher predicted odds |
| ExtracurricularActivities = Yes (No) | 1.95 (1.66-2.28) | associated with higher predicted odds |
| AptitudeTestScore (per 1 SD) | 1.65 (1.51-1.81) | largest of the standardised scores |
| SSC_Marks / SoftSkillsRating / CGPA / HSC_Marks (per 1 SD) | 1.33 / 1.32 / 1.29 / 1.23 | each about 1.2-1.3 |
| Workshops = 2 vs 0 | 1.53 (1.30-1.80) | |
| Workshops = 1 vs 0 | 0.73 (0.61-0.86) | level 1 is *below* level 0, as found in Phases 2-3 |
| Workshops = 3 vs 0 | 0.99 (0.71-1.39) | about equal to 0 (n = 208) |
| Projects = 1 / 2 vs 3 | 0.66 (0.55-0.79) / 0.73 (0.61-0.86) | fewer than 3 projects is associated with lower odds than 3 |
| Projects = 0 vs 3 | 1.65 (0.63-6.04) | **unreliable: 19 students; interval includes 1; do not interpret** |
| Internships = 0 / 2 vs 1 | 1.26 (1.06-1.49) / 1.25 (1.07-1.48) | level 1 is slightly lower than both 0 and 2; effect is small |

- The two binary variables and aptitude carry the largest associations. The count variables remain non-monotonic after adjustment (Workshops 1 < 0, Projects 1 and 2 < 3), which supports the Phase 4 choice of treating them as categories.
- Internships has a small coefficient once the other variables are included, and (below) almost no predictive importance, in contrast with its strong raw association in the EDA, which suggests that raw association overlaps with that of the other predictors (this is an inference from the model, not tested directly).
- These are conditional, in-sample associations from one model and one representation; the intervals ignore that representation and reference levels were chosen with the data in view.

### C. Permutation importance and global comparison (validation)
Shuffling each raw column in the full pipeline (20 repeats, seed 42) gives the AUC lost (baseline validation AUC 0.872) and the log loss added (baseline 0.441):

| Rank | Feature | AUC lost (mean +/- SD) | Log loss added |
|---|---|---|---|
| 1 | AptitudeTestScore | 0.0208 +/- 0.0036 | 0.0333 |
| 2 | PlacementTraining | 0.0177 +/- 0.0043 | 0.0267 |
| 3 | ExtracurricularActivities | 0.0125 +/- 0.0022 | 0.0189 |
| 4 | Workshops/Certifications | 0.0081 +/- 0.0015 | 0.0136 |
| 5 | SoftSkillsRating | 0.0060 +/- 0.0015 | 0.0098 |
| 6 | SSC_Marks | 0.0050 +/- 0.0019 | 0.0087 |
| 7 | Projects | 0.0039 +/- 0.0011 | 0.0069 |
| 8 | HSC_Marks | 0.0038 +/- 0.0016 | 0.0064 |
| 9 | CGPA | 0.0036 +/- 0.0013 | 0.0064 |
| 10 | Internships | 0.0003 +/- 0.0006 | 0.0006 |

- **Consistent across methods:** aptitude, placement training and extracurricular activities are the three most influential features under permutation importance (AUC and log loss agree on ranks 1-7 and 10) and under the SD of each feature's contribution to the linear predictor (Spearman rank correlation 0.92 over only 10 features, a rough guide). Internships is last in both.
- **Where they differ.** SSC_Marks is 4th by contribution SD but 6th by permutation importance; Projects is 9th by contribution SD but 7th by permutation; CGPA is 7th vs 9th. These are small shifts among features with similar sizes.
- **Why importance is shared and unstable.** The five scores are moderately correlated (aptitude with HSC r = 0.56, with Projects 0.54, soft skills 0.51, SSC 0.51), so shuffling one column removes only the part of its information that the others do not duplicate; the single-feature values are therefore not exclusive contributions, and ranks 5-9 (AUC loss 0.0036-0.0060, SDs 0.0013-0.0019) are close enough that their order should not be over-read. There is no single "true" ranking.
- Importance describes what the **fitted model relies on to predict**, not what would change an outcome.

### D. Individual validation-student explanations
Contributions are `coefficient x transformed value`, summed per feature, relative to the training mean / reference level; each case's contributions plus the intercept (-1.69) reproduce the model's log-odds exactly.

| Case | Actual | p | Class at 0.50 / 0.35 | Main positive | Main negative |
|---|---|---|---|---|---|
| 1 Very high | Placed | 0.91 | Placed / Placed | Placement training +1.01, extracurricular +0.67, aptitude 87 (+0.47) | none |
| 2 Very low | NotPlaced | 0.02 | NotPlaced / NotPlaced | CGPA 8.0 (+0.12) | aptitude 69 (-0.63), soft skills 3.6 (-0.49), Projects = 1 (-0.41) |
| 3 Boundary | Placed | 0.50 | Placed / Placed | placement training +1.01, extracurricular +0.67, HSC 88 (+0.32) | Projects = 1 (-0.41), aptitude 76 (-0.20) |
| 4 False positive (most confident) | NotPlaced | 0.93 | Placed / Placed | training +1.01, extracurricular +0.67, aptitude 90 (+0.65) | none |
| 5 False negative (most confident) | Placed | 0.02 | NotPlaced / NotPlaced | none | soft skills 3.6 (-0.49), CGPA 6.6 (-0.44), Projects = 1 (-0.41) |
| 6 Middle | NotPlaced | 0.34 | NotPlaced / NotPlaced | placement training +1.01, SSC 83 (+0.38), soft skills 4.8 (+0.32) | Projects = 2 (-0.32), Workshops = 1 (-0.31) |

- The explanations are additive and readable: the model's very confident cases are decided by the same few features that dominate globally (training, extracurricular, aptitude, then the count levels).
- **The most confident false positive (case 4) has essentially the top profile** (CGPA 8.6, 2 internships, 3 projects, 2 workshops, aptitude 90, soft skills 4.8, training and extracurricular both Yes, SSC 80, HSC 85) and is almost indistinguishable from the correctly-predicted case 1. This matches the Phase 2 finding of 8 conflicting-label profiles in the same top region (identical predictors, different outcomes): the features carry no information that could separate such students. Cases 4 and 5 are deliberately the *most extreme* errors, not typical ones.

### E. Feature-group ablation (logistic regression, same preprocessing and regularisation)
`ABL_E` reproduces the Phase 4 `LR_B` validation probabilities and the Phase 5 OOF probabilities exactly (asserted).

| Group | Features | CV AUC (mean +/- SD) | CV log loss | Val AUC (95% CI) | Val log loss | Val Brier | Val F1 @ 0.50 |
|---|---|---|---|---|---|---|---|
| ABL_A Academic | 3 | 0.840 +/- 0.013 | 0.488 | 0.823 (0.802-0.845) | 0.510 | 0.166 | 0.711 |
| ABL_B + Aptitude | 4 | 0.859 +/- 0.013 | 0.462 | 0.844 (0.825-0.864) | 0.482 | 0.157 | 0.734 |
| ABL_C + Technical | 7 | 0.866 +/- 0.011 | 0.451 | 0.856 (0.836-0.875) | 0.465 | 0.150 | 0.745 |
| ABL_D Employability only | 3 | 0.834 +/- 0.004 | 0.496 | 0.828 (0.807-0.849) | 0.507 | 0.166 | 0.725 |
| ABL_E Full | 10 | 0.879 +/- 0.010 | 0.431 | 0.872 (0.854-0.890) | 0.441 | 0.141 | 0.756 |

Reference (always predict the train rate): AUC 0.500, log loss 0.679, Brier 0.243.

| Step (bigger minus smaller) | Validation dAUC (95% CI) | Validation d log loss (95% CI) | CV folds improved (AUC / log loss) |
|---|---|---|---|
| Aptitude added to academic (B - A) | +0.021 (0.013-0.030) | -0.029 (-0.040 to -0.018) | 5/5, 5/5 |
| Technical counts added to B (C - B) | +0.012 (0.006-0.018) | -0.017 (-0.025 to -0.009) | 5/5, 5/5 |
| Employability added to C (E - C) | +0.016 (0.008-0.023) | -0.024 (-0.035 to -0.013) | 5/5, 5/5 |
| Full vs academic only (E - A) | +0.049 (0.037-0.061) | -0.070 (-0.087 to -0.053) | 5/5, 5/5 |
| Full vs employability only (E - D) | +0.044 (0.029-0.058) | -0.066 (-0.083 to -0.046) | 5/5, 5/5 |

- **Academic information alone** (CGPA, SSC, HSC) already gives validation AUC 0.82 and log loss 0.51, well above the 0.50 / 0.68 reference.
- **Employability alone** (soft skills, extracurricular, placement training; only 3 features) is as informative as academic alone (AUC 0.828 vs 0.823, log loss 0.507 vs 0.510).
- **Each added group improved predictive performance** on top of what was already included (aptitude +0.021 AUC, technical counts +0.012, employability +0.016), consistently in all 5 folds and with validation intervals excluding 0. No group is redundant, and none dominates: the full model improves on academic-only by 0.049 AUC and 0.070 log loss.
- **Cautions.** The groups overlap and are correlated, so these increments are order-dependent (employability's gain is measured only after everything else; aptitude's after academic only), not clean contributions of each group. The increments are modest, and the CV SDs (about 0.011-0.013 AUC) show that the smaller steps (about 0.012) are of the same size as fold-to-fold variation, although they are consistent in sign. Validation intervals of the individual models overlap widely; only the paired differences carry the evidence.

### F. Error analysis (validation, `LR_B`, threshold 0.50)
- **Overall.** 295 errors: 135 false positives (9.1% of students) and 160 false negatives (10.8%). The observed error rate (19.8%) equals the error rate expected **if the probabilities were perfectly calibrated** (mean of min(p, 1-p) = 19.6%): the model errs about as often as its own probabilities say it should, so the errors are mostly irreducible uncertainty given these ten features, not a calibration failure.
- **Where the errors are.** 70% of all errors fall among the 42% of students with 0.2 < p < 0.8. The predicted-probability distributions of Placed and NotPlaced overlap substantially (overlap coefficient 0.39).
- **High-confidence (p >= 0.8 or <= 0.2) correct predictions:** 773 students (52.0%).
- **High-confidence incorrect predictions: 88 students (5.9% of students; 29.8% of all errors).**
  - *HC false positives* (p >= 0.80 but NotPlaced): **28 students** (1.9%; 9.5% of errors); within the high band the NotPlaced share is 10.1% against 12.2% expected. Their profile is close to the correctly predicted high-confidence Placed students (all standardised differences |d| <= 0.44; 100% have placement training and 96% extracurricular activities). Nothing in the recorded features distinguishes them.
  - *HC false negatives* (p <= 0.20 but Placed): **60 students** (4.0%; 20.3% of errors); within the low band the Placed share is 10.3% against 8.3% expected. Compared with other low-probability students they have more extracurricular activity (38% vs 16%; d = 0.52), higher aptitude (d = 0.46) and more projects (d = 0.36), i.e. they sit somewhat higher in the profile range than the typical low-probability student but still below 0.2.
- **Practical ceiling.** About 20% error at the default threshold, and roughly 1 in 10 confident predictions wrong in both bands, appears to be the limit of what these ten features support in this dataset, consistent with the 8 conflicting-label profiles (Phase 2) and the model-family tie (Phase 4-5). Small groups (28 and 60 students): the feature patterns are descriptive only.

### G. Does the dataset structure limit what a model can achieve?
The observed structure is **consistent with a rule-based or constructed dataset**: strong, step-like and non-monotonic count effects (Internships, Workshops, Projects), hard floors and ceilings on the scores, very large systematic effects across all predictors, heavy class overlap with a performance plateau shared by logistic regression, boosting and forest, and conflicting labels at the top of the profile range. The dataset card gives no provenance, so this is **not** established, and nothing here proves the data are synthetic. If a rule plus label noise generated the data, the attainable accuracy is bounded by that process, and **results should not be generalised automatically to real university placement populations, other campuses or other years**; the coefficients are descriptive of this dataset, not of how placement works.

### H. Should the provisional model remain?
**Yes: logistic regression with categorical counts (`LR_B`, raw probabilities) remains the provisional model, still not final.** Phase 6 adds interpretability and error evidence but changes none of the Phase 4-5 conclusions: the coefficients are readable and sensible, the explanations are additive and exact, the full 10-feature model beats every partial group, the errors match the model's own stated uncertainty, and no calibration step is needed. The only new operational item is the threshold, which is a policy choice with a flat F1 plateau.

### I. Recommendation for Phase 7
1. **Fix the evaluation protocol before opening `test.csv`:** freeze `LR_B` (Phase 4 pipeline, C = 1, raw probabilities, fitted on the training set only), the operating threshold(s) to report (the default 0.50 and the F1-oriented 0.35, both declared in advance), and the metric list (ROC-AUC, log loss, Brier, calibration slope / intercept / reliability, and precision / recall / F1 at the declared thresholds with bootstrap intervals). GB_B and RF_B may be reported as pre-declared secondary comparators with no re-selection.
2. **Open the test set exactly once**, with the same audit hook, and report the result whatever it is.
3. Decide in advance whether to refit on train + validation for any deployment artifact; this must not change the reported test result.
4. Subgroup and stability checks (small count levels, top-profile region) can be added as clearly labelled exploratory analyses on train/validation only.

### Limitations
- Validation was already used for model development, so validation figures for the chosen threshold and for the already-selected model are optimistic; the OOF analysis is the independent check.
- Coefficient intervals are Wald intervals for the unpenalised fit, conditional on the representation and reference levels; permutation importance is not additive in the presence of correlated predictors.
- Ablation groups overlap and are order-dependent; five CV folds give coarse stability evidence; the validation bootstrap treats students as independent.
- The high-confidence error groups are small (28 and 60 students); the individual cases were chosen by fixed rules (two are deliberately the most extreme errors).
- One possibly rule-generated dataset; no real-world generalisation claim is made.
