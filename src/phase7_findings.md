> **Evidence labels.** *Development evidence* = Phases 1-6 (train, validation, cross-validation; unchanged by this phase). *Final test evidence* = Phase 7: **one** evaluation of the frozen model on the held-out test split (n = 1,487; observed placement rate 41.56%, 618 Placed / 869 NotPlaced). No modelling decision below was changed after the test set was opened (see the access history that follows). All statements are about association and prediction in one observational, possibly rule-generated dataset; none is causal. 95% intervals are percentile bootstrap intervals (2,000 resamples of test students, seed 42).

> **Test-set access history (disclosed, not hidden).** Test data were not used for model development, selection, tuning, calibration, or threshold selection. Earlier exploratory work accessed the eventual test rows descriptively: the Phase 1-3 notebooks (data understanding and split construction, EDA, statistical analysis) loaded the full data, including the rows later assigned to the test split, for descriptive and split-sanity analyses; no predictive model was fitted or evaluated on those rows. Before the gated Phase 7 evaluation, two orientation shell commands also read `data/processed/test.csv` outside the audit hook (`head -2`, which displayed the header and one data row, and `wc -l data/processed/*.csv`, line counts only); neither was used for any model decision, tuning, threshold selection, calibration or fitting. The audit hooks in the Phase 4-6 notebooks show that *those notebooks* did not open the test files; that describes those notebooks, not the whole project history. During Phase 7 the test set was intentionally accessed once, under the final-evaluation audit gate, after the modelling protocol had been frozen.

## A. Final model
**Frozen specification (recorded in `results/metrics/modeling/08_frozen_protocol.json` before the test set was opened).**
- **Primary model: Logistic Regression B (`LR_B`)** - CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks standardised; Internships, Projects, Workshops/Certifications one-hot categorical (reference levels 1, 3, 0); ExtracurricularActivities and PlacementTraining binary; L2, C = 1, **no class weighting**, **raw probabilities, no recalibration** (Phase 5: sigmoid negligible, isotonic not helpful, no method met the materiality rule).
- **Thresholds (both declared in advance):** **0.50** (reference / default) and **0.35** (the *selected operating threshold under the F1-oriented criterion*; Phase 6). No other threshold was evaluated on test and 0.35 is not called "the best" threshold.
- **Pre-declared secondary comparators:** `GB_B` and `RF_B` with their frozen Phase 4 configurations; not used to select or replace `LR_B`.

**Two distinct models.** The final Logistic Regression B specification was frozen before test access. After all model, calibration, and threshold decisions were complete, the same frozen specification was refit using the combined training + validation data and then evaluated once on the held-out test set. The CV and validation development results come from models fitted on train data only (the CV models on 80% of train); the final test metrics come from the frozen specification refit on train + validation. The test set was not used to make any subsequent modelling decision.
| | Phase 4/5/6 *development* model | Phase 7 *final* model |
|---|---|---|
| Specification | frozen pipeline above | identical (`sklearn.clone` of the Phase 4 pipeline; parameters asserted equal to Phase 4 and to `model_configs.json`) |
| Fitted on | train (6,938) | **train + validation (8,425)**, after every decision was frozen |
| Used for | validation, CV, calibration, threshold selection, explainability | test predictions only |

For each of the three models, a clone fitted on **train only** reproduced the saved Phase 4 validation probabilities (tolerance 1e-8), which shows that the clone is the Phase 4 specification and that the fit is deterministic; only then was the clone refitted on train + validation. Refitting re-estimates fitted quantities only: `StandardScaler` means moved by at most 0.02 and SDs by at most 0.05 (SSC_Marks SD 10.36 to 10.41) and the largest `LR_B` coefficient change was 0.111 log-odds (`Projects = 0`, which has about 19 training students and is not interpretable); every other coefficient moved by at most 0.058. No preprocessing, representation, hyper-parameter, feature, regularisation, calibration method or threshold changed. Models are saved in `models/phase7/`; Phase 4 artifacts are byte-identical (checked by SHA-256 at the start and end of the run).

## B. Test discrimination (LR_B)
| Metric | Estimate | 95% bootstrap CI |
|---|---|---|
| **ROC-AUC** | **0.886** (0.8857) | 0.868 - 0.902 |
| Log loss | 0.420 (0.4196) | 0.394 - 0.449 |
| Brier | 0.134 (0.1335) | 0.124 - 0.145 |

A randomly chosen Placed test student receives a higher predicted probability than a randomly chosen NotPlaced one about 89% of the time. Reference values (Phase 4): always predicting the training rate gives AUC 0.500, log loss 0.679, Brier 0.243.

## C. Probability quality and calibration (LR_B, raw probabilities, test)
| Quantity | Value | Reading |
|---|---|---|
| Calibration slope (joint fit) | **1.051** (0.951-1.152) | compatible with 1 |
| Calibration intercept (joint fit) | +0.041 (-0.099 to +0.181) | compatible with 0 |
| Calibration-in-the-large (slope fixed at 1) | +0.032 (-0.104 to +0.169) | mean prediction 0.411 vs observed 0.416 |
| ECE (10 equal-count bins) | **0.016** | at or below the noise expected under *perfect* calibration (mean 0.023, 95th percentile 0.034) |
| ECE (10 equal-width bins, sensitivity) | 0.017 | |

- **Reliability diagram** (`phase7_test_reliability.png`, `08_test_calibration_bins.csv`; 10 equal-count bins of 148-149 students): the largest per-bin gap is 0.041 (top decile: mean predicted 0.905, observed 0.946); the next largest are 0.031 (bin 8: 0.739 vs 0.770), 0.024 (bin 5: 0.273 vs 0.248) and 0.015 (bin 2). **No bin's mean prediction falls outside its Wilson 95% interval.** (Gap in the CSV = mean predicted minus observed; positive = over-prediction.)
- The one visible regularity is mild **under-prediction at the top** (observed above predicted by about 4 points in the top decile and 3 points in decile 8, while decile 9 is on the diagonal; students with p > 0.80 are placed 90.5% of the time against a mean prediction of 87.7%). The same pattern was seen in Phase 5 (top decile 0.909 predicted vs 0.949 observed), so it is a consistent small feature of the model rather than a test-only surprise. It is within sampling noise on this sample size and **no recalibration was applied or tested**; it is reported as a limitation.
- Probabilities are therefore acceptably calibrated on test, in line with Phase 5. As in Phase 5, a probability describes a **group** of similar students (for example decile 9: mean prediction 0.843, observed placement rate 0.838), not a guarantee for an individual.

**Probability distribution (LR_B, test; `08_test_probability_distribution_summary.csv`, `phase7_test_probability_distribution.png`).**
| | Test (refit model) | Phase 5 train OOF |
|---|---|---|
| Minimum / maximum | 0.007 / 0.942 | 0.005 / 0.949 |
| Mean / median | 0.411 / 0.334 | - / 0.349 |
| Q1 / Q3 | 0.100 / 0.740 | 0.097 / 0.748 |
| Share below 0.20 | 38.9% (n = 579) | 38.9% |
| Share 0.20 - 0.80 | 42.0% (n = 624) | 40.2% |
| Share above 0.80 | 19.1% (n = 284) | 20.8% |
| Observed placed rate, p < 0.20 | 8.1% (mean predicted 8.2%) | 8.4% |
| Observed placed rate, p > 0.80 | 90.5% (mean predicted 87.7%) | 89.5% |
| Observed placed rate, 0.20 - 0.80 | 50.3% | 48.8% |

The two distributions are very similar; the differences (about 2 points per band) are small relative to sampling variation and are not read as deterioration. About 58% of test students fall in a confident band, and in both bands the observed rate is within about 3 points of the mean prediction. The middle band is placed about half the time, i.e. genuinely uncertain, as in Phase 5.

## D. Threshold 0.50 (reference / default), LR_B, test
| Accuracy | Precision | Recall = Sensitivity | F1 | Specificity | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.812 (0.792-0.831) | 0.779 (0.746-0.811) | 0.765 (0.732-0.797) | 0.772 (0.744-0.797) | 0.846 (0.821-0.869) | 473 | 735 | 134 | 145 |

The model labels 40.8% of test students "Placed" against a true rate of 41.6%.

## E. Threshold 0.35 (selected operating threshold under the F1-oriented criterion), LR_B, test
| Accuracy | Precision | Recall = Sensitivity | F1 | Specificity | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.802 (0.781-0.820) | 0.722 (0.689-0.754) | 0.851 (0.822-0.878) | 0.781 (0.756-0.804) | 0.766 (0.737-0.793) | 526 | 666 | 203 | 92 |

The model labels 49.0% of test students "Placed".

**Threshold comparison (LR_B; only the two pre-declared thresholds were evaluated).**
| Threshold | Accuracy | Precision | Recall | F1 | Specificity | Sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.812 | 0.779 | 0.765 | 0.772 | 0.846 | 0.765 |
| 0.35 | 0.802 | 0.722 | 0.851 | 0.781 | 0.766 | 0.851 |

**Trade-off.** Lowering the threshold from 0.50 to 0.35 converts 53 false negatives into 69 additional false positives: recall rises by 8.6 points (0.765 to 0.851), precision falls by 5.7 points, specificity falls by 8.0 points, accuracy by 1.1 points, and F1 differs by only 0.009 (the F1 intervals overlap almost completely; no paired interval for the F1 difference was computed, so no difference in F1 is claimed). On the final test set, the 0.35 threshold produced higher recall and F1 than 0.50, while also producing lower precision and specificity (0.35 was selected in Phase 6 by the predeclared validation F1-oriented rule; 0.50 is retained as the conventional reference threshold). This reproduces the Phase 6 picture: 0.35 favours recall, 0.50 is the more conservative positive-classification rule. **Neither threshold is universally preferable**: the right choice depends on the relative cost of a false "likely placed" versus a missed placed student, which this project has not defined. The probability, not the label, remains the primary output.

## F. Secondary models (descriptive only; frozen Phase 4 configurations refitted on train + validation)
| Model | ROC-AUC (95% CI) | Log loss | Brier | F1 @ 0.50 | F1 @ 0.35 | Calibration slope | ECE |
|---|---|---|---|---|---|---|---|
| **LR_B (primary)** | 0.886 (0.868-0.902) | 0.420 | 0.134 | 0.772 | 0.781 | 1.05 | 0.016 |
| GB_B | 0.886 (0.868-0.902) | 0.419 | 0.134 | 0.752 | 0.778 | 1.06 | 0.031 |
| RF_B | 0.886 (0.868-0.902) | 0.420 | 0.135 | 0.756 | 0.778 | 1.15 | 0.024 |

| Paired bootstrap difference (LR_B minus other) | ROC-AUC | Log loss | Brier |
|---|---|---|---|
| vs GB_B | -0.000 (-0.003, +0.003) | +0.001 (-0.004, +0.006) | -0.000 (-0.003, +0.002) |
| vs RF_B | -0.000 (-0.004, +0.004) | -0.001 (-0.007, +0.007) | -0.001 (-0.004, +0.001) |

- On the three primary metrics the three models are **indistinguishable** on test (AUC 0.8857-0.8859; log loss 0.4186-0.4204; Brier 0.1335-0.1347; every paired interval includes 0), which continues the Phase 4/5 tie. At 0.50, `GB_B` and `RF_B` have lower recall (0.727 / 0.725 vs 0.765) at similar precision, hence lower F1 (0.752 / 0.756 vs 0.772); at 0.35 all three have F1 about 0.78. These F1 differences were not tested and are descriptive. `RF_B` again has a calibration slope above 1 (1.15: probabilities slightly too moderate), as in Phase 5 (1.08).
- **No model is declared superior from these numbers, and `LR_B` is not replaced.** The primary model was frozen before the test set was opened; test differences this small are within sampling noise.

## G. Development vs test (LR_B)
| | Phase 5 train 5-fold CV (mean; fold SD) | Phase 4 validation (single split, train-only model) | Phase 7 TEST (single split, train + validation model) |
|---|---|---|---|
| ROC-AUC | 0.879 (SD 0.0095) | 0.872 (0.854-0.890) | **0.886** (0.868-0.902) |
| Log loss | 0.431 (SD 0.0157) | 0.441 (0.412-0.470) | **0.420** (0.394-0.449) |
| Brier | 0.139 (SD 0.0059) | 0.141 (0.130-0.152) | **0.134** (0.124-0.145) |
| Calibration slope / intercept | 0.99 / -0.002 | 0.98 / +0.039 | 1.05 / +0.041 |
| ECE (10 equal-count bins) | 0.021 | 0.024 | 0.016 |

(`results/metrics/modeling/08_development_vs_test.csv`; the development numbers are read from the saved Phase 4/5 files.)

- **Broadly consistent with the development estimates.** Test metrics fall within approximately one CV standard deviation of the corresponding CV means. Test AUC is 0.007 above the CV mean (0.7 fold SDs) and 0.014 above validation; test log loss and Brier are slightly *better* than both development estimates (0.7 and 1.0 fold SDs). The test 95% intervals contain both the CV mean and the validation value for all three metrics. Fold SDs (folds of about 1,390 students) are a rough yardstick for single-split sampling variation at the test size (1,487), so all differences are of the size expected from sampling alone.
- **Reading the higher test performance.** It does not by itself establish improved generalisation and may reflect sampling variation and the larger train + validation refit. The comparison cannot separate a lucky or easy test split from the 21% extra training data of the refit model (a small gain is plausible for a 15-parameter model), and the validation split's slight optimism (it informed Phase 4 selection) is not clearly visible either. The honest reading is "consistent with the development estimates", not "the model is better than expected".
- **Calibration remains reasonable**: slope and intercept compatible with 1 and 0, ECE below its noise yardstick, as in development. The small top-end under-prediction persists (Section C).
- Nothing here is labelled overfitting: the differences are not evidence of it, and the development-phase train-validation gaps (Phase 4) were small for LR_B.

## H. Limitations
- **Dataset.** One dataset, one split, 1,487 test students: intervals are about +/-0.017 for AUC and +/-0.02 for accuracy. Phase 2 found 8 conflicting-label profiles in the top region; error rates near 20% at 0.50 are consistent with a performance ceiling for these ten features (Phase 6).
- **Possible constructed / rule-like structure.** The observed structure (bounded scores with hard floors and ceilings, step-like and non-monotonic count effects, very strong associations, a model-family tie, conflicting labels) is **consistent with a rule-based or constructed dataset, although this is not established by the dataset source.** **Strong performance on this dataset does not establish equivalent performance on real-world university placement populations, other campuses or other years.**
- **Validation-selection optimism.** Model, representation and threshold were chosen on validation (and Phase 3 EDA informed the categorical-count representation), so development figures at the chosen settings are slightly optimistic; the test split is the independent check, but it is a single sample.
- **Test rows in earlier descriptive phases.** The Phase 1-3 notebooks loaded the full data, including the rows later assigned to the test split (notebooks 03 and 04 also loaded `modeling_dataset.csv` and `test.csv` for descriptive and split-sanity analyses; disclosed in notebook 05). No predictive model was fitted or evaluated on test rows before Phase 7, but the descriptive analyses that motivated treating counts as categories included test rows, so test independence is not absolute. The Phase 4-6 audit hooks describe those notebooks only, not the whole project history.
- **Threshold dependence.** Classification metrics depend on the threshold; 0.35 was selected under an F1 criterion that weights errors equally. The 0.35 vs 0.50 F1 difference is small relative to its interval; the costs of the two error types were never defined.
- **Refit vs development model.** Test results describe the train + validation refit; validation and CV numbers describe train-only models.
- **Calibration.** Small, consistent top-end under-prediction (about 3-4 points in deciles 8 and 10) was observed and deliberately not recalibrated.
- **Observational / non-causal.** Coefficients and importance describe associations learned by the model; they do not say that changing a feature (e.g. attending placement training) would change an outcome.
- **Test-access disclosure (process deviation).** During orientation, before the notebook (and its audit hook) ran, two shell commands touched `data/processed/test.csv` outside the hook: `head -2` (displaying the header and the first data row) and `wc -l data/processed/*.csv` (line counts only; the count matches the documented 1,487 + header). Neither result was used in any decision (every decision was already frozen and recorded), but it means the notebook's hook attests only to what happened inside the notebook process.

## I. Final conclusion
**Decision.** No methodological problem occurred inside the pre-registered evaluation (single open, frozen specification asserted identical, no tuning after opening, processed/raw data and Phase 4 artifacts byte-identical). **For this dataset and experimental protocol, Logistic Regression B was retained as the final model.** It is final only in that sense: it is not claimed to be the best model in general, universally superior, or guaranteed to generalise, and it does not show that any feature causes placement.

**What the project can legitimately claim, based on the complete experiment.**
1. On this dataset, with a leakage-controlled split (duplicates collapsed, conflicting-label profiles excluded, zero profile overlap), a logistic regression on ten recorded variables, with the count variables treated as categories, ranks held-out students with ROC-AUC about 0.89 (95% interval 0.87-0.90), log loss 0.42 and Brier 0.134, against 0.50 / 0.68 / 0.24 for a base-rate predictor.
2. Its raw probabilities are acceptably calibrated (slope 1.05, ECE 0.016); no recalibration was needed, and the small top-end under-prediction is a stated limitation.
3. Gradient boosting and random forest give statistically indistinguishable test performance; the extra complexity bought nothing measurable, so the interpretable model was retained.
4. At the reference threshold 0.50 the model reaches accuracy 0.81, precision 0.78 and recall 0.77; at the pre-declared 0.35 it trades some precision for recall (0.72 / 0.85) with a slightly higher but not formally compared F1 (0.781 vs 0.772). Which is appropriate depends on costs the project has not defined.
5. Test performance was broadly consistent with the development estimates (within approximately one CV standard deviation of the CV means); the higher test performance does not by itself establish improved generalisation and may reflect sampling variation and the larger train + validation refit.
6. The associations the model uses (aptitude, placement training, extracurricular activities and the non-monotonic count variables dominate) are descriptive of this dataset and are not causal.

**What it cannot claim.** Equivalent performance on real placement populations (the dataset's structure is consistent with a constructed or rule-based origin, not established), individual-level certainty (about 20% of students are misclassified at 0.50 and some confident predictions are wrong), or that any threshold or any model is best in general.
