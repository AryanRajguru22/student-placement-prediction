> All numbers are **validation** results (n = 1,487; Placed rate 41.56%) at the default 0.5 threshold. The test set was never opened (asserted below). Everything is association/prediction, not causation. Because hyper-parameters and the provisional model were chosen on this same validation split, the chosen model's validation scores are slightly optimistic.

### A. Validation model comparison
| Model (selected config) | Rep. | ROC-AUC | Log loss | Brier | F1 | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| Logistic regression (C=1) | A numeric | 0.869 | 0.446 | 0.142 | 0.762 | 0.757 | 0.803 |
| **Logistic regression (C=1)** | **B categorical** | **0.872** | **0.441** | **0.141** | 0.756 | 0.741 | 0.802 |
| Decision tree (A: depth 3; B: depth 5, min leaf 5) | A / B | 0.827 / 0.846 | 0.495 / 0.480 | 0.161 / 0.155 | 0.721 / 0.743 | 0.725 / 0.777 | 0.767 / 0.776 |
| Random forest (300 trees) | A / B | 0.868 / 0.869 | 0.445 / 0.443 | 0.144 / 0.143 | 0.755 / 0.749 | 0.723 / 0.718 | 0.805 / 0.800 |
| Gradient boosting (depth 2) | A / B | 0.870 / 0.871 | 0.442 / 0.441 | 0.142 / 0.142 | 0.752 / 0.758 | 0.725 / 0.731 | 0.801 / 0.806 |
| Reference: always predict train rate | - | 0.500 | 0.679 | 0.243 | 0 | 0 | 0.584 |

- Logistic regression, random forest and gradient boosting are **practically tied**: ROC-AUC 0.868-0.872 and log loss 0.441-0.446, far inside each model's 95% bootstrap CI (for example AUC CI roughly 0.854-0.890, log loss 0.412-0.470). Only the single decision tree is clearly worse (log loss 0.48-0.50; paired CI for the gap to the best excludes 0).
- All models beat the base-rate reference substantially (Brier skill about 0.42 versus the base rate; log loss 0.44 versus 0.68). A random Placed student receives a higher predicted probability than a random NotPlaced student about 87% of the time.
- The class-balanced logistic regressions have the same AUC but slightly worse log loss and Brier (0.447-0.452 and 0.144-0.146) and lower precision; they trade precision for recall (0.79-0.80) by design and were kept as comparison only.
- Accuracy (about 0.80) is similar across the competitive models and is not the deciding metric.

### B. Numeric (A) vs categorical (B) count representation
Paired bootstrap, B minus A, at each family's selected configuration:
| Family | dAUC (95% CI) | dLog loss (95% CI) | Reading |
|---|---|---|---|
| Logistic regression | +0.002 (-0.001, +0.006) | -0.005 (-0.010, +0.000) | B slightly better; borderline, not clearly distinguishable |
| Decision tree | +0.019 (+0.010, +0.028) | -0.015 (-0.030, -0.001) | B clearly better |
| Random forest | +0.001 | -0.002 (-0.006, +0.001) | no distinguishable difference |
| Gradient boosting | +0.000 | -0.000 (-0.003, +0.002) | no distinguishable difference |
- Treating the counts as categories matters for the shallow tree, helps the logistic regression marginally (consistent with the non-monotonic Workshops/Projects patterns found in Phase 3), and is irrelevant for ensembles that can already split on thresholds.
- The effect is small relative to validation noise, so the representation decision is a mild preference, not a proven difference.

### C. Provisional model selection (validation only)
**Logistic regression with categorical counts (Variant B), L2, C = 1.0, no class weighting** (`models/phase4/LR_B.joblib`; selection record in `results/metrics/modeling/05_provisional_selection.json`).

### D. Why it was selected
Rule fixed in advance: among eligible models, take the lowest validation log loss, then treat all models whose paired-bootstrap CI for the log-loss difference includes 0 as tied and choose the most interpretable.
1. **ROC-AUC / log loss / Brier:** LR-B has the best log loss (0.4408) and Brier (0.1409) and the highest AUC among the non-balanced models (0.8717); gradient boosting (0.4413) and random forest (0.4432) are statistically indistinguishable.
2. **F1/recall:** F1 0.756 and recall 0.741 are in line with the other competitive models (sanity check only; threshold untuned).
3. **Train-validation gap:** smallest of the competitive models (AUC +0.009, log loss +0.013).
4. **Interpretability:** coefficients and odds ratios are directly readable, which matters for explaining a placement probability.
5. **Sensible probabilities:** mean predicted probability 0.410 versus observed 0.416; no pile-up of extreme probabilities beyond about 4% below 0.02 or above 0.98.
Nothing more complex earned its extra complexity on this validation split. This is provisional: it rests on one split of 1,487 students and a search run on the same split.

### E. Train-validation gap
| Model | Train AUC | Val AUC | AUC gap | Log-loss gap | Flag |
|---|---|---|---|---|---|
| LR-A / LR-B | 0.878 / 0.881 | 0.869 / 0.872 | 0.008 / 0.009 | 0.012 / 0.013 | ok |
| DT-A / DT-B | 0.844 / 0.870 | 0.827 / 0.846 | 0.016 / 0.024 | 0.020 / 0.041 | ok |
| GB-A / GB-B | 0.887 / 0.890 | 0.870 / 0.871 | 0.017 / 0.019 | 0.025 / 0.030 | ok |
| RF-B | 0.913 | 0.869 | 0.044 | 0.066 | suspicious |
| **RF-A** | 0.924 | 0.868 | **0.056** | **0.089** | **large (excluded)** |
The random forests memorise the training data more than the other models do (training AUC above 0.91) for no validation benefit; RF-A was excluded from selection. Unregularised deep decision trees (depth None, leaf 1) had validation log loss above 11, showing why probability quality and overfitting must be checked, not just accuracy.

### F. Probability-quality observations
- Predicted probabilities from LR, RF and GB are broad, not just near 0 or 1; about 41-49% of validation students receive a probability between 0.2 and 0.8, and the overlap coefficient between the Placed and NotPlaced distributions is about 0.40 for LR/GB. Even the best models therefore cannot cleanly separate the groups, and a high AUC does not mean near-perfect individual prediction.
- The decision tree produces a few coarse probability levels (leaf proportions), which explains its worse log loss and Brier score despite similar accuracy.
- Random forest places 3-6% of students at extreme probabilities (below 0.02 or above 0.98); whether that confidence is warranted cannot be judged from these metrics and needs the calibration analysis planned for Phase 5.
- Mean predicted probability is within 0.6 percentage points of the observed rate for every model.
- **Logistic-regression coefficients (associations, not causal effects):** holding the other model variables constant, the fitted model associates PlacementTraining = Yes (OR 2.7 vs No) and ExtracurricularActivities = Yes (OR 1.9 vs No) with higher placement odds, and one training-SD higher aptitude (OR 1.65), SSC (1.33), soft skills (1.32), CGPA (1.29) and HSC (1.23) with higher odds. With categorical counts, Workshops = 1 is associated with lower odds than Workshops = 0 (OR 0.73, reference 0), and Projects = 1 and 2 with lower odds than Projects = 3 (OR 0.66, 0.73); the Projects = 0 coefficient (OR 1.65 vs 3) rests on only 19 training students and should not be interpreted. Internships has little association once the other variables are held constant (numeric OR 1.01 per SD; categorical levels 0 and 2 both slightly above level 1), unlike its strong raw association in the EDA.
- **Tree importances (not causal):** aptitude, HSC marks and extracurricular activities carry most of the importance in the ensembles (aptitude about 0.17-0.20, HSC about 0.18-0.25, extracurricular about 0.10-0.15); Internships and PlacementTraining receive little tree importance. In the depth-3 tree, HSC marks account for about two thirds of the total impurity reduction.

### G. Remaining questions for Phase 5 / Phase 6
1. **Calibration:** how well calibrated is each model (reliability curves, calibration slope/intercept)? Would Platt or isotonic recalibration, fitted without the validation labels, improve log loss?
2. **Robustness of the selection:** does stratified cross-validation on the training data (with repeated folds) confirm that logistic regression, gradient boosting and random forest remain tied, and is the A-versus-B preference stable?
3. **Stronger interpretation:** permutation importance and/or SHAP (with correlated predictors in mind), and stability of logistic coefficients across resamples.
4. **Decision threshold:** if a classification rule is needed, which threshold balances precision and recall for the intended use? (Deferred; not tuned here.)
5. **Non-monotonic count effects:** do tree models gain from interactions (Extracurricular x Training, Projects x Workshops) that a main-effects logistic model misses? Would adding interactions to the logistic model close any gap?
6. **Performance ceiling and subgroup behaviour:** how do models perform within subgroups (e.g. top-profile region where conflicting labels were found, small levels such as Workshops = 3)? How much of the remaining error looks irreducible?
7. **Generalisation caveat:** results describe this single (possibly rule-generated) dataset; the held-out test set remains untouched for the one-time final evaluation in Phase 6.
