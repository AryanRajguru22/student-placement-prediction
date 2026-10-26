> **Evidence labels.** *Train CV* = 5-fold cross-validation on the training set only (n = 6,938). *Nested* = out-of-sample recalibration, calibrators fitted only on data that excludes the evaluated student. *Validation confirmation* = one evaluation on the validation set (n = 1,487) with calibrators fitted on train only. **`test.csv` was never opened** (a hook raises an error on any attempt; the audit above lists every data file read). Associations only; no causal claims. Hyper-parameters for GB and RF were chosen on the validation split in Phase 4, so train-CV estimates for those two are not fully independent of validation.

### Model stability: is the Phase 4 tie real?
| 5-fold CV on TRAIN (mean +/- SD across folds) | ROC-AUC | Log loss | Brier | F1 (secondary) |
|---|---|---|---|---|
| Logistic regression (B) | 0.879 +/- 0.010 | 0.431 +/- 0.016 | 0.139 +/- 0.006 | 0.756 |
| Gradient boosting (B) | 0.876 +/- 0.011 | 0.434 +/- 0.020 | 0.141 +/- 0.008 | 0.750 |
| Random forest (B) | 0.876 +/- 0.010 | 0.433 +/- 0.015 | 0.140 +/- 0.006 | 0.746 |

- **Cross-validation supports the Phase 4 conclusion: the three models are practically tied.** The gap between the best and worst fold-mean is only about 0.2-0.3 of the typical fold-to-fold standard deviation for every primary metric (AUC 0.003 vs SD 0.010; log loss 0.004 vs 0.017; Brier 0.001 vs 0.006).
- Logistic regression has the best mean on all three primary metrics, but the margins are tiny. Paired fold differences (LR minus other; negative is better for log loss/Brier): vs GB, AUC +0.003 (LR higher in 5/5 folds), log loss -0.004 (4/5), Brier -0.001 (3/5); vs RF, AUC +0.003 (5/5), log loss -0.002 (4/5), Brier -0.001 (4/5). **Every Nadeau-Bengio-corrected 95% interval includes 0** (for example LR-GB log loss -0.011 to +0.004). The uncorrected t-intervals for AUC alone just exclude 0, but they ignore that fold training sets overlap and rest on five folds, so they should not be over-read.
- The most that can be said: LR shows a *small, directionally consistent* AUC edge (all five folds), which is not distinguishable from noise under a conservative interval. Nothing here supports declaring a winner.
- The Phase 4 validation AUC for LR (0.872) lies inside the CV range (0.868-0.890), so the validation split did not look unusually favourable.

### Calibration of the raw probabilities
| Train OOF (out-of-sample) | Slope (95% CI) | Calibration-in-the-large | ECE (10 equal-count bins) | ECE if perfectly calibrated (mean / 95th pct) |
|---|---|---|---|---|
| Logistic regression | 0.99 (0.95-1.03) | -0.000 | 0.021 | 0.011 / 0.016 |
| Gradient boosting | 0.98 (0.93-1.02) | -0.002 | 0.016 | 0.011 / 0.015 |
| Random forest | **1.08 (1.03-1.13)** | -0.001 | 0.016 | 0.011 / 0.016 |

- **The raw probabilities are already reasonable for all three models.** Average predicted probability equals the observed placement rate (41.6%), and slopes are close to 1.
- **Logistic regression and gradient boosting** have slopes statistically compatible with 1 (neither over- nor under-dispersed). **Random forest** has a slope above 1 (CI excludes 1): its probabilities are slightly *too moderate*, with the top decile predicted 0.934 but observed 0.955 and the second-lowest decile over-predicted (0.067 vs 0.042).
- **Local departures are small but visible.** For logistic regression, the 52-68% range is over-predicted by about 5 percentage points (bin 7: predicted 0.605, observed 0.550), the 21-35% range under-predicted by about 4 points (0.276 vs 0.313), and the top decile under-predicted by about 4 points (0.909 vs 0.949); the lowest decile is over-predicted (0.022 vs 0.014). Gradient boosting shows the same pockets (0.583 vs 0.550; 0.270 vs 0.314). Several of these (LR bins 5, 7 and 10) lie just outside the single-bin 95% interval, which is consistent with LR's ECE slightly exceeding the noise yardstick, but the departures are small in absolute terms (at most about 5-6 points).
- **ECE, read against its noise yardstick:** logistic regression's training-OOF ECE (0.021) is modestly above what sampling noise alone would give for perfect calibration (95th percentile 0.016), whereas GB and RF are at the boundary. On the **validation** set, **all ECEs fall inside the noise range** (for example LR 0.024 vs null 95th percentile 0.034). So miscalibration is small and only weakly detectable; ECE is supplementary and bin-dependent.

### Does recalibration help? (nested train analysis, then validation confirmation)
| Change vs original | LR log loss | GB log loss | RF log loss | AUC change |
|---|---|---|---|---|
| Sigmoid, train nested | -0.0000 (CI -0.0001, +0.0001) | -0.0000 (-0.0004, +0.0003) | -0.0006 (-0.0015, +0.0002) | about 0 |
| Isotonic, train nested | -0.0008 (-0.0034, +0.0018) | +0.0015 (-0.0008, +0.0037) | **+0.0029 (+0.0006, +0.0052), worse** | -0.0015 to -0.0017 |
| Sigmoid, validation | -0.0000 | -0.0001 | -0.0005 | about 0 |
| Isotonic, validation | +0.0012 | +0.0015 | +0.0031 | -0.0006 to -0.0011 |

- **Sigmoid (Platt) calibration changes almost nothing.** It moves slopes to 1.00 by construction but log loss and Brier move by at most 0.0006, with intervals including 0, and it improves log loss in only 2-3 of 5 folds. The effect is **negligible**.
- **Isotonic calibration does not help.** It lowers ECE (about 0.009 for all three), but ECE is the most binning-sensitive measure, whereas log loss and Brier are unchanged or *worse* (significantly worse for random forest). It also lowers AUC slightly (about 0.0015) because it collapses predictions into about 215-240 distinct values (ties), and it assigns the clipped extreme probabilities 0.001/0.999 to 0.8-2.5% of students, which is over-confident at the extremes. With about 5,500 calibration points per model it adds many parameters for no measurable gain.
- **No method met the pre-declared materiality rule** (log-loss gain of at least 0.005 with an interval below zero) for any model, on train or validation. Sigmoid cannot repair the logistic regression's small local wiggles because they are not logistic-shaped, and isotonic's flexibility costs as much variance as it removes bias.

### Where the probabilities sit, and whether confidence is justified (train OOF; `06_probability_distribution_summary.csv`)
| Original probabilities | Min / max | Quantiles 5 / 25 / 50 / 75 / 95% | Below 0.2 | 0.2-0.8 | Above 0.8 | Observed placed rate: below 0.2 / above 0.8 |
|---|---|---|---|---|---|---|
| Logistic regression | 0.005 / 0.949 | 0.022 / 0.097 / 0.349 / 0.748 / 0.909 | 38.9% | 40.2% | 20.8% | 8.4% (mean predicted 8.0%) / 89.5% (87.7%) |
| Gradient boosting | 0.008 / 0.976 | 0.029 / 0.103 / 0.337 / 0.738 / 0.935 | 39.1% | 40.6% | 20.3% | 9.0% (8.7%) / 89.0% (89.5%) |
| Random forest | 0.006 / 0.979 | 0.027 / 0.119 / 0.361 / 0.692 / 0.935 | 34.8% | 47.8% | 17.3% | 7.7% (8.6%) / 91.1% (89.7%) |

- **Most students receive a confident probability.** About 60% (LR, GB) to 52% (RF) fall below 0.2 or above 0.8, but no model assigns anything beyond about 0.005 or 0.98 (minimum 0.005-0.008, maximum 0.95-0.98). Random forest places more students in the uncertain 0.2-0.8 middle (47.8% vs about 40%): it is the most hedged of the three, consistent with its slope above 1.
- **Extremes are justified here, but by the reliability evidence, not by their extremity.** In the bands below 0.2 and above 0.8, the observed placement rate is within about 2 percentage points of the mean predicted probability for all three models. A probability of 0.9 is therefore not an over-claim in this dataset. That is a statement about calibration, not about discrimination: confident predictions can still be wrong for individual students (roughly 1 in 10 students above 0.8 is not placed, and about 1 in 12 below 0.2 is placed).
- **The uncertain middle is genuinely uncertain.** Students predicted 0.2-0.8 are placed about 48-49% of the time under every model, so this band carries little more than coin-flip information; this is where the overlap between the Placed and NotPlaced distributions seen in Phase 4 lives.
- No model produces the coarse probability levels of the Phase 4 decision tree, and none produces 0/1 extremes.

### Probability quality: which candidate is most defensible?
- All three produce acceptably calibrated probabilities; none needs recalibration to be usable.
- **Logistic regression** has the best mean log loss, Brier and AUC in CV, a calibration slope of 0.99, the smallest train-validation gap in Phase 4, and directly interpretable coefficients. It is the most defensible *provisional* candidate, not a proven superior one.
- Gradient boosting is equally well calibrated and statistically tied, but offers no probability-quality advantage that would justify opacity. Random forest is slightly under-dispersed (slope 1.08) and had the largest train-validation gap in Phase 4.
- **Systematic regions:** modest over-prediction in the mid-to-upper range (about 52-68%) and under-prediction at the highest (> 0.9) and 21-35% ranges for LR/GB; at the extremes the RF is too timid. These are at most about 5-6 points and partly within sampling noise.

### Provisional model
- **Logistic regression with categorical counts (LR_B) remains the provisionally preferred model.** Phase 5 gives no evidence to replace it, and cross-validation supports the earlier tie, so the choice rests on interpretability and consistency rather than on a demonstrated accuracy advantage.
- **Calibration need not be part of the final pipeline**: the raw LR probabilities are acceptably calibrated, and sigmoid/isotonic add complexity without measurable benefit. If a final pipeline later includes any calibration step it should be sigmoid, not isotonic, but this phase does not recommend it.
- No final model is declared. The test set remains reserved for one-time evaluation.

### What a predicted probability means (for the report and viva)
A predicted probability of 0.80 does **not** mean that this particular student has an 80% chance of being placed. It supports the group-level statement: *among students who receive predicted probabilities around 0.80, and who resemble the students in this dataset, roughly 80% are expected to be placed.* The reliability diagrams check exactly this, and the statement holds only approximately (within a few percentage points in the mid-to-upper range). The model uses ten recorded variables from one, possibly rule-generated, campus dataset and describes association, not what would happen if a student changed a characteristic.

### Limitations
- Five folds give coarse stability estimates; fold training sets overlap, and the bootstrap intervals for recalibration treat pooled OOF predictions as independent, so they understate uncertainty slightly.
- OOF/CV models are trained on 80% of the training data; the final models use 100%. Calibrators fitted on OOF probabilities and applied to full-train models carry this small mismatch.
- ECE and reliability bins depend on the binning (10 equal-count bins here); ECE should be read with its noise yardstick.
- Validation (n = 1,487, about 149 per bin) cannot detect small miscalibration; its CITL interval (about +/-0.14 on the log-odds scale) is wide.
- GB/RF hyper-parameters were selected on validation in Phase 4; results describe one dataset and need not transfer to other institutions.

### Recommendation for Phase 6
1. **Freeze the pipeline before touching the test set:** LR_B (Phase 4 pipeline, C = 1, categorical counts), raw probabilities, **no recalibration**, fitted on the **training set only** (the exact artifact that was selected and validated; refitting a 16-parameter model on 21% more data is expected to change little).
2. **Pre-register the evaluation:** one pass over `test.csv` reporting ROC-AUC, log loss, Brier, calibration slope/CITL, a reliability diagram with Wilson intervals and its ECE noise yardstick, and bootstrap intervals; GB_B and RF_B may be reported as pre-declared secondary comparators with **no re-selection** based on the outcome.
3. **Decide any classification threshold before Phase 6** from train OOF or validation predictions (the default 0.5 is acceptable if no cost trade-off is stated); do not tune on test.
4. **Interpretation work** (permutation importance/SHAP, coefficient stability) can be done on train/validation before or alongside the final evaluation, but must not influence model choice after the test is seen.
