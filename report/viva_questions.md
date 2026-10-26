# Viva Preparation — Predicting Student Placement Probability

All answers use results saved in the project and reported in `report/final_report.md`. Section numbers (§) refer to that report. No new analysis was run to write this file.

## Thirty-second summary

I estimated, from ten recorded student characteristics, the probability that a student is placed. I audited the data first (72 duplicate surplus rows collapsed, 16 conflicting-label rows excluded, 33 profiles that crossed the supplied splits), so I built a new leakage-free stratified split (6,938 / 1,487 / 1,487). Logistic regression, gradient boosting and random forest were practically tied on probability quality; I retained Logistic Regression B (`LR_B`) for its interpretability, with raw probabilities (no recalibration met the predefined criterion). The threshold 0.35 was selected on validation; 0.50 is the reference. In one final held-out evaluation, `LR_B` obtained ROC-AUC 0.886, log loss 0.420 and Brier 0.134, with calibration slope 1.05. Everything is association, not causation, in one dataset that may be constructed.

## Numbers to know

| Item | Value |
|---|---|
| Original dataset | 10,000 rows, 11 columns; 4,197 Placed / 5,803 NotPlaced |
| Cleaning | 72 same-label duplicate surplus rows collapsed; 16 rows in 8 conflicting-label groups excluded → 9,912 rows |
| Leakage | 33 feature profiles crossed the original supplied files |
| Final split | Train 6,938 / Validation 1,487 / Test 1,487 (placed rate 41.56% in each) |
| Validation, `LR_B` | AUC 0.872, log loss 0.441, Brier 0.141 |
| 5-fold CV, `LR_B` | AUC 0.879 ± 0.010, log loss 0.431 ± 0.016, Brier 0.139 ± 0.006 |
| Calibration (development) | slope ≈ 0.99, ECE ≈ 0.021 |
| Threshold | 0.35 selected (validation F1 0.766); 0.50 reference (validation F1 0.756) |
| Final test (n = 1,487) | AUC 0.886 (0.868–0.902), log loss 0.420 (0.394–0.449), Brier 0.134 (0.124–0.145); slope 1.05, ECE 0.016 |
| Test at 0.50 | accuracy 0.812, precision 0.779, recall 0.765, F1 0.772; TP/TN/FP/FN 473/735/134/145 |
| Test at 0.35 | accuracy 0.802, precision 0.722, recall 0.851, F1 0.781; TP/TN/FP/FN 526/666/203/92 |
| Odds ratios (model) | placement training ≈ 2.75, extracurricular ≈ 1.95, aptitude ≈ 1.65 per SD |

---

## A. Project basics and metrics

**1. Why did you choose probability prediction instead of binary classification?**
A binary label hides how certain the model is: two students with estimated probabilities of 0.92 and 0.55 both get "Placed" at 0.50. A probability keeps that information, lets different users choose different thresholds for their own error costs, and can be checked for calibration. The probability is the primary output; the classification is secondary and derived from a threshold (§2–3).

**2. Why is accuracy not your primary metric?**
Accuracy depends on one threshold, ignores how confident the predictions are, and is weak under moderate imbalance (always predicting NotPlaced gives about 58.4%). The project's goal is a trustworthy probability, so proper scoring rules (log loss, Brier score), ROC-AUC and calibration are primary. Accuracy is reported as secondary (0.812 at 0.50 and 0.802 at 0.35 on test).

**3. What does ROC-AUC = 0.886 mean?**
It is the probability that a randomly chosen Placed student receives a higher estimated probability than a randomly chosen NotPlaced student — about 89% of the time on the test set (95% CI 0.868–0.902). It measures ranking, does not depend on a threshold or on calibration, and is **not** an accuracy: 0.886 does not mean 88.6% of students are classified correctly.

**4. What does log loss = 0.420 mean?**
Log loss is the average negative log of the probability the model gave to the outcome that actually happened. It heavily penalises confident wrong predictions, and lower is better. The base-rate predictor scores 0.679, so 0.420 is substantially better; the test 95% CI is 0.394–0.449. It is a proper scoring rule, so it rewards honest, calibrated probabilities.

**5. What does a Brier score of 0.134 mean?**
It is the mean squared difference between the estimated probability and the 0/1 outcome; lower is better and it is also a proper scoring rule. The base-rate predictor has 0.243, so 0.134 is substantially lower (95% CI 0.124–0.145). It combines discrimination and calibration in one number.

**6. What is the difference between ROC-AUC and F1?**
ROC-AUC summarises ranking over all thresholds; F1 is the harmonic mean of precision and recall at one specific threshold and ignores true negatives and error costs. A model can have good AUC and a poor F1 at a badly chosen threshold. That is why AUC is reported with the threshold-free probability metrics and F1 is used only for the threshold analysis.

**7. Can two models have identical AUC but different probability quality?**
Yes. AUC depends only on the ranking of students, so any monotone transformation of the probabilities leaves it unchanged while changing calibration, log loss and Brier. In the project the three models all had test AUC 0.886 but different calibration (slope 1.05 for `LR_B`, 1.06 for `GB_B`, 1.15 for `RF_B`; ECE 0.016, 0.031, 0.024) — which is why calibration was assessed separately.

---

## B. Dataset

**8. Describe the dataset.**
Hugging Face `Krooz/Campus_Recruitment_CSV` (CC0-1.0): 10,000 rows, 11 columns, 4,197 Placed and 5,803 NotPlaced. The ten predictors are CGPA, SSC and HSC marks (academic); aptitude score, internships, projects, workshops/certifications (technical); soft-skills rating, extracurricular activities and placement training (employability). The dataset card gives no provenance (§6).

**9. What is the difference between the original dataset and the modeling dataset?**
The original has 10,000 rows (41.97% Placed). After removing 72 same-label duplicate surplus rows and 16 conflicting-label rows, the modeling dataset has 9,912 rows (41.56% Placed). Everything after the audit — EDA, statistics, modeling, evaluation — uses the 9,912 rows.

**10. What is the difference between validation and test data?**
Validation data are used during development to compare models, choose hyperparameters and the threshold, so they influence decisions and their scores become somewhat optimistic. Test data are held out for one final evaluation of the frozen model and must not influence any modeling decision. Cross-validation on training data gives a third, independent development estimate.

**11. Why use stratified splitting?**
Stratification preserves the class proportion in each split (41.55–41.56% Placed in train, validation and test), so the splits are comparable and metrics such as precision and recall are not distorted by different class balance.

---

## C. Data quality

**12. Why did you remove conflicting-label rows?**
There were 8 groups in which two rows had identical values on all ten predictors but opposite labels. For these profiles the features cannot determine the label, so the label depends on something unrecorded or contains noise. Keeping both would teach the model contradictory targets; keeping one would be arbitrary. All 16 rows were excluded. Honest caveat: they all sit at the top of the profile range, so excluding them may make performance in that region look somewhat better than it would with them included, which is part of why I say performance should not be over-generalised.

**13. Why not use a majority vote?**
Each conflicting group has exactly one Placed and one NotPlaced row, so there is no majority; any resolution would be a coin flip presented as data cleaning. Excluding the group is the transparent option and the excluded rows are listed in `02_conflicting_label_groups.csv`.

**14. What about the duplicates that had the same label?**
There were 66 groups of exact duplicates (138 rows) with consistent labels; 72 surplus rows were collapsed to one record per profile. Leaving them would let identical profiles appear in both training and evaluation data and inflate scores.

---

## D. Leakage

**15. What is data leakage?**
Leakage is any way information from the evaluation data reaches the model-building process, so that performance estimates are optimistic. Here the concern was **duplicate feature-profile leakage**: identical student profiles appearing in both training and validation/test data, so the model is partly scored on profiles it has already seen. A second kind — using test data for decisions — was prevented by the frozen protocol and audit controls (§7–8).

**16. Why did you discard the original split?**
33 feature profiles (69 rows) occurred in more than one of the supplied train/validation/test files, so evaluating on them would partly measure memorisation, and flexible models (random forests, boosting) could benefit more than simple ones, biasing the comparison. I pooled the cleaned 9,912 rows and made a stratified 70/15/15 split with an explicit assertion of zero feature-profile overlap between splits.

---

## E. Statistics

**17. What is the significance of VIF = 1.96?**
The variance inflation factor measures how much a coefficient's variance is inflated by correlation with the other predictors. The maximum VIF was 1.96 (HSC marks with all ten predictors), far below the usual cut-offs of 5 or 10, so multicollinearity is low to moderate. The strongest pair is aptitude and HSC marks (r = 0.56). No variable was dropped.

**18. What is an odds ratio?**
The ratio of the odds of the outcome under one condition to the odds under another, where odds = p/(1 − p). An odds ratio of 1 means no association; above 1 higher odds. In the logistic model it is the multiplicative change in the predicted odds associated with a category versus its reference (or a one-SD increase in a score), holding the other variables fixed.

**19. Does OR = 2.75 mean the placement probability doubles?**
No. OR = 2.75 for placement training means the model's predicted **odds** are multiplied by 2.75, not the probability. The probability change depends on the starting point: illustrative arithmetic — odds 0.25 (p = 0.20) become 0.69 (p ≈ 0.41); odds 1 (p = 0.50) become 2.75 (p ≈ 0.73); odds 9 (p = 0.90) become 24.75 (p ≈ 0.96).

**20. Why is that interpretation wrong?**
Because odds and probability are different scales. At small probabilities a given odds ratio roughly multiplies the probability, but it cannot do so near 1, since probabilities are capped at 1. Also, the odds ratio is a model-conditional association in one observational dataset, not the effect of intervening on training attendance.

**21. Why did you use CMH stratification, Holm correction and effect sizes?**
CMH stratified by aptitude tercile checked whether aptitude explained the count-variable patterns: it shrank the large steps but the non-monotonic dips persisted. Holm correction controlled false positives across many tests. With about 10,000 students nearly everything is significant, so effect sizes (Cohen's d, Cramér's V, odds ratios) and confidence intervals, not p-values, guide interpretation.

---

## F. Preprocessing

**22. Why did you treat count variables categorically?**
EDA and statistics showed non-monotonic patterns (Workshops dip at 1, Internships flat then a jump at 2, Projects step at 3), which a single linear slope would misrepresent. I compared numeric (A) and categorical (B) representations on validation: B helped the decision tree clearly (ΔAUC +0.019), helped logistic regression marginally (ΔAUC +0.002; Δlog loss −0.005, CI −0.010 to +0.000) and made no difference for forest and boosting. So the preference is mild, not proven.

**23. Why standardise continuous variables?**
For L2-penalised logistic regression, standardising puts the five scores on the same scale so the penalty treats them comparably, and it makes coefficients interpretable "per training SD". Tree-based models are invariant to monotone rescaling and use unscaled values. Scalers are fitted on training data only, inside the pipeline.

**24. Why use one-hot encoding?**
One-hot encoding gives each level of Internships, Projects and Workshops its own coefficient, so non-monotonic effects can be represented. Reference levels are Internships = 1, Projects = 3, Workshops = 0, and the binary variables use "No" as reference. This yields 15 coefficients plus an intercept.

---

## G. Logistic regression

**25. What is regularisation?**
A penalty on coefficient size that reduces overfitting. `LR_B` uses an L2 penalty, which shrinks coefficients toward zero. It stabilises estimates for small levels (for example, it pulls the `Projects = 0` coefficient toward zero, since only 19 training students have that level).

**26. What does C = 1 mean?**
In scikit-learn, C is the inverse of regularisation strength: smaller C means stronger shrinkage. C = 1 is the moderate default and was the configuration retained in the validation search; no class weighting was used.

**27. How does logistic regression produce a probability?**
It computes a linear combination of the transformed inputs — the log-odds, logit(p) = intercept + Σ coefficient × value — and converts it through the logistic function p = 1/(1 + e^(−logit)). Because the log-odds are exactly additive, each prediction can be decomposed into per-feature contributions.

---

## H. Model comparison

**28. Why was Logistic Regression B retained?**
By a rule fixed in advance: lowest validation log loss, models whose paired-bootstrap interval for the log-loss difference contains zero are tied, and within the tied set prefer interpretability. `LR_B`, `GB_B` and `RF_B` were practically tied (validation AUC 0.872 / 0.871 / 0.869; log loss 0.441 / 0.441 / 0.443), so I retained the interpretable one. "For this dataset and experimental protocol, Logistic Regression B was retained as the final model" — not a claim of general superiority.

**29. Why not Random Forest?**
It was not better: tied on probability quality, a larger train–validation gap (AUC gap 0.044 for `RF_B`, flagged suspicious; `RF_A` 0.056, excluded), a calibration slope above 1 (1.08 in CV, 1.15 on test; probabilities slightly too moderate), and no interpretable coefficients. The extra complexity bought no measurable gain.

**30. Why not use neural networks?**
They were not tested. With ten tabular features and about 8,400 development rows, three very different model families reached the same performance, which suggests the limit is the information in the features (and label noise), not model flexibility. A neural network would add complexity and lose the transparency that suits a probability tool. That is my expectation; it is not a tested result.

**31. Why did you include a decision tree and a dummy model?**
The tree shows what a transparent nonlinear model gives (clearly worse: validation log loss 0.480, and coarse probability levels). The dummy (always predict the training rate: AUC 0.500, log loss 0.679, Brier 0.243) is a sanity floor that every model must beat.

---

## I. Cross-validation

**32. Why did you use cross-validation, and what did it show?**
Five-fold stratified CV on the training set gives a performance estimate less dependent on one split and a fold-to-fold SD as a yardstick for sampling noise. `LR_B` obtained AUC 0.879 ± 0.010, log loss 0.431 ± 0.016, Brier 0.139 ± 0.006; the three models differed by only 0.2–0.3 of a fold SD, and all Nadeau–Bengio-corrected intervals for the paired differences included zero. It also supplied out-of-fold predictions for calibration and threshold analysis.

**33. What are the limitations of your cross-validation?**
Only five folds; overlapping fold training sets; each fold model uses 80% of the training data; GB/RF hyperparameters were chosen on validation, so their CV scores are not fully independent of it; and bootstrap intervals on pooled out-of-fold predictions treat predictions as independent, understating uncertainty slightly.

---

## J. Calibration

**34. What is calibration?**
A model is calibrated if, among students given a probability near p, a fraction near p are actually placed. It is essential for a probability-focused project: a model could classify well while reporting probabilities that are systematically too extreme or too moderate, which would mislead anyone reading "0.9" as "about nine in ten".

**35. What is the calibration slope?**
The slope from regressing the observed outcomes on the logit of the predicted probabilities. A slope of 1 is ideal; below 1 means predictions are too extreme, above 1 too moderate. `LR_B` had 0.99 (CI 0.95–1.03) out-of-fold and 1.05 (CI 0.95–1.15) on test.

**36. What is ECE?**
Expected calibration error: the average absolute gap between the mean predicted probability and the observed rate across probability bins (10 equal-count bins here). It depends on the binning, so it is read against a noise yardstick — the ECE expected from sampling noise alone if the model were exactly calibrated. Test ECE was 0.016 versus a noise mean of 0.023 (95th percentile 0.034).

**37. Why did you not recalibrate?**
I predeclared a materiality criterion: a recalibration method must improve log loss by at least 0.005 with a confidence interval entirely below zero. Sigmoid changed log loss by about 0.0000 and isotonic did not help (and over-confidently clipped extremes), so no method met the criterion and raw `LR_B` probabilities were retained.

**38. How would you improve calibration?**
First check whether the small top-decile under-prediction (≈ 0.905 predicted vs ≈ 0.946 observed on test) is stable on more data. If it is, options are more data, a recalibration step fitted on separate held-out data, or a slightly more flexible model form (for example, spline terms). Any such change should be justified by the predefined criterion, because recalibration adds parameters and variance.

---

## K. Threshold selection

**39. Why 0.35 instead of 0.50?**
A rule fixed in advance: on a 0.20–0.80 grid, choose the threshold with the highest validation F1 (ties go to the value closest to 0.50). That gave 0.35 (validation F1 0.766). It is the selected operating threshold under an F1-oriented criterion, not "the best" one: F1 is within 0.01 of its maximum from 0.30 to 0.55, and 0.50 is practically tied (ΔF1 +0.010, CI −0.007 to +0.028).

**40. What happens when the threshold decreases?**
More students are labelled Placed: recall rises, precision and specificity fall. On the test set, moving from 0.50 to 0.35 converted 53 false negatives into 69 additional false positives (recall 0.765 → 0.851, precision 0.779 → 0.722, specificity 0.846 → 0.766).

**41. Is 0.35 universally optimal?**
No. It is optimal only under the F1 criterion on validation data, which weights precision and recall equally. The right threshold depends on the relative cost of false positives and false negatives, which this project did not define. No inferential test was done for the F1 difference on test, and no difference is claimed.

**42. How would you choose a threshold if false negatives cost 5× false positives?**
For well-calibrated probabilities, the cost-minimising rule is to predict Placed when p ≥ cost(FP)/(cost(FP) + cost(FN)) = 1/6 ≈ 0.17. That is below the 0.20–0.80 grid I examined, so I would extend the grid, confirm the choice on development data (out-of-fold or validation), and check that calibration holds in the low-probability range. This is a decision-theory calculation from stated costs, not a result from this project.

---

## L. Explainability

**43. What is permutation importance?**
It shuffles one raw column at a time in the full pipeline and measures how much validation performance drops (20 repeats). Aptitude lost 0.0208 AUC, placement training 0.0177 and extracurricular activities 0.0125; Internships lost almost nothing (0.0003). Because predictors are correlated, importance is shared and ranks 5–9 are close, so their order should not be over-read.

**44. Is feature importance causal?**
No. It describes what the fitted model relies on to predict, not what would change an outcome. The same holds for odds ratios, ablation increments and individual contributions: they are associations learned from observational data.

**45. What does an individual contribution mean?**
For one student, logit(p) = intercept + Σ contributions, where each contribution is the coefficient × transformed value relative to a baseline student (all scores at the training average, all categories at their reference). It tells you how the model arrived at its estimate, not what the student could do to change the outcome.

**46. Why is Projects = 0 unreliable?**
Only 30 students in the cleaned data have Projects = 0 (19 in training). Its odds ratio is 1.65 with a 95% interval of 0.63–6.04, which includes 1, and the penalty shrinks the coefficient. The application flags it as rare, and it is not interpreted.

**47. What is ablation, and what does it show?**
Ablation refits the model on feature groups to see what each adds. Validation AUC: academic only 0.823, + aptitude 0.844, + technical counts 0.856, full 0.872; employability alone 0.828. Each added group improved performance in all five CV folds, but the groups overlap, so increments are order-dependent and not independent causal contributions.

---

## M. Final test

**48. Why does the test score exceed the validation score?**
Test AUC 0.886 vs validation 0.872 (log loss 0.420 vs 0.441). Test performance was broadly consistent with development estimates, with test metrics falling within approximately one CV standard deviation of the corresponding CV means. The higher test performance does not by itself establish improved generalization and may reflect sampling variation and the larger train+validation refit. I do not read it as "better than expected".

**49. Why was the final model refit on train + validation?**
After all decisions were frozen, the same specification was refit on 8,425 students (21% more data) to use all non-test labelled data. A train-only clone first reproduced the saved Phase 4 validation probabilities (tolerance 10⁻⁸), and refitting moved coefficients by at most 0.058 log-odds (except `Projects = 0`, 0.111). The caveat: test results describe the refit model, while CV and validation numbers describe train-only models.

**50. Did you use the test set during development?**
The test data were not used for model development, selection, tuning, calibration or threshold selection, and no model was fitted on test rows. But the full picture has three parts: early exploratory notebooks (Phases 1–3) loaded the eventual test rows for descriptive and statistical analysis; two orientation shell commands touched the file before the final run (`head -2` displayed the header and one data row; `wc -l` gave line counts); and Phase 7 accessed the test set once under an audit gate after the protocol was frozen.

**51. Explain the test-access caveat honestly.**
I would say: "The test rows were not used to fit, select, tune, calibrate or choose a threshold. However, my descriptive EDA and statistics included those rows, and those analyses motivated considering a categorical representation of the count variables, which I then chose on validation data. Two shell commands also viewed the file header, a row and line counts before the final run, and nothing was derived from them. So the test split is not independent in the strictest sense, and I report that openly." Controls: protocol written before opening, one gated read, marker file blocking re-execution.

**52. Does early descriptive access invalidate the test?**
It weakens the strictest independence claim, but it does not invalidate the evaluation: descriptive statistics were label-aware summaries of the whole dataset (effects were replicated on the training split alone), no predictive model touched test rows, and the representation was chosen on validation. The risk is a small optimism in how well the representation suits the test rows; a fresh external dataset would be the clean remedy.

---

## N. Application

**53. What does the application do, and how was it verified?**
A Streamlit app takes the ten inputs, validates them, and returns the estimated probability, the classification at 0.35 and at the 0.50 reference, a descriptive probability band, and an additive explanation. It uses the frozen `models/phase7/LR_B_final.joblib` exactly as saved. The verification script confirmed (122 checks in its original Phase 8A form; 178 in the final release form, all passing when rerun at release), among other things, that app probabilities equal direct model predictions, that the thresholds are applied correctly, that the explanation reconstructs the log-odds (error below 10⁻⁹) and that invalid inputs are rejected. It never reads the test split.

**54. Can this model be deployed in a real placement cell?**
Not on this evidence. The data may be constructed, there is no external validation, the threshold depends on undefined costs, and fairness and privacy have not been assessed. It could serve as a decision aid in a pilot after validating on the institution's own data, auditing subgroup performance and defining costs — never as an automated high-stakes decision.

---

## O. Limitations and reflection

**55. What are the major limitations?**
(1) Uncertain representativeness; (2) structure consistent with a rule-based or constructed dataset (not established); (3) a single dataset and (4) a single final split; (5) validation-based selection optimism; (6) a threshold that depends on undefined costs; (7) associations, not causal effects; (8–9) the disclosed early descriptive access to eventual test rows; (10) mild top-decile under-prediction; (11) real outcomes depend on factors not in the data; (12) a probability is not a guarantee for an individual (about 20% of students are misclassified at 0.50).

**56. What ethical risks exist?**
Over-reliance on a probability for high-stakes decisions; privacy of student data; possible unfairness because marks, extracurriculars and training access may reflect unequal opportunity, with no subgroup analysis possible (no sensitive attributes in the data); and learned patterns that may reproduce historical or constructed rules rather than real recruiter behaviour. The model should support, not replace, human judgment (§26).

**57. What would you do with a larger real-world dataset?**
Repeat the audit (duplicates, conflicts, leakage); split by time or by institution to test transfer; compare the same model families with nested resampling; check calibration and recalibrate only under a predefined criterion; define costs with stakeholders before choosing a threshold; run subgroup and fairness audits; and keep the test set sealed until the protocol is frozen.

**58. How would you validate generalization?**
External validation on other campuses and years (temporal and site-based splits), repeated or nested cross-validation with more folds, subgroup performance checks, and calibration monitoring after any deployment. A single random split on one dataset, as here, cannot show transfer to a different population.

**59. What is the single biggest methodological lesson from this project?**
Design the evaluation before trusting any score. Auditing duplicates and conflicts revealed that the supplied split leaked feature profiles; fixing that, fixing the protocol before opening the test set, judging models on proper probability metrics, and reporting caveats (including the test-access history) honestly mattered more than the choice of algorithm — three very different algorithms ended in a practical tie.


---

## P. Training configuration and convergence

These answers rest on the saved final model, the saved configuration files and the executed notebooks; see Appendix C of the report. The reproduction check mentioned below was a read-only audit computation, not part of the original training.

**60. How many epochs did you train for?**
None; an epoch count does not apply. An epoch is one complete pass through the training examples in mini-batch training, as in a neural network. The final Logistic Regression is fitted by the L-BFGS optimiser, which is measured in optimiser iterations, and the forest and boosting models are built tree by tree.

**61. What solver and settings did the final Logistic Regression use?**
`LR_B` (`models/phase7/LR_B_final.joblib`): solver `lbfgs`, L2 penalty, `C = 1`, `max_iter = 5000`, no class weighting, 15 input columns, fitted on 8,425 rows (train + validation). Its fitted `n_iter_` is `[30]`; it has one entry because the problem is binary and a single logistic function is fitted.

**62. Did the model converge? How do you know?**
Yes. It stopped after 30 optimiser iterations against a configured limit of 5,000, so the limit was never binding. L-BFGS stops when its tolerance is met, and scikit-learn raises a warning when it stops at the limit instead. The executed notebooks contain no such warning, and notebook 05 asserts `n_iter_[0] < 5000` after each logistic fit. A read-only reproduction in the audit gave the same coefficients exactly and 30 iterations again.

**63. Why not retrain with more iterations to be safe?**
Because there is nothing to gain. The L2-regularised logistic loss is convex, and the optimiser had already stopped on tolerance, so more allowed iterations would return the same solution. Retraining only to obtain a larger number would also put the frozen artifact and its recorded hashes at risk for no scientific benefit.

**64. Do you have the training logs, such as the loss at each iteration?**
No. Per-iteration losses were not logged. What was recorded is the final `n_iter_` of the saved models (30 for the final `LR_B`; 11, 30, 11 and 31 for the four Phase 4 logistic models), the assertion and the absence of warnings. I do not have iteration counts for the individual grid-search, cross-validation or inner fits, and I do not quote any.

**65. What is the difference between an iteration, a fit, a fold and a tree?**
An optimiser iteration is one L-BFGS update inside a single fit. A fit is one complete pipeline training call (scaler, encoders and model). A fold is one partition of cross-validation, so five-fold cross-validation of one model means five fits. A tree is one decision tree: Random Forest B has 300 independent trees, and Gradient Boosting B has 300 sequential boosting stages, each adding one tree. These are different quantities and cannot be added up or used interchangeably.

**66. How many models did you train in total?**
About 237 pipeline fits, counted from the loops in the build scripts of the saved notebooks (107 baseline and grid fits, 15 cross-validation, 75 inner calibration, 3 validation-confirmation, 31 ablation and 6 final-stage fits), plus 36 small calibrator fits. That is an estimate derived from code, not an exact training log, and it excludes dry runs whose outputs were not saved. Only one fit is the deployed model: `LR_B` refitted on 8,425 rows.

**67. What was each part of the data used for?**
Train (6,938) fitted the model parameters. Validation (1,487) compared models and configurations and selected the operating threshold under the predefined F1 rule. Five-fold stratified cross-validation on the training set assessed stability and calibration of the already-chosen configurations; it did not select the final model. The test set (1,487) gave one final held-out evaluation after the specification was frozen. The final model was then refitted on train + validation (8,425 rows) without changing its specification.

**68. Was the test set untouched until the end?**
No, and the report does not claim that. Phases 1–3 used the eventual test rows for descriptive analysis only, and two orientation shell commands touched the file before the final evaluation (§8). The test data were not used for model development, selection, tuning, calibration or threshold selection, and the final evaluation was done once, after the protocol was frozen.

**69. Is Logistic Regression the best model?**
Not demonstrably. On the test set Logistic Regression, Gradient Boosting and Random Forest were practically tied on ROC-AUC (0.8857, 0.8858 and 0.8859; CI for Logistic Regression 0.868–0.902). I retained Logistic Regression for simplicity, interpretability, better calibration (ECE 0.016 against 0.031 and 0.024) and a smaller train–validation gap. That is a choice for this dataset and protocol, not a claim of superiority.

**70. Why does threshold 0.35 give higher recall but lower precision, and is it better than 0.50?**
A lower threshold labels more students Placed. That finds more of the truly placed students (recall 0.851 against 0.765; false negatives 92 against 145) but also labels more NotPlaced students as Placed (false positives 203 against 134), so precision falls (0.722 against 0.779). Accuracy is 0.802 at 0.35 (1,192 of 1,487 correct) and 0.812 at 0.50 (1,208 correct). Threshold 0.35 was selected by the predefined validation-F1 rule; the interval for its F1 advantage over 0.50 includes zero, so it is an operating choice, not a proven better or universally optimal threshold.

**71. Does the threshold change the ROC-AUC?**
No. ROC-AUC (0.8857) is computed from the probabilities over all possible thresholds, so it does not depend on 0.35 or 0.50. The threshold changes only the confusion matrix and the metrics derived from it: accuracy, precision, recall and F1.

**72. Is the dataset real or synthetic?**
I do not know. The dataset card does not describe how the data were collected, and I make no claim either way. The project records its source as Hugging Face `Krooz/Campus_Recruitment_CSV` with licence CC0-1.0, as documented in the project; I did not re-check the licence online.
