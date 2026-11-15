# 8. The Other Models We Evaluated

*The course slides do not cover trees or ensembles; this chapter is standard theory* <span class="tag s">SUPPLEMENT</span> *tied to our saved models.* Four model families were compared in notebook 05: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, plus a "dummy" reference that always predicts the training placement rate (0.584 NotPlaced / 0.416 Placed).

**Two input versions.** *Variant A* treats the three counts as ordinary numbers; *Variant B* one-hot encodes them (because EDA showed non-monotonic effects). **Logistic Regression uses the 15 processed columns** described in Chapter 6. **The three tree models use 18 columns** (verified by loading `DT_B`, `RF_B`, `GB_B`): all levels of the counts are kept (3 + 4 + 4 = 11 columns), the five scores are *not* scaled (trees only compare values with thresholds, so scale is irrelevant), plus the two Yes/No columns: 5 + 11 + 2 = 18.

## 8.1 Decision Tree

* **What.** A flowchart of yes/no questions. **Node** = a question ("Is aptitude ≤ 78?"); **branch** = an answer; **leaf** = an end box that stores the class mix (and so a probability) of the training rows that reached it; the first node is the **root**.
* **How it splits.** At each node it tries many questions and keeps the one that makes the two child groups *purer*. **Gini impurity** of a node with class proportions pₖ: G = 1 − Σ pₖ². **Entropy**: H = −Σ pₖ log₂ pₖ. **Information gain** = parent impurity − weighted average of the children's impurities. (scikit-learn's default, and our setting, is Gini.)
* **Worked example.** A node holds 10 students: 6 Placed, 4 Not. G = 1 − (0.6² + 0.4²) = **0.48**. A split sends 6 to the left (5 Placed, 1 Not) and 4 to the right (1 Placed, 3 Not): G_left = 1 − (0.833² + 0.167²) = 0.278, G_right = 1 − (0.25² + 0.75²) = 0.375; weighted = 0.6 × 0.278 + 0.4 × 0.375 = **0.317**; gain = 0.48 − 0.317 = **0.163**. With entropy: parent 0.971, children 0.650 and 0.811, weighted 0.715, information gain **0.257**.
* **Depth and overfitting.** Deeper tree = more questions = can memorise individual students. An unlimited tree reached training AUC 1.000 and validation log loss 11+ (Chapter 5).
* **Our DT_B.** `criterion='gini'`, `max_depth=5`, `min_samples_leaf=5`, 32 leaves (verified). Validation AUC 0.846.

## 8.2 Random Forest

* **Ensemble learning** = many models vote; errors of individual models partly cancel. **Bootstrap sampling** = draw *n* rows *with replacement* from the training set (about 63% of distinct rows appear in each sample). A random forest trains many trees, each on its own bootstrap sample, and at every split lets the tree consider only a random subset of the columns (`max_features='sqrt'`: √18 ≈ 4 columns). The forest **averages the trees' probabilities**.
* **Why it helps.** One deep tree is unstable (small data change → very different tree). Averaging many de-correlated trees reduces this variance, so predictions are more stable. This is **bagging** (bootstrap aggregating).
* **Our RF_B.** 300 trees, `max_depth=10`, `min_samples_leaf=10`, `max_features='sqrt'`, `bootstrap=True`, `random_state=42` (verified). Validation AUC 0.869.

## 8.3 Gradient Boosting

* **Sequential learning with weak learners.** A *weak learner* is a very small tree (ours: depth 2). Boosting starts from a simple guess (the class base rate), computes the **errors/residuals** (for log loss, essentially *y − p* for every student), fits a small tree to those errors, and **adds** the tree's output, scaled by the **learning rate**, to the running score. Repeat for `n_estimators` rounds.
* **Numbers:** a student who is Placed (y = 1) with current p = 0.60 has residual 0.40; the next tree learns to push such students up; the score becomes old score + 0.05 × tree output. A **small learning rate** (0.05) takes careful steps and needs more trees (300).
* **Bagging vs boosting.** Bagging: independent trees in parallel, *average* them, reduces variance. Boosting: dependent trees in sequence, each *fixes the previous ones*, reduces bias but can overfit if too many or too deep.

![Random forest versus gradient boosting](masterbook_assets/img/f_ensembles.png)

* **Our GB_B.** `GradientBoostingClassifier`, `n_estimators=300`, `learning_rate=0.05`, `max_depth=2`, `min_samples_leaf=1`, `loss='log_loss'` (verified; 300 stages fitted). Validation AUC 0.871.

## 8.4 What we compared, and why Logistic Regression was kept <span class="tag m">REPORTED</span>

Hyper-parameters of each tree family were searched on the **validation** set with a manual loop (`itertools.product`, not `GridSearchCV`; random search was not used): DT 12 settings, RF 18, GB 16 per variant. The best setting by validation log loss was kept.

| Model (Variant B) | Validation AUC | Validation log loss | **Test AUC** | Test log loss | Test ECE | Test cal. slope |
|---|---|---|---|---|---|---|
| Logistic Regression (L2, C=1) | 0.8717 | 0.4408 | **0.8857** | 0.4196 | **0.016** | 1.05 |
| Gradient Boosting | 0.8706 | 0.4413 | 0.8858 | 0.4186 | 0.031 | 1.06 |
| Random Forest | 0.8693 | 0.4432 | 0.8859 | 0.4204 | 0.024 | 1.15 |
| Decision Tree | 0.8464 | 0.4799 | not evaluated on test | – | – | – |
| Dummy (always the base rate) | 0.5000 | 0.6788 | – | – | – | – |

**Conclusion, stated carefully.** LR, GB and RF are **statistically indistinguishable** on validation, five-fold CV (AUC 0.879 / 0.876 / 0.876) and test (differences of 0.0001–0.0002 AUC, paired intervals include zero). Logistic Regression was retained, **not because it won**, but because (i) the selection rule was fixed in advance: lowest validation log loss within the indistinguishable set, ties broken by interpretability (`05_provisional_selection.json`); (ii) it has 15 readable coefficients; (iii) its test calibration (ECE 0.016) was the best of the three. The Decision Tree was clearly worse and was dropped before the final stage.

<div class="wrap" markdown="1">
**Key takeaways.** Trees ask questions (Gini/entropy); forests average many independent trees; boosting adds small trees sequentially. LR, RF, GB tied; LR kept for simplicity and calibration. Trees used 18 columns; LR used 15.
**Formulas.** G = 1 − Σpₖ²; H = −Σpₖ log₂pₖ; gain = parent − weighted children.
**Revision questions.** (1) Gini of a 50/50 node? (0.5) (2) What does bootstrap mean? (3) Bagging vs boosting? (4) Why did the tree models not need scaling? (5) Why was LR kept even though it did not win?
**Professor may ask.** "Why not the most complex model?" It did not perform better. "Was LR uniquely best?" No: the saved results do not support that claim.
</div>

# 9. Model Training: What Actually Happened

## 9.1 What `.fit()` does <span class="tag v">VERIFIED</span>

```python
pipe = make_pipeline("LR", "B", C=1.0, class_weight=None)   # build the Pipeline (Chapter 6)
pipe.fit(X_train, y_train)                                  # X_train: 6,938 × 10 DataFrame, y_train: 0/1
p_val = pipe.predict_proba(X_val)[:, 1]                     # probabilities for the 1,487 validation rows
```

`fit` runs the pipeline's steps in order:

1. **`prep.fit_transform(X_train)`**: the scaler learns μ and σ for 5 columns; the encoders learn their category lists; the 10 raw columns become a 6,938 × 15 numeric matrix.
2. **`model.fit(matrix, y_train)`**: L-BFGS iteratively changes 16 numbers (15 coefficients + the intercept) to minimise the penalised log loss of Chapter 7, starting from zeros, until the gradient is tiny.
3. The fitted pipeline now holds **learned parameters**: scaler means and σ's, encoder categories, 15 coefficients, 1 intercept. Our **hyper-parameters** (C, solver, max_iter, tree depth, …) were *chosen by us* and are not learned by the fit.

*How the model learns from labelled examples:* every iteration compares its current probabilities with the true 0/1 labels, measures the cross-entropy, and moves the 16 numbers in the direction that reduces it. There is no hidden layer and no epoch loop.

## 9.2 The experiments, in order

| Notebook | Phase | Experiment | Data used for fitting → scoring |
|---|---|---|---|
| 05 | 4 | LR (4 versions), dummy, C-sensitivity, DT/RF/GB grids; bootstrap CIs (1,000 resamples) | train → validation |
| 06 | 5 | 5-fold CV of LR_B, GB_B, RF_B; nested recalibration test (sigmoid and isotonic); validation confirmation | train folds → held-out folds; train → validation |
| 07 | 6 | Threshold analysis 0.20–0.80; explainability; feature-group ablation; error analysis | OOF and validation predictions; ablation fits |
| 08 | 7 | Frozen protocol; refit on train + validation; **one** test evaluation (2,000 bootstrap resamples) | train+validation → test |

**Calibration** here means *checking and possibly correcting the probabilities*. Sigmoid (Platt) and isotonic recalibration were fitted in a leakage-safe nested way, but **no method met the pre-declared materiality rule** (`06_calibration_method_decision.csv`: `material=False` for all six rows), so the app uses the raw model probabilities.

## 9.3 How many loops, how many fits? <span class="tag e">ESTIMATE</span>

Every number below was obtained by reading the loop structure of `src/build_notebook_05.py` … `08.py` (the scripts that generate the notebooks) and multiplying. Three different kinds of knowledge are mixed; the last column says which:

| Operation | Loop structure in the code | Fits | Kind |
|---|---|---|---|
| Decision-tree grid | 12 settings × 2 variants | 24 | **recorded** (rows in `05_tuning_grids.csv`) |
| Random-forest grid | 18 × 2 | 36 | recorded (same file) |
| Gradient-boosting grid | 16 × 2 | 32 | recorded (same file); the 92 rows record `fit_seconds`, summing to 94.3 s |
| LR baselines | A/B × plain/balanced | 4 | counted from code |
| LR C-sensitivity | 2 variants × 5 values of C | 10 | counted from code |
| Dummy baseline | 1 | 1 | counted |
| 5-fold CV | 3 models × 5 folds | 15 | **recorded** (`06_cv_fold_results.csv`: 15 rows) |
| Nested calibration | 5 outer × 3 models × 5 inner | 75 | counted from code |
| Validation confirmation | 3 models | 3 | counted |
| Feature-group ablation | 1 full + 5 groups × 5 folds + 5 validation fits | 31 | 25 fold fits recorded; rest counted |
| Phase 7 | 3 reproducibility refits on train + 3 final fits on train + validation | 6 | counted |
| **Total** | 107 + 15 + 75 + 3 + 31 + 6 | **237** | **ESTIMATE** |

Plus about 36 tiny calibrator fits (a two-parameter sigmoid and an isotonic map; 15 + 15 on the folds, 3 + 3 on validation) that are not full pipeline fits.

<div class="note" markdown="1">
**How to describe it honestly.** "About 237 pipeline fits, reconstructed by counting the loops in the notebook-building scripts. That is the number of fits in **one complete execution** of the final scripts; it is not an execution log. How many times the notebooks were re-run while developing them is <span class="tag u">UNKNOWN</span>." The count **excludes** statsmodels fits used for statistical inference, preprocessor-only demonstrations, and permutation importance (it re-scores, never refits). Only 132 of the 237 fits (92 grid + 15 CV + 25 ablation fold fits) have saved per-fit result rows; per-fit iteration counts were not saved.
</div>

**Do not confuse three things:** (i) an *optimiser iteration* (one L-BFGS step; our final model: 30); (ii) a *fit* (one whole `.fit()` call, containing many iterations); (iii) an *epoch* (a pass over the data in neural-network training: none in this project). The final, saved model was fitted **once**, on 8,425 rows, took 30 iterations, and was stored with `joblib.dump` as `models/phase7/LR_B_final.joblib`. A separate read-only refit on train + validation reproduced the saved coefficients exactly (recorded in `final_report.md`, Appendix C.2).

<div class="wrap" markdown="1">
**Key takeaways.** `.fit()` = fit preprocessing, then optimise 16 numbers. Models were compared on validation; CV and calibration were stability checks; the final LR was refit once on 8,425 rows. ≈237 fits is an estimate by loop counting; 30 is the exact iteration count of the final model; no epochs.
**Formulas.** 92 + 4 + 10 + 1 + 15 + 75 + 3 + 31 + 6 = 237.
**Revision questions.** (1) Which parameters are learned and which chosen? (2) What are the 16 numbers L-BFGS adjusts? (3) Why is 237 an estimate? (4) Difference between a fit and an iteration? (5) Why was calibration not applied?
**Professor may ask.** "How many epochs?" None. "How long did it take?" Per-fit seconds exist only for the grid (94 s in total); overall runtime was not logged (UNKNOWN).
</div>

# 10. Model Evaluation: the Complete Masterclass

*The course syllabus lists these metrics (F1, AUC-ROC, cross-validation) under Model Validation, but no slides on them were provided, so this chapter is standard theory* <span class="tag s">SUPPLEMENT</span>*, using our real numbers.*

## 10.1 The confusion matrix and the threshold-dependent metrics

At a threshold, every student falls in one of four cells. **Positive = Placed.**

![Confusion-matrix layout and a toy ROC curve](masterbook_assets/img/f_roc_confusion.png)

| Cell | Meaning | Example |
|---|---|---|
| **TP** true positive | predicted Placed, actually Placed | correct good news |
| **TN** true negative | predicted Not Placed, actually Not Placed | correct |
| **FP** false positive | predicted Placed, actually Not | false alarm / over-optimistic |
| **FN** false negative | predicted Not Placed, actually Placed | missed a student who was placed |

Using the test counts at threshold 0.50 (TP 473, TN 735, FP 134, FN 145; total 1,487; actually Placed = 473 + 145 = 618, actually Not = 735 + 134 = 869):

| Metric | Formula | Meaning | Calculation | When it matters |
|---|---|---|---|---|
| **Accuracy** | (TP + TN) / n | share of correct calls | (473 + 735)/1,487 = **0.812** | balanced classes; can mislead on imbalance |
| **Precision** | TP / (TP + FP) | of those called Placed, how many truly were | 473/607 = **0.779** | false alarms are costly |
| **Recall** (sensitivity, TPR) | TP / (TP + FN) | of the truly Placed, how many we found | 473/618 = **0.765** | misses are costly |
| **Specificity** (TNR) | TN / (TN + FP) | of the truly Not Placed, how many we cleared | 735/869 = **0.846** | |
| **F1** | 2PR / (P + R) | harmonic mean of precision and recall | 2(0.779)(0.765)/(0.779 + 0.765) = **0.772** | one summary of P and R |

**Why the harmonic mean?** It is dragged down by whichever of P or R is lower, so you cannot hide a bad recall behind a good precision. **Why accuracy alone is not enough:** answering "Not Placed" for everybody scores 58.4% accuracy with recall 0.

## 10.2 Threshold-independent metrics: scoring the probabilities themselves

**ROC curve and AUC.** For *every* threshold from 1 down to 0 compute recall (true-positive rate) and false-positive rate FP/(FP + TN); the plot of one against the other is the **ROC curve**. The **AUC** (area under it) equals the probability that a randomly chosen Placed student receives a higher score than a randomly chosen Not Placed student. *Toy example:* Placed scores 0.9, 0.6, 0.4 and Not Placed scores 0.7, 0.3, 0.2. Of the 9 pairs, Placed wins in 7 (0.9 beats all three; 0.6 beats 0.3 and 0.2; 0.4 beats 0.3 and 0.2), so **AUC = 7/9 = 0.778**. 0.5 = coin flip, 1.0 = perfect ranking.

**Log loss** = binary cross-entropy (Chapter 7). Punishes confident mistakes; lower is better. **Brier score** = mean of (p − y)². Toy: p = [0.9, 0.2, 0.6, 0.7], y = [1, 0, 1, 0]: (0.01 + 0.04 + 0.16 + 0.49)/4 = **0.175**. Both are *proper scoring rules*: they reward honest probabilities.

**Calibration.** A model is *calibrated* if among students given about 40% roughly 40% are placed. A *reliability curve* groups predictions into bins and compares mean predicted with the observed rate. **Calibration slope**: regress the outcome on logit(p); 1.0 is ideal, below 1 means over-confident predictions, above 1 slightly under-confident. **Calibration intercept** (and "calibration-in-the-large") tests whether predictions are on average too high or low; 0 is ideal. **Expected Calibration Error (ECE)**: split into bins and average |observed rate − mean prediction|, weighted by bin size. Our ECE uses bins of equal size (quantile bins).

**Confidence intervals.** A single test number is a sample estimate. We resampled the 1,487 test students **with replacement 2,000 times**, recomputed the metric each time, and took the 2.5th and 97.5th percentiles: a 95% *bootstrap* interval. It says how much the *statistic* could change with a different sample of students; it is **not** a statement about an individual's prediction.

| | Threshold-dependent | Threshold-independent |
|---|---|---|
| Metrics | accuracy, precision, recall, specificity, F1, confusion matrix | ROC-AUC, log loss, Brier score, calibration, ECE |
| Needs | a chosen cut-off (0.50 or 0.35) | only the probabilities |

## 10.3 Our test results <span class="tag m">REPORTED</span> (`results/metrics/modeling/08_*.csv`, n = 1,487, 618 placed)

| Metric | Value | 95% CI | Meaning |
|---|---|---|---|
| **ROC-AUC** | **0.8857** | 0.868 – 0.902 | in 88.6% of Placed-vs-Not-Placed pairs, the Placed student gets the higher probability. **Not "88.57% accuracy"** |
| Log loss | 0.420 | 0.393 – 0.449 | the always-the-base-rate dummy scored 0.679 on validation |
| Brier score | 0.134 | 0.124 – 0.145 | the same dummy scored 0.243 on validation |
| Calibration slope | 1.05 | 0.95 – 1.15 | essentially 1: probabilities neither over- nor under-confident |
| Calibration intercept | +0.04 | −0.10 – +0.18 | mean predicted 0.411 vs observed 0.416 |
| ECE | 0.016 | noise level: 0.023 mean, 0.034 at 95% | gap as small as chance alone would produce |

| Threshold | Accuracy | Precision | Recall | Specificity | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.812 | 0.779 | 0.765 | 0.846 | 0.772 | 473 | 735 | 134 | 145 |
| 0.35 | 0.802 | 0.722 | 0.851 | 0.766 | 0.781 | 526 | 666 | 203 | 92 |

At 0.50, correct = 473 + 735 = 1,208 of 1,487; at 0.35, 526 + 666 = 1,192. Lowering the threshold finds 53 more genuinely placed students (FN 145 → 92) but gives 69 more false alarms (FP 134 → 203).

![Test confusion matrices at 0.50 and 0.35](masterbook_assets/img/r_conf050.png)
![](masterbook_assets/img/r_conf035.png)

**Why ROC-AUC is not accuracy.** AUC compares *rankings* over all thresholds; accuracy counts correct *labels* at one threshold. Ours: AUC 0.886 but accuracy 0.812 (0.50) or 0.802 (0.35). **Development vs test:** CV AUC 0.879, validation 0.872, test 0.886. These are consistent, not proof of "no loss on new data": the higher test value may be sampling variation plus the larger final fit.

<div class="wrap" markdown="1">
**Key takeaways.** Threshold metrics (accuracy, precision, recall, F1) depend on the cut-off; AUC, log loss, Brier and calibration judge the probabilities. Test AUC 0.8857 (0.868–0.902), log loss 0.420, Brier 0.134, slope 1.05, ECE 0.016.
**Formulas.** Acc = (TP+TN)/n; P = TP/(TP+FP); R = TP/(TP+FN); F1 = 2PR/(P+R); Brier = mean (p−y)²; AUC = P(score_placed > score_notplaced).
**Revision questions.** (1) Compute precision at 0.35 (526/729 = 0.722). (2) Why F1 not accuracy? (3) What does AUC = 0.5 mean? (4) What does slope 1.05 say? (5) What does a CI of 0.868–0.902 mean?
**Professor may ask.** "Is 88.57% your accuracy?" No: AUC is a ranking metric. "Is 37.3% a guarantee?" No: a group frequency among similar students, if calibrated.
</div>
