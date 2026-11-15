# 16. Limitations, Risks and Future Improvements

| Area | The issue | What a careful answer says |
|---|---|---|
| **Provenance** | Origin of the dataset is undocumented; structure looks rule-generated (hard limits, huge effects, step patterns) | "I cannot claim it is real institutional data. I treat results as a demonstration of a sound pipeline." <span class="tag u">UNKNOWN</span> |
| **Representativeness / generalisation** | One dataset, one split, one college-like population; narrow ranges | strong scores here do not establish performance elsewhere; no external validation |
| **Dataset bias** | Marks, training access and extracurriculars can reflect unequal opportunity; labels may encode past hiring preferences | the model may reproduce such patterns |
| **Correlation vs causation** | Observational data | coefficients and bars are associations; changing an input in the app is not advice |
| **Calibration** | Test slope 1.05, ECE 0.016, but a small under-prediction in the top decile (predicted 0.905, observed 0.946) | probabilities are good on this data; not recalibrated because no method met the materiality rule |
| **Supported input ranges** | CGPA 6.5–9.1, aptitude 60–90, soft skills 3.0–4.8, SSC 55–90, HSC 57–88; Projects = 0 has 22 rows | out-of-range inputs are refused; rare levels warn |
| **Threshold choice** | Costs of FP and FN were never defined; 0.35 not conclusively better | choose the threshold from the real costs |
| **Test-set caveat** | Early EDA looked descriptively at test rows | use the qualified wording (Chapter 5) |
| **Explainability limits** | Contributions are baseline-dependent associations | not SHAP, not causal |
| **Production** | Local app only; no hosting, authentication, monitoring, drift detection; no evidence of any real use | a demonstration, not a service |
| **Privacy and fairness** | Entries stay in session memory and are never logged; no sensitive attributes in the data, so no fairness audit was possible | real use needs consent, governance and subgroup audits |

## What would be needed before real placement decisions

1. Documented, **consented data from the institution**, with a proper data dictionary.
2. **External validation** on a later cohort or a second college; re-check calibration.
3. A **defined cost** for false positives and false negatives, and a threshold derived from it.
4. A **fairness audit** across relevant groups, and a review of whether features such as SSC/HSC marks proxy for disadvantage.
5. **Human-in-the-loop use**: a decision *aid*, never an automatic decision; an appeal route.
6. Operational controls: hosting, access control, monitoring for drift, retraining policy.

**Future improvements** (none done): widen the supported ranges with new data and a full refit; repeated or nested cross-validation; interaction features (for example Extracurricular × Training); an uncertainty display per prediction; subgroup analysis.

<div class="wrap" markdown="1">
**Key takeaways.** The result shows the *pipeline* is sound, not that the model works on real students. Unknown provenance, no external validation, undefined costs, no fairness audit.
**Revision questions.** (1) Why is the app limited to CGPA 6.5–9.1? (2) Why might SSC marks be unfair as a predictor? (3) Name three requirements before real use. (4) Why is a probability not a guarantee? (5) What is not claimed about 0.35?
**Professor may ask.** "What is the weakest part?" Dataset provenance and representativeness. "Would you deploy it?" Not on this evidence.
</div>

# 17. Presentation and Demonstration

Say these in your own words; they are deliberately plain.

## 17.1 The pitches

**30 seconds.** "I built a Streamlit app that takes ten details about a student, like CGPA, projects and aptitude score, and gives the estimated probability of being placed. It uses logistic regression and reached a test ROC-AUC of 0.886. I also explain which inputs pushed the number up or down, and I am careful about what the result does and does not mean."

**1 minute.** Add: "The data is a public dataset of 10,000 rows. I removed 72 duplicates and 16 contradictory rows, leaving 9,912, and split them 70/15/15. I compared logistic regression, a decision tree, a random forest and gradient boosting. Three of them performed almost the same, so I kept logistic regression because I can explain every number in it. The app shows the probability and, side by side, the label at two thresholds."

**3 minutes.** Problem → data and cleaning → preprocessing (10 columns to 15) → model and why logistic → results (AUC 0.886, calibrated; accuracy 81% at 0.50) → two thresholds with the 37.3% example → limits. Finish with "the dataset's origin is undocumented, so this is a demonstration of a sound method."

**5 minutes.** Use the eight slides below, about 40 seconds each.

## 17.2 Slide-by-slide (5–6 minutes)

| # | Slide | Show | Say |
|---|---|---|---|
| 1 | Problem | "10 inputs → probability" | "Predictive analytics: learn from past records, estimate a new student's chance." |
| 2 | Data | cleaning flow figure | "Hugging Face, CC0. 10,000 → 9,912 after removing 72 duplicates and 16 conflicting rows. Origin undocumented." |
| 3 | Preparation | pipeline figure, split bar | "Standardise 5 scores, one-hot 5 columns: 10 become 15. Stratified 6,938 / 1,487 / 1,487, seed 42." |
| 4 | Models | validation table | "Four families; logistic, forest and boosting tied, so I kept the simplest." |
| 5 | Logistic regression | sigmoid figure, formula | "Weighted sum, sigmoid, probability. L2 penalty, C = 1, fitted in 30 iterations." |
| 6 | Results | metric table | "Test AUC 0.886, log loss 0.420, well calibrated. AUC is not accuracy: accuracy is 81%." |
| 7 | Thresholds and demo | threshold figure, then the app | "One probability, two rules." |
| 8 | Limits and close | limitations list | "Associations, not causes; real data needed before real use." |

## 17.3 Live demonstration script

1. **Before class:** run `run_app.bat`; open <http://localhost:8501>. The first load can take a few seconds: wait, do not refresh repeatedly.
2. **Point out the sections:** *Student profile* (three groups: Academic, Technical, Employability; ten required fields, each with its supported range); the three buttons (**Predict**, **Load typical profile**, **Clear profile**); *Result* (percentage, labels at both thresholds, band); *Key factors* (blue raises, orange lowers); *Show the full calculation*; *Inputs used*; *Model information* (model, data, held-out results); *About the thresholds and the limits of this tool*.
3. Press **Load typical profile**, then **Predict**: **37.3%**. "Placed at 0.35, Not Placed at 0.50: the same probability and two rules."
4. Explain the bars: "Placement training and extracurriculars push up; having 2 projects and 1 workshop, compared with the reference levels, pushes down. These are associations."
5. Change **CGPA to 8.45** → **43.9%**, and show "Inputs used" holds 8.45.
6. Type **CGPA 9.8** → red message "Out of range: enter a value from 6.5 to 9.1" and no result. "The model has never seen that range, and a made-up number would look more reliable than it is."
7. If asked about credibility: "You are right to ask. The source is public and undocumented; I treat the result as a demonstration of method."

**How to introduce the dataset:** "public CC0 dataset on Hugging Face, ten columns, 10,000 rows, origin undocumented". **The model:** "a weighted sum turned into a probability by the sigmoid". **The results:** "AUC 0.886, calibrated, accuracy 81% at 0.50".

**Close.** "For this dataset and protocol, logistic regression gave the best balance of accuracy, calibration and explainability. It shows the pipeline is sound; it does not show that the model works on a different college, so I would validate it on real institutional data before anyone relied on it."

**Avoid / say instead.**

| Avoid | Say |
|---|---|
| "88.6% accurate" | "AUC 0.886; accuracy 81.2% at 0.50" |
| "The test set was untouched" | "Not used for development, selection, tuning, calibration or threshold choice" |
| "0.35 is the best threshold" | "Selected by a predeclared validation-F1 rule; 0.50 is practically tied" |
| "CGPA increases your chances" | "Higher CGPA is associated with higher placement in this dataset" |
| "It trained for N epochs" | "Fitted by L-BFGS, converged in 30 iterations" |
| "Real student data / deployed" | "Public dataset of unknown origin; local demo app" |

# 18. The Viva Question Bank

*110 questions. ★ = basic, ★★ = intermediate, ★★★ = hard. Answers are short on purpose; the chapter numbers tell you where to read more.*

## A. Basic predictive analytics ★

**Q1. What is predictive analytics?** Using historical data, statistics and machine learning to estimate a future or unknown outcome ("learning from the past to predict the future", course Unit 1). Ours estimates P(placed). *(Ch. 1)*
**Q2. Descriptive vs predictive vs prescriptive?** What happened / what will happen / what should we do. We do the first in EDA, the second in the model; never the third.
**Q3. What is supervised learning?** Learning from examples that carry the correct label. Each training student has a known Placed/NotPlaced.
**Q4. Classification vs regression?** Category versus number. Ours is classification (binary), even though the algorithm is called Logistic *Regression*.
**Q5. Feature and target?** Feature = input (CGPA…); target = what we predict (PlacementStatus).
**Q6. Why is it binary classification?** Two outcomes: Placed (1) and NotPlaced (0).

## B. Dataset ★

**Q7. Source and licence?** Hugging Face `Krooz/Campus_Recruitment_CSV`, CC0-1.0 (public-domain dedication). *(Ch. 2)*
**Q8. Size and structure?** 10,000 rows × 11 columns originally, split in three files (7,225/1,275/1,500); 9,912 rows after cleaning.
**Q9. Is it real student data?** Unknown. The structure looks rule-generated, so I do not claim it is real.
**Q10. Which columns are categorical?** ExtracurricularActivities and PlacementTraining (Yes/No) plus the target; the three counts are numeric but encoded as categories.
**Q11. Why are Internships/Projects/Workshops encoded as categories although they are numbers?** EDA showed non-monotonic placement rates (workshops 28%, 21%, 73%, 56%), so a straight-line effect would be wrong.
**Q12. What does the dataset not contain?** Sensitive attributes, a data dictionary, provenance.

## C. EDA ★★

**Q13. What is EDA and why first?** Understanding data before modelling; it exposed duplicates, non-monotonic counts and class balance. *(Ch. 3)*
**Q14. Which plots did you use?** Histograms with KDE, box plot, count plot, violin, bar charts with error bars, scatter, ECDF, heat-maps; no pie chart.
**Q15. What does the correlation heat-map tell you, and what can't it?** Which features move together (largest r = 0.56, aptitude-HSC); not cause.
**Q16. Is there class imbalance?** Moderate: 41.56% Placed. Handled by judging probability quality; class weighting was tested and not adopted.
**Q17. Did you remove outliers?** No. The IQR rule flagged none in the five scores.

## D. Data cleaning ★★

**Q18. What cleaning did you do?** Removed 72 surplus exact duplicates and excluded 16 rows (8 groups) with conflicting labels: 10,000 → 9,912. *(Ch. 4)*
**Q19. Exact vs conflicting duplicates?** Same features and same label vs same features but different labels.
**Q20. Why not a majority vote for conflicts?** Each group has one of each label (a tie); guessing would invent data.
**Q21. Why remove duplicates?** They over-weight profiles and let the model be scored on what it memorised.
**Q22. Any missing values or imputation?** None missing; none imputed.
**Q23. Could cleaning bias results?** Slightly: the conflicting rows are all top-end profiles; only 0.88% of rows affected.

## E. Preprocessing, one-hot encoding, standardisation ★★

**Q24. Why preprocess?** Models need numbers; and scale matters for regularisation. *(Ch. 6)*
**Q25. What is one-hot encoding?** One 0/1 column per category.
**Q26. Why drop one category?** Remaining columns determine it; it removes redundancy with the intercept and gives "relative to reference" coefficients.
**Q27. What is the dummy-variable trap?** Keeping all K columns makes them sum to 1 (perfectly redundant), making coefficients ambiguous.
**Q28. What is standardisation?** z = (x − μ)/σ, mean 0, SD 1.
**Q29. Standardisation vs normalisation?** Z-score has no fixed range; min-max maps to [0, 1]. We used standardisation.
**Q30. Fit vs transform?** Fit learns μ, σ, categories from data; transform applies them.
**Q31. 10 columns → 15: how?** 5 scaled + (2 + 3 + 3) counts + 2 binary = 15.

## F. Train / validation / test ★★

**Q32. Why split?** To measure performance on data the model has not learned from. *(Ch. 5)*
**Q33. Roles of the three sets?** Train fits; validation chooses; test scores once.
**Q34. Sizes and method?** 6,938 / 1,487 / 1,487; two stratified `train_test_split` calls, seed 42.
**Q35. What does stratify do?** Keeps the Placed share (41.56%) equal in all parts.
**Q36. Why refit on train + validation?** More data after the choices are frozen: 8,425 rows.
**Q37. Did you leak the test set?** Not into development, selection, tuning, calibration or threshold selection; early descriptive EDA looked at those rows and is disclosed.

## G. Logistic Regression ★★

**Q38. What is Logistic Regression?** A linear score turned into a probability by the sigmoid; fitted by maximum likelihood. *(Ch. 7)*
**Q39. Why "regression"?** It linearly regresses the log-odds.
**Q40. Linear vs logistic?** Linear predicts any number; logistic predicts a probability between 0 and 1.
**Q41. What do the coefficients mean?** The change in log-odds per unit (per SD for scores, or versus the reference level); eᵇ is the odds multiplier.
**Q42. What is the intercept?** The log-odds for the reference student (−1.661, i.e. P = 0.16).
**Q43. Does scaling matter for LR?** With the L2 penalty and L-BFGS, yes.
**Q44. Is LR a neural network?** No: no layers, neurons or epochs.

## H. Sigmoid and probability ★★

**Q45. Formula of sigmoid?** p = 1/(1 + e⁻ᶻ).
**Q46. Why needed?** To squash any real score into (0, 1).
**Q47. What is z = 0?** p = 0.5.
**Q48. Is the sigmoid an "activation function" in your model?** It is the logistic link; there is no neural layer. The course slide mentions activation layers because the same function is used there.

## I. Loss, regularisation, C, L-BFGS ★★★

**Q49. What is the loss function?** Binary cross-entropy (log loss) = −mean[y ln p + (1−y) ln(1−p)].
**Q50. Why that loss?** It is the negative log-likelihood of the Bernoulli model: minimising it = maximum likelihood.
**Q51. Why take logs of the likelihood?** Products of many small numbers underflow; sums are easier and have the same maximiser.
**Q52. What is regularisation?** A penalty on large weights to reduce over-fitting.
**Q53. L1 vs L2?** L1 (sum of |w|) can zero weights; L2 (sum of w²) shrinks smoothly. We use L2.
**Q54. What does C do?** It is the *inverse* strength: small C = strong penalty, large C = weak. C = 1 here.
**Q55. Is the intercept penalised?** No (verified from the scikit-learn source and numerically).
**Q56. What is L-BFGS?** A quasi-Newton optimiser, the solver. It is not an activation function.

## J. Model training and loops ★★★

**Q57. What does `.fit()` do?** Learns scaler/encoder values, then L-BFGS finds 16 numbers (15 coefficients + intercept). *(Ch. 9)*
**Q58. Learned parameter vs hyper-parameter?** Learned: coefficients. Chosen by us: C, solver, depth.
**Q59. How many fits?** About 237, from counting loops in the build scripts; an estimate, not a log.
**Q60. How many epochs?** None. The final model took 30 optimiser iterations (cap 5,000).
**Q61. What was cross-validation used for?** Stability and calibration checks on train only (5 folds), not model choice.
**Q62. How was the model chosen?** Lowest validation log loss within the statistically indistinguishable set, tie broken by interpretability.

## K. Decision Tree, Random Forest, Gradient Boosting ★★

**Q63. How does a decision tree split?** It picks the question that most reduces Gini impurity (1 − Σp²).
**Q64. Why do deep trees overfit?** They can memorise single students: our unlimited tree had train AUC 1.0, validation log loss 11+.
**Q65. What is a random forest?** Many trees on bootstrap samples with random feature subsets; probabilities averaged.
**Q66. What is gradient boosting?** Small trees added one after another, each fitted to the previous errors; scaled by the learning rate.
**Q67. Bagging vs boosting?** Parallel and averaged (variance) vs sequential and corrective (bias).
**Q68. What were your RF and GB settings?** RF: 300 trees, depth 10, leaf 10, sqrt features. GB: 300 stages, rate 0.05, depth 2.
**Q69. Why not use the forest?** It did not beat LR; LR is simpler and better calibrated here.

## L. Model evaluation ★★

**Q70. Confusion matrix?** TP/TN/FP/FN counts; at 0.50: 473/735/134/145. *(Ch. 10)*
**Q71. Accuracy, precision, recall formulas?** (TP+TN)/n; TP/(TP+FP); TP/(TP+FN).
**Q72. What is F1?** Harmonic mean of precision and recall.
**Q73. What is ROC-AUC?** Probability a random Placed student outranks a random Not Placed one; 0.8857.
**Q74. Why is AUC not accuracy?** AUC ranks over all thresholds; accuracy is one threshold's correct share (0.812).
**Q75. Log loss and Brier?** Proper scores for probabilities: 0.420 and 0.134; lower is better.
**Q76. What is calibration slope 1.05?** Very close to ideal 1: probabilities are neither over- nor under-confident.
**Q77. What does the CI 0.868–0.902 mean?** The range of AUC across 2,000 resamples of test students; it is not about one student.

## M. Thresholds ★★★

**Q78. Why two thresholds?** One probability, two rules: 0.50 is conventional; 0.35 is our validation-F1-selected operating threshold. *(Ch. 11)*
**Q79. How was 0.35 chosen?** Highest validation F1 on a 0.20–0.80 grid, ties toward 0.50; fixed in advance.
**Q80. Is 0.35 the best?** No: F1 is flat from 0.35 to 0.45 and the validation gain over 0.50 (+0.010) has a CI including zero.
**Q81. What does lowering the threshold do?** Raises recall (0.765 → 0.851), lowers precision (0.779 → 0.722).
**Q82. What happens at p = 0.373?** Placed at 0.35, Not Placed at 0.50.
**Q83. Does the probability change with the threshold?** Never.

## N. Explainability ★★

**Q84. How does the app explain predictions?** Coefficient × processed value, summed per feature; contributions plus intercept equal the log-odds. *(Ch. 12)*
**Q85. Is SHAP used?** No.
**Q86. What does a bar mean?** Contribution to the log-odds relative to a baseline student.
**Q87. Does it prove causation?** No: associations from observational data.

## O. Python libraries and Streamlit ★

**Q88. Why pandas?** Tables: reading CSVs, cleaning, one-row inputs. *(Ch. 13)*
**Q89. Why scikit-learn Pipeline?** One object that preprocesses and predicts identically in training and app.
**Q90. What is joblib?** Saves/loads the fitted pipeline.
**Q91. How does Streamlit work?** The script re-runs on each interaction; widgets return values; `session_state` keeps state.
**Q92. What is `st.cache_resource` for?** Load the model once.
**Q93. Which libraries are only in notebooks?** Matplotlib, Seaborn, statsmodels, Jupyter.

## P. Application architecture ★★

**Q94. Walk me through a prediction.** Validate → DataFrame → scale/encode → z → sigmoid → two thresholds → contributions → render. *(Ch. 14)*
**Q95. Why a frozen model?** Reproducible, instant, and exactly the evaluated model.
**Q96. What happens if CGPA = 9.8?** Rejected with a range message; nothing is clipped.
**Q97. How does the app detect a wrong model file?** SHA-256 and scikit-learn version checks.

## Q. Limitations ★★

**Q98. Biggest weakness?** Unknown data provenance and no external validation. *(Ch. 16)*
**Q99. Can real placement offices use it?** Not on this evidence; it would need real consented data, validation, defined costs and a fairness audit.
**Q100. Is 37.3% the student's real chance?** No: the estimate for similar students in this dataset, if calibrated.
**Q101. What would you do with more time?** Define costs, validate on a second dataset, subgroup audit.

## R. Difficult follow-ups ★★★

**Q102. If L-BFGS needed 30 iterations, why max_iter = 5000?** A safety cap; hitting it would signal non-convergence. The build code asserts `n_iter_ < 5000`.
**Q103. Why not use accuracy as the training loss?** It is not smooth (no gradient); log loss is smooth, convex and rewards honest probabilities.
**Q104. Is fitting the scaler on all 8,425 rows leakage?** No: the test rows are excluded; the earlier development pipelines were fitted on train only.
**Q105. Why is the penalty only 0.04% of the objective, and does C matter?** With 8,425 rows the data term dominates; AUC barely changes from C = 0.1 to 100.
**Q106. Two correlated features: what happens?** Weight is shared between them and individual coefficients are less interpretable; VIF here is below 2.
**Q107. Validation was used to choose: is the test number optimistic?** The test set was not used to choose; validation figures are the optimistic ones.
**Q108. Sigmoid vs softmax?** Sigmoid for two classes; softmax normalises K scores for K classes (course Multinomial LR).
**Q109. What does the odds ratio 2.65 for training mean?** Odds of Placed are 2.65 times higher for trained students, others fixed. Association only.
**Q110. How would you test the model on another college's data?** Run its `predict_proba` on that data without refitting, compare AUC/calibration, expect calibration shift, and refit only with a new protocol.
