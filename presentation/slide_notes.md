# Speaker notes: Predicting Student Placement Probability

Approximately 10–12 minutes in total (about 30–60 seconds per slide). Notes are embedded in `final_presentation.pptx` and are identical to the text below.

## Slide 1 — Title slide

Good morning. This project is about estimating the probability that a student is placed, and about checking that this probability can be trusted. I will tell it as a methodological story: the data, a leakage problem I found and fixed, model comparison, calibration and threshold, one final held-out evaluation, and the limits. Throughout I say prediction and association, never cause: the model estimates a probability, it does not determine a placement. First, why a probability is more useful than a label.

## Slide 2 — A probability keeps what a label discards

Most placement models answer a yes-or-no question: Placed or NotPlaced. The problem is that a label hides how certain the model is. The illustration on the right is hypothetical: two students with estimated probabilities of 0.92 and 0.55 are both labelled Placed at the usual 0.50 cut-off, yet they carry very different certainty. So the goal here is to estimate the probability of being placed given ten student characteristics, written P of Y equals Placed given X. A classification is then derived from that probability with a threshold, and different users can choose different thresholds. The key message to remember: the probability is the primary output and the classification is secondary. Next, the data.

## Slide 3 — Ten characteristics in three groups

The data come from the Hugging Face dataset Krooz slash Campus Recruitment CSV, released under CC0. The original file has 10,000 rows and 11 columns: ten predictors and the target, PlacementStatus. In the original data, 4,197 students are Placed and 5,803 are NotPlaced. I group the predictors into three families. Academic: CGPA and the SSC and HSC marks. Technical: aptitude test score, internships, projects, and workshops or certifications. Employability: soft skills rating, extracurricular activities and placement training. These numbers describe the original dataset, before cleaning. The dataset card does not say how the data were collected.

## Slide 4 — The supplied split was not safe for evaluation

This is one of the strongest parts of the project. Before any modelling I audited the data. Of the 10,000 rows, 72 were surplus exact duplicates with the same label, and I collapsed them. Eight groups of rows had identical values on all ten predictors but opposite labels; I excluded all 16 of those rows and did not use a majority vote, because with one row per label there is no majority and any choice would be arbitrary. That leaves 9,912 modelling rows. More importantly, 33 feature profiles crossed the supplied train, validation and test files, so evaluating on those files would partly measure memorisation. I therefore discarded the supplied split and made a new stratified 70, 15, 15 split: 6,938 training, 1,487 validation and 1,487 test rows. The final splits have no feature-profile overlap.

## Slide 5 — Strong associations, but counts are not monotonic

After cleaning, I explored the data. The scores that separate Placed and NotPlaced students most are aptitude, with a standardised mean difference, Cohen's d, of 1.23, and HSC marks with 1.18. Among binary variables, the observed placement rate is 61.6 percent with extracurricular activities against 13.7 percent without, an odds ratio of about 10.1; for placement training the odds ratio is about 5.7. These are associations in this dataset; I make no causal claim. The chart shows the second finding: the three count variables do not rise steadily. Internships are flat from 0 to 1 and then jump at 2; workshops dip at level 1; projects level 0 rests on only 30 students. That motivated testing a categorical representation of the counts. Now, how I tested these patterns formally.

## Slide 6 — Statistics informed the design, not the evaluation

I tested these observations formally. For the scores I used Mann-Whitney and Welch tests with effect sizes. For binary variables, chi-square tests, Cramér's V and odds ratios. For the count variables, chi-square tests and pairwise comparisons, and a Cochran-Mantel-Haenszel analysis stratified by aptitude to see whether aptitude explained the patterns; it explained part of them but the dips persisted. Holm correction handled multiple testing and variance inflation factors checked redundancy. Two numbers to remember: the maximum VIF is 1.96, so multicollinearity is low, and aptitude alone has a single-feature AUC of about 0.81. The key point is at the bottom: these statistics shaped the representation and model choices, but with 10,000 students almost anything is significant, and none of them measures out-of-sample prediction or calibration. That had to be tested with models.

## Slide 7 — From raw data to a frozen model: test comes last

This slide is the whole methodology in one picture. The first three steps, in navy, are data work: the quality audit, the leakage-aware split, then preprocessing. The teal steps are development, done with training and validation data only: baseline models, validation comparison, five-fold cross-validation, calibration, threshold analysis, and explainability with ablation. Only then was the final model frozen, in amber, with its specification, thresholds and metrics written to a protocol file. The one-time test evaluation happens after that, and the interactive application comes last. The test set was evaluated once, after the protocol was frozen. I will say honestly in a moment what that does and does not mean for early descriptive analyses. Now let me walk through the development steps, starting with the model comparison.

## Slide 8 — Three models were practically tied

I compared logistic regression, a decision tree, a random forest, gradient boosting and a base-rate dummy on the validation set. The selection rule was fixed in advance: the primary criterion was validation log loss; models whose paired-bootstrap interval for the log-loss difference contained zero were treated as tied; and within the tied set the most interpretable model was preferred. Logistic regression B, gradient boosting B and random forest B are practically tied on all three metrics. Their differences are far smaller than the bootstrap intervals. Only the decision tree is clearly worse, and every model beats the dummy. So I do not claim that logistic regression beat the others. For this dataset and protocol, LR B was retained because it is interpretable, has strong probability performance and is the simplest of the three.

## Slide 9 — Raw probabilities were already well calibrated

Five-fold stratified cross-validation on the training data confirmed the tie and gave a yardstick for noise: LR B had ROC-AUC 0.879 plus or minus 0.010, log loss 0.431 plus or minus 0.016 and Brier 0.139 plus or minus 0.006. Next, calibration. A probability is only useful if it means what it says: among students given about 0.8, roughly 80 percent should be placed. The calibration slope of the raw probabilities was 0.99, where 1 is ideal, and the expected calibration error was 0.021, close to what sampling noise alone would give. I also tried sigmoid and isotonic recalibration. The rule, fixed in advance, required a log-loss improvement of at least 0.005 with a confidence interval entirely below zero. No method met it, so the raw LR B probabilities were retained.

## Slide 10 — The threshold is a trade-off, not a model property

Producing a probability and turning it into a label are different tasks. The classification threshold is a policy choice. I fixed a rule in advance: on a grid from 0.20 to 0.80, pick the threshold with the highest validation F1, with ties going to the value closest to 0.50. That selected 0.35, with validation F1 of 0.766. At 0.35 recall is higher, 0.838 against 0.741, and precision is lower, 0.705 against 0.772. At 0.50 the rule is more conservative, with higher precision and higher specificity. But F1 is flat from about 0.30 to 0.55 and 0.50 is practically tied, so 0.35 is the selected operating threshold under an F1-oriented criterion, not a universally optimal one. The right choice depends on the relative cost of the two errors, which this project did not define.

## Slide 11 — What the model relies on to predict

Logistic regression is additive on the log-odds scale: the logit of the probability equals an intercept plus a sum of per-feature contributions, each being a coefficient times the transformed value, measured against a baseline student. At model level, the strongest associations were placement training with an odds ratio of about 2.75, extracurricular activities about 1.95 and aptitude about 1.65 per standard deviation. The chart shows permutation importance on validation: shuffling aptitude costs about 0.021 of AUC, placement training 0.018 and extracurricular activities 0.013, while internships add almost nothing once the others are included. These are predictive contributions, not causal effects. Ablation next.

## Slide 12 — Each group adds information; errors sit in overlaps

Ablation asks how much each feature group adds. Using logistic regression with the same preprocessing, validation AUC was 0.823 with academic features alone, 0.844 after adding aptitude, 0.856 after adding the technical counts, and 0.872 for the full model. Each added group improved performance in all five cross-validation folds, but the groups overlap, so the increments depend on order and are not independent contributions. On the right, error analysis at threshold 0.50 on validation: about 19.8 percent of students are misclassified, and roughly 70 percent of those errors lie where the predicted probability is between 0.2 and 0.8. Eighty-eight students were wrong with high confidence. The model is most uncertain where student profiles overlap.

## Slide 13 — One final held-out evaluation

This is the single final evaluation. The model specification, the thresholds and the metrics were frozen first; the final model was refit on train plus validation and evaluated once on the 1,487 test students. The three headline numbers are probability metrics: ROC-AUC 0.886, log loss 0.420 and Brier score 0.134. Please note that 0.886 is a ROC-AUC, not an accuracy. Calibration was good: slope 1.05 and expected calibration error 0.016. At threshold 0.50, accuracy is 0.812 with precision 0.779 and recall 0.765. At 0.35, recall rises to 0.851 and F1 to 0.781 while precision falls to 0.722: that is 53 fewer missed placed students for 69 more false alarms, and I do not claim a significant F1 difference. Test performance was broadly consistent with the development estimates.

## Slide 14 — The frozen model as an interactive application

The final step, Phase 8A, is a Streamlit application. A user enters the ten student characteristics; all fields are required and invalid values are rejected rather than silently clipped. The app loads the frozen Phase 7 artifact, LR B refit on train plus validation, exactly as saved: nothing is retrained, tuned or recalibrated. It returns the estimated probability, the classification at the 0.35 operating threshold, the 0.50 reference classification, a descriptive probability band, and an additive explanation of the contributions that reconstructs the model's log-odds to numerical precision. The application was verified against direct model predictions by a verification script that had 122 checks in its original Phase 8A form and has 178 in the final release form, all passing, and it never reads the held-out test split. This is a schematic of the workflow; no screenshot was captured, so I do not claim one.

## Slide 15 — Conclusions and limitations

To conclude, four points. First, leakage-aware cleaning changed the evaluation design: without it, repeated profiles would have leaked across splits. Second, logistic regression provided strong, interpretable probability estimates, practically tied with the nonlinear models. Third, calibration and threshold analysis made the probability output more meaningful. Fourth, the model reached ROC-AUC 0.886 on the final held-out evaluation. The limits matter as much: one dataset of uncertain representativeness, a structure that may be rule-based or constructed, associations that are not causes, a threshold that depends on costs, no external validation, and early descriptive analyses that touched the eventual test rows, which I disclosed; no model, threshold or calibration decision used them. The final line: a useful predictive estimate within this dataset and protocol, not a guarantee for any individual student. Thank you; I am happy to take questions.

---
Total words in notes: 1812 (about 12.1 minutes at 150 words per minute).
