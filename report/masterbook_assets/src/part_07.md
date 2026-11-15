# Glossary

| Term | Meaning (with our example) |
|---|---|
| Accuracy | share of correct labels at a threshold (0.812 at 0.50) |
| AUC / ROC-AUC | probability a random Placed student outranks a random Not Placed one (0.8857) |
| Bagging / bootstrap | train models on resamples drawn with replacement, then average (random forest) |
| Binary classification | two possible classes (Placed / NotPlaced) |
| Boosting | add weak models one after another, each fixing earlier errors (gradient boosting) |
| Brier score | mean of (p − y)²; lower is better (0.134) |
| C | inverse regularisation strength; smaller = stronger penalty (C = 1) |
| Calibration | whether stated probabilities match observed frequencies (slope 1.05, ECE 0.016) |
| CC0 | public-domain dedication; free reuse |
| Coefficient / weight | learned multiplier of a feature in the score (15 of them) |
| ColumnTransformer | applies different preprocessing to different columns |
| Confusion matrix | table of TP, TN, FP, FN |
| Convergence | optimiser stopped because the gradient was tiny (30 iterations) |
| Cost / loss function | single number measuring how wrong the model is (log loss) |
| Cross-validation | repeated train/hold-out rotation (5-fold stratified) |
| Data leakage | information from outside the training data influencing the model |
| Epoch | one pass over the data in iterative neural-network training (none here) |
| Feature / predictor | input column (10 raw, 15 processed) |
| Fit / transform | learn from data / apply what was learned |
| Frozen model | the saved artifact the app loads and never changes |
| Generalisation | doing well on unseen data |
| Gini impurity / entropy | measures of how mixed a tree node is |
| Hyper-parameter | setting chosen by us, not learned (C, max_iter, depth) |
| Intercept (b₀) | score for the reference student (−1.661) |
| L1 / L2 | penalty on |w| / on w² |
| L-BFGS | quasi-Newton optimiser (solver), not an activation function |
| Log-odds (logit) | ln(p/(1−p)) = z |
| Log loss / cross-entropy | −mean[y ln p + (1−y) ln(1−p)] |
| MLE | choose parameters that make the observed labels most probable |
| Odds / odds ratio | p/(1−p) / factor e^b by which odds change |
| One-hot encoding | one 0/1 column per category |
| Overfitting / underfitting | too tuned to training noise / too simple to learn the pattern |
| Pipeline | scikit-learn chain: preprocessing → model |
| Precision / recall / specificity / F1 | TP/(TP+FP) / TP/(TP+FN) / TN/(TN+FP) / harmonic mean of P and R |
| Reference level | the category dropped from one-hot encoding |
| Regularisation | penalty on large weights |
| Sigmoid | 1/(1+e⁻ᶻ); the logistic link |
| Standardisation | z = (x − μ)/σ |
| Stratified split | each part keeps the class proportion |
| Supervised learning | learning from labelled examples |
| Threshold | cut-off turning a probability into a label (0.35, 0.50) |

# Formula Sheet

| Topic | Formula |
|---|---|
| Mean, variance, SD | μ = Σx/n; σ² = Σ(x−μ)²/n; σ = √σ² |
| Z-score | z = (x − μ)/σ |
| Min-max (course) | x′ = (x − x_min)/(x_max − x_min) |
| IQR outlier fences | Q1 − 1.5·IQR, Q3 + 1.5·IQR |
| Linear predictor | z = b₀ + b₁x₁ + … + bₙxₙ |
| Sigmoid | p = 1/(1 + e⁻ᶻ) |
| Log-odds | ln(p/(1−p)) = z |
| Odds ratio | OR = eᵇ |
| Bernoulli | P(Y=y) = pʸ(1−p)¹⁻ʸ |
| Likelihood | L = Π pᵢʸⁱ (1−pᵢ)¹⁻ʸⁱ |
| Log loss | −(1/n) Σ[yᵢ ln pᵢ + (1−yᵢ) ln(1−pᵢ)] |
| Gradients (sum loss) | ∂L/∂w = Σ(pᵢ − yᵢ)xᵢ; ∂L/∂b = Σ(pᵢ − yᵢ) |
| Gradient step | w ← w − η·∂L/∂w |
| scikit-learn objective (L2) | (1/n)Σ loss + ‖w‖²/(2Cn)  ⇔  CΣ loss + ½‖w‖² |
| Gini; entropy | 1 − Σpₖ²; −Σpₖ log₂pₖ |
| Accuracy; precision; recall | (TP+TN)/n; TP/(TP+FP); TP/(TP+FN) |
| Specificity; F1 | TN/(TN+FP); 2PR/(P+R) |
| Brier | mean (p − y)² |
| Threshold in z | t = 0.35 ⇔ z ≥ −0.619; t = 0.50 ⇔ z ≥ 0 |

# One-Page Quick Revision Sheet

**Project in one line:** Ten student inputs → 15 processed columns → Logistic Regression (L2, C = 1, L-BFGS, 30 iterations) → probability → two thresholds (0.35, 0.50) → Streamlit app with explanations.

**Numbers to know:** 10,000 → −72 −16 → **9,912** · split **6,938 / 1,487 / 1,487** · seed **42** · refit **8,425** · 10 → **15** columns · tree models 18 columns · ≈**237** fits (estimate) · **30** iterations · test **AUC 0.8857** (0.868–0.902) · log loss **0.420** · Brier **0.134** · slope **1.05** · ECE **0.016** · at 0.50: **473 / 735 / 134 / 145**, accuracy 0.812, precision 0.779, recall 0.765, F1 0.772 · at 0.35: **526 / 666 / 203 / 92**, accuracy 0.802, precision 0.722, recall 0.851, F1 0.781 · typical profile **37.3%** (z = −0.52) · CGPA 8.45 → **43.9%** · CGPA range 6.5–9.1.

**Always say:** one model, one probability, two thresholds · AUC is not accuracy · LR tied with RF and GB, kept for simplicity and calibration · test set not used for development, selection, tuning, calibration or threshold choice (early descriptive EDA disclosed) · associations, not causes · 237 is an estimate · dataset origin unknown.

**Never say:** the test set was untouched · 0.35 is the best · 88.57% accuracy · real student data · deployed · epochs · SHAP · it is a neural network.

# Appendix A: Evidence Map

| Claim | Where it is verified |
|---|---|
| 10,000 raw rows; 72 duplicates; 8 groups / 16 rows; 9,912 | `results/metrics/02_data_quality_audit.csv`; `src/build_notebook_02.py` |
| Split 6,938 / 1,487 / 1,487, seed 42, stratified, two `train_test_split` calls | `results/metrics/02_split_summary.csv`; `src/build_notebook_02.py` lines 153–154 |
| Pipeline: `prep` (`num` StandardScaler 5, `cnt` OneHotEncoder drop [1,3,0], `bin` OneHotEncoder drop first) + `model` | loaded `models/phase7/LR_B_final.joblib` |
| 15 output columns and names; encoder categories | `pre.get_feature_names_out()`, `categories_` of the same artifact |
| C = 1, L2 (`l1_ratio=0`), lbfgs, max_iter 5000, no class weight, `n_iter_=[30]`, `random_state=42` | `get_params()` and `n_iter_` of the artifact; `models/phase7/final_model_configs.json` |
| Objective convention (intercept not penalised; 1/(C·n)) | `sklearn/linear_model/_logistic.py` (`l2_reg_strength = 1.0 / (C * sw_sum)`); numeric gradient check on the 8,425 rows (max \|gradient\| 6.2e-5 < tol 1e-4) |
| Scaler mean/σ from 8,425 rows (CGPA 7.6974 / 0.6390) | `pre.named_transformers_['num']` |
| Worked examples (37.3%, 43.9%, 2.1%, 89.8%) | `decision_function` / `predict_proba` of the artifact, recomputed by hand |
| Course examples (z = 6 → 0.9975; GD w = 0.4689, b = −0.0311) | `Model 5- Logistic Regression.pdf` slides 25–26, 42–44; recomputed in Python |
| DT_B 32 leaves depth 5; RF_B 300 trees; GB_B 300 stages; trees use 18 columns | loaded `models/phase4/*.joblib` |
| Fit counts (92, 4, 10, 1, 15, 75, 3, 31, 6) | loops in `src/build_notebook_05.py` – `08.py`; `05_tuning_grids.csv` (92 rows); `06_cv_fold_results.csv` (15); `07_feature_ablation_fold_results.csv` (25) |
| Test metrics (AUC, CI, log loss, Brier, ECE, slope, confusion counts) | `results/metrics/modeling/08_test_classification_metrics.csv`, `08_test_bootstrap_ci.csv`, `08_test_calibration_metrics.csv`; `app/model_metadata.json` |
| Threshold rule and evidence | `results/metrics/modeling/07_threshold_selection.json` |
| Model-selection rule | `results/metrics/modeling/05_provisional_selection.json` |
| Explainability method; UI cut of ±0.005 | `app/explainability.py`; `app/ui_helpers.py` (`factor_html`) |
| Test-set access history | `08_frozen_protocol.json`, `08_test_access_log.json`; `src/phase7_findings.md` |
| Tools and versions | `requirements-app.txt`; `requirements.txt`; installed versions printed in this session |

# Appendix B: Unknowns and Estimates

| Status | Item |
|---|---|
| <span class="tag u">UNKNOWN</span> | Origin of the dataset (real or generated); the meaning of the columns beyond their names; how many times notebooks were re-run during development; total training runtime and hardware; per-fit iteration counts (only the final model's 30 is saved) |
| <span class="tag e">ESTIMATE</span> | ≈237 pipeline fits (and ≈36 calibrator fits): counts of code paths in one execution of the final build scripts, not an execution log |
| Not re-verified here | CC0 licence page online (recorded in project documents); the "178 checks passed" figure is the 27 October 2026 record (re-run recommended before your viva: `python app/verify_app.py`); UI fix tests mentioned in project history (46/46, 37/37) have no saved record |
| Manual review suggested | Compare the "Typical profile → 37.3%" and "CGPA 8.45 → 43.9%" numbers in your live app; confirm your name, date and university on the cover |

# Appendix C: Course Coverage Map

| Topic | In your slides? | In our project? |
|---|---|---|
| Descriptive / predictive / prescriptive | Yes (Unit 1, Lectures 1–3) | descriptive and predictive only |
| Data collection, cleaning, missing values, outliers (z-score, IQR) | Yes (Module 2) | duplicates and conflicts cleaned; IQR informational; no imputation |
| Normalisation, standardisation, label and one-hot encoding, binning, log transform | Yes | standardisation and one-hot used; the rest not |
| Feature engineering and selection (filter, wrapper, embedded, PCA) | Yes | not used (L2 regularisation is a form of embedded shrinkage, but no selection was done) |
| Charts: bar, histogram, pie, heat-map, scatter | Yes | all except pie used (plus box, violin, ECDF) |
| Linear regression (least squares, multiple) | Yes | background only |
| Logistic regression: sigmoid, MLE, Bernoulli, cross-entropy, gradient descent | Yes | **core of the project** |
| Multinomial (softmax) and ordinal logistic regression | Yes | not used (binary target) |
| Regularisation, L-BFGS, C | No slides | used (supplement in Chapter 7) |
| Train/validation/test, cross-validation, ROC-AUC, F1 | Syllabus only | used (supplement in Chapters 5 and 10) |
| Decision trees, random forest, boosting | No slides | used for comparison (Chapter 8) |
| Time series, LDA/QDA, Poisson | Syllabus | not used |
