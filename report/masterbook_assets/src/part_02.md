# 4. Data Cleaning and Preparation

## 4.1 Why raw data cannot always be used directly <span class="tag c">COURSE</span>

Your course (Module 2) defines data cleaning as "detecting, correcting, or removing inaccurate, incomplete, duplicate, inconsistent, and irrelevant data" and lists five steps: handle missing values, remove duplicates, correct inconsistent data, handle outliers, standardise formats. Its own baby example: "If your school prints two ID cards accidentally, only one valid card should be kept."

Our audit (notebook 02) applied the same checklist:

| Check | Result | Action |
|---|---|---|
| Missing values | none | nothing to impute |
| Invalid values (CGPA outside 0–10, marks outside 0–100, negative or non-integer counts) | none | nothing to fix |
| Inconsistent formats | none (text columns only contain Yes/No and Placed/NotPlaced) | nothing to standardise |
| Outliers | IQR rule flags none in the five score columns | nothing removed |
| **Exact duplicate rows** | 66 groups = 138 rows; 72 surplus copies | **kept one copy of each: −72** |
| **Conflicting labels** | 8 groups = 16 rows | **excluded all of them: −16** |
| Overlap between the supplied train/validation/test files | 33 feature profiles (69 rows) appeared in more than one file | removed by rebuilding the split (Chapter 5) |

<div class="note" markdown="1">
**Do not invent other filtering.** There were exactly two removal operations: surplus exact duplicates (72 rows) and conflicting-label rows (16 rows). Nothing else was filtered, no outlier was dropped, and nothing was imputed.
</div>

## 4.2 The two operations in detail <span class="tag v">VERIFIED</span> (`src/build_notebook_02.py`)

**(a) Exact duplicates.** Two rows are exact duplicates if all 11 columns (10 features and the label) are identical. Identical students with the same answer add no new information and make the model over-trust that profile. In pandas:

```python
exact_extra = df.duplicated(subset=COLS, keep="first")   # True for every copy after the first
n_exact_extra = int(exact_extra.sum())                    # 72
```

**(b) Conflicting labels.** Group the rows by their ten features. A group in which the label is *not* the same everywhere is a conflict. Eight such groups exist, each with exactly two rows (one Placed, one NotPlaced). All eight have the top-end profile (aptitude 90, two workshops, three projects, extracurricular and training both Yes). With no majority in a pair and no way to know which label is right, we excluded the rows instead of guessing:

```python
kept = df[~is_conf]                                        # drop the 16 conflicting rows
model_df = kept.drop_duplicates(subset=FEATURES, keep="first")   # then one row per remaining profile
assert 10000 - 16 - 72 == len(model_df) == 9912
```

![Cleaning flow with verified counts](masterbook_assets/img/g_cleaning.png)

**Effect on the data.** 88 of 10,000 rows (0.88%) were affected. Because the 16 contradictory rows are all top-end profiles, the cleaned data slightly under-represent that corner: a stated limitation, not a measured one.

## 4.3 Data leakage <span class="tag s">SUPPLEMENT</span>

**What:** any route by which information that would not exist at prediction time (or that belongs to the test data) influences training. **Why it matters:** it makes the measured score look better than the real score. **Examples in our project that we guarded against:**

1. The *same student profile* in both train and test files (the 33 overlapping profiles): the model would be graded on something it had memorised.
2. Fitting the scaler or encoder on all data, so test information flows into the mean/SD: avoided by fitting inside the pipeline on training rows only (Chapter 6).
3. Using the test set to choose the model or threshold: avoided by the validation set and a frozen protocol (Chapter 5).

<div class="wrap" markdown="1">
**Key takeaways.** 10,000 − 72 duplicates − 16 conflicting rows = **9,912**. No missing/invalid values, no outliers removed. Cleaning also removed cross-file leakage.
**Formulas.** 10,000 − 72 − 16 = 9,912.
**Revision questions.** (1) Difference between exact duplicate and conflicting duplicate? (2) Why not take a majority vote? (3) How many groups and rows were excluded as conflicting? (4) Why did we also rebuild the split? (5) Define data leakage with one example.
**Professor may ask.** "Why drop instead of fix the 16 rows?" No ground truth exists to decide which label is right. "Could filtering bias the result?" Slightly, at the top end; only 0.88% of rows.
</div>

# 5. Train, Validation and Test Splitting

*The course slides do not contain a lecture on this topic (the syllabus lists it under Model Validation). This chapter is standard theory* <span class="tag s">SUPPLEMENT</span> *connected to our code.*

## 5.1 Why we split, and the three roles

If you teach a student with exam paper A and then test with exam paper A, a high mark tells you they remembered paper A, not that they understood the subject. The same holds for models.

| Set | Rows | Role | Analogy |
|---|---|---|---|
| **Training** | 6,938 | The model learns its numbers here (scaler mean/SD, coefficients) | practice questions |
| **Validation** | 1,487 | We compare models and settings, and choose the threshold here | a mock exam |
| **Test** | 1,487 | Final one-time score after all choices are frozen | the real exam |

![The split used in the project](masterbook_assets/img/f_split.png)

6,938 + 1,487 + 1,487 = 9,912. After all choices were frozen, the final model was **refitted on train + validation = 8,425 rows**, with the same settings, to use more data. The test rows were never part of any fit.

**Overfitting** = the model learns the quirks and noise of the training rows, so it scores very well on them and badly on new rows. **Underfitting** = the model is too simple to capture even the real pattern, so it scores badly everywhere. **Generalisation** = performing well on rows it has never seen, which is the actual goal. *Our own example of overfitting:* an unrestricted decision tree (`max_depth=None`, `min_samples_leaf=1`) reached **training AUC 1.000** but a validation AUC of only 0.67–0.68 and a validation log loss of 11.05–11.47 (`05_tuning_grids.csv`). Restricting the depth fixed it.

## 5.2 How we split <span class="tag v">VERIFIED</span> (`src/build_notebook_02.py`)

```python
train, tmp = train_test_split(model_df, test_size=0.30, stratify=model_df[TARGET], random_state=42)
val, test  = train_test_split(tmp,      test_size=0.50, stratify=tmp[TARGET],      random_state=42)
```

* **Two calls**: first 70% / 30%, then the 30% is halved into 15% / 15%.
* **Stratified** means each part keeps the same class proportion as the whole: Placed is 41.55% (train), 41.56% (validation), 41.56% (test), 41.56% overall. Without stratification a small test set could by chance contain far fewer Placed students and give a misleading score.
* **Random seed.** Shuffling uses pseudo-random numbers; `random_state=42` fixes the sequence, so everyone who re-runs the code gets the identical split and results. That is **reproducibility**. The number 42 has no special meaning; any fixed number works.
* The three parts share **zero** feature profiles (checked: `final_cross_split_profile_overlap = 0`).

## 5.3 Cross-validation <span class="tag s">SUPPLEMENT</span>

A single validation set gives one noisy estimate. **K-fold cross-validation** splits the training data into *K* equal folds; the model is fitted *K* times, each time holding out a different fold for scoring, and the *K* scores are averaged. **Stratified K-fold** keeps the class ratio in every fold.

![5-fold cross-validation](masterbook_assets/img/f_cv.png)

*Our usage:* `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 6,938 training rows (notebook 06). It was used to check the *stability* of the three leading models and to test calibration methods, not to pick between them. Result for LR B: mean ROC-AUC 0.879 (fold SD 0.0095). Each fit uses 80% of the training rows.

## 5.4 The honest statement about the test set <span class="tag v">VERIFIED</span>

| Phase | What touched the eventual test rows | What it was used for |
|---|---|---|
| 1–3 (data understanding, quality, EDA, statistics) | The test rows were loaded for **descriptive** analysis (summaries, plots, tests on the whole dataset) | understanding the data; **not** fitting or scoring any prediction model |
| Orientation commands | Two shell commands (`head`, `wc -l`) glanced at `test.csv` | nothing was used from them |
| 4–6 (preprocessing, models, calibration, threshold) | The notebooks' own audit hooks blocked access to the test file | model development, selection, tuning, calibration, threshold |
| 7 (final evaluation) | **One gated read**, logged 2026-10-02 18:51 UTC (`08_test_access_log.json`), after the protocol was frozen (`08_frozen_protocol.json`) | the single reported test result |

<div class="note" markdown="1">
**Say:** "The test set was not used for model development, selection, tuning, calibration or threshold selection." **Never say** "the test set was untouched throughout the project": early descriptive EDA did look at those rows. **Limitation this introduces:** the descriptive look (for example, knowing the overall placement rate or a feature's range) is a weak form of information leak; it cannot change the fitted numbers, but it means independence is not absolute. In addition, we chose the final model on validation scores, which makes validation-based figures slightly optimistic; the test score is therefore the cleaner estimate.
</div>

<div class="wrap" markdown="1">
**Key takeaways.** Split 70/15/15 = 6,938 / 1,487 / 1,487, stratified, seed 42, two `train_test_split` calls. Validation for choosing, test for one final score, refit on 8,425. Five-fold stratified CV on training rows for stability only. Test-set wording must be qualified.
**Formulas.** n = 6,938 + 1,487 + 1,487; CV score = mean of K fold scores.
**Revision questions.** (1) Why not evaluate on training data? (2) Overfitting vs underfitting? (3) What does stratify do? (4) What does `random_state=42` guarantee? (5) What was CV used for here?
**Professor may ask.** "Did you touch the test set?" Qualified answer above. "Why also refit on validation?" More data after choices are frozen.
</div>

# 6. Data Preprocessing and Feature Engineering

## 6.1 What and why <span class="tag c">COURSE</span>

The course says preprocessing is "the process of making raw data ready for processing and analysis", and "machine learning algorithms require numerical input". A logistic-regression formula multiplies numbers by weights; it cannot multiply the word "Yes". The course lists the transformations we need: **feature scaling** (standardisation, normalisation) and **encoding** (label encoding, one-hot encoding). **Feature engineering** (creating new features such as BMI from weight and height) was *not* used in our project: we only encoded and scaled the ten original columns.

## 6.2 Concept card: Standardisation (z-score) <span class="tag c">COURSE</span>

* **What.** Rescale a numeric column so that its mean is 0 and its standard deviation is 1.
* **Why.** Our columns live on different scales (CGPA ≈ 6.5–9.1, marks ≈ 55–90). Two reasons matter for Logistic Regression: the L2 penalty (Chapter 7) punishes all coefficients equally, which is only fair if the inputs are on comparable scales; and the optimiser converges faster and more reliably. (The course note says standardisation is "required for Logistic Regression"; strictly, an *unpenalised* logistic regression would still give valid answers without it, but ours is penalised, so it matters.)
* **Simple words.** "How many standard deviations above or below the average is this student?"
* **Formula.**

<div class="eq">z = (x − μ) / σ</div>

*x* = the raw value; *μ* = mean of the column; *σ* = standard deviation of the column; *z* = standardised value. scikit-learn's `StandardScaler` uses the *population* σ (divides by *n*, `ddof=0`), the same as the course's examples.

* **Course worked example** {40, 50, 60, 70, 80}: μ = 60, σ = 14.14, so z = −1.41, −0.71, 0.00, 0.71, 1.41.
* **Our worked example (real numbers from the saved scaler).** The scaler stored, for CGPA, μ = 7.6974 and σ = 0.6390. A student with CGPA 8.45: z = (8.45 − 7.6974)/0.6390 = **+1.178**. For aptitude 80: μ = 79.3215, σ = 8.1600, z = **+0.083**.

<div class="note" markdown="1">
**Where do μ and σ come from?** From the 8,425 rows the final model was fitted on (train + validation), not from the training rows alone. (`model_metadata.json` calls the mean "train_mean": a loose label; the saved scaler's CGPA mean is 7.69740, while the train-only mean is about 7.699.)
</div>

* **Code.** `StandardScaler()` inside the pipeline (never by hand).
* **Without it.** The penalty would favour large-scale columns and the optimiser may need many more iterations.
* **Normalisation** (min-max, course formula x′ = (x − x_min)/(x_max − x_min)) squeezes into [0, 1]. We did **not** use it. Comparison: standardisation has no fixed range and is less sensitive to outliers (course table), which is why it suits regression-type models.
* **Limits.** Assumes mean and SD are meaningful; it does not change the shape of the distribution.

## 6.3 Concept card: One-hot encoding <span class="tag c">COURSE</span>

* **What.** Replace one categorical column by several 0/1 columns, one per category. The course example turns Colour {Red, Green, Yellow} into three columns Red, Green, Yellow.
* **Why not label encoding (0, 1, 2)?** It would tell the model that level 2 is "twice" level 1 in a straight line, which our EDA shows is false for the counts (workshops: 28.4%, 21.1%, 72.6%, 56.3%).
* **Dummy-variable trap / dropping a category.** If you keep *all* K columns, they always add up to 1, which duplicates the intercept (perfect redundancy) and makes each coefficient depend arbitrarily on the others. Dropping one **reference level** gives K − 1 columns; each coefficient then reads "compared with the reference level". <span class="tag s">SUPPLEMENT</span>
* **Our actual encoders** (loaded from the artifact):

| Columns | Categories found by the fitted encoder | Dropped (reference) | Output columns |
|---|---|---|---|
| Internships | 0, 1, 2 | 1 | `Internships_0`, `Internships_2` |
| Projects | 0, 1, 2, 3 | 3 | `Projects_0`, `Projects_1`, `Projects_2` |
| Workshops/Certifications | 0, 1, 2, 3 | 0 | `Workshops/Certifications_1`, `_2`, `_3` |
| ExtracurricularActivities | No, Yes | No (`drop='first'`) | `ExtracurricularActivities_Yes` |
| PlacementTraining | No, Yes | No | `PlacementTraining_Yes` |

The reference levels for the three counts are the most frequent level in the training data (the code uses the training mode); `handle_unknown='error'` means an unseen category would raise an error instead of silently being ignored.

* **Worked example.** A student with Internships = 2, Projects = 3, Workshops = 1, Extracurricular = Yes, Training = No becomes:

| Internships_0 | Internships_2 | Projects_0 | Projects_1 | Projects_2 | Work_1 | Work_2 | Work_3 | Extra_Yes | Train_Yes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **1** | 0 | 0 | 0 | **1** | 0 | 0 | **1** | 0 |

Projects = 3 is the reference, so all three Projects columns are 0.

## 6.4 From 10 predictors to 15 processed features <span class="tag v">VERIFIED</span>

![Pipeline: raw inputs → 15 columns → score → probability → thresholds](masterbook_assets/img/f_pipeline.png)

| Group | Columns in | Transformer | Columns out |
|---|---|---|---|
| continuous | CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks | `StandardScaler` | 5 |
| counts | Internships, Projects, Workshops/Certifications | `OneHotEncoder(drop=[1, 3, 0])` | 2 + 3 + 3 = 8 |
| binary | ExtracurricularActivities, PlacementTraining | `OneHotEncoder(drop='first')` | 1 + 1 = 2 |
| | | **Total** | **5 + 8 + 2 = 15** |

**No column is "passed through unchanged":** the two Yes/No columns are one-hot encoded (they end up as a 0/1 column, but that is the encoder's output, not a pass-through). `ColumnTransformer(remainder='drop')` would discard any other column, and `verbose_feature_names_out=False` keeps the plain column names above. The 15 output names in order: CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks, Internships_0, Internships_2, Projects_0, Projects_1, Projects_2, Workshops/Certifications_1, Workshops/Certifications_2, Workshops/Certifications_3, ExtracurricularActivities_Yes, PlacementTraining_Yes.

## 6.5 Fit, transform, fit_transform and pipelines <span class="tag s">SUPPLEMENT</span>

| Call | What it does | Example |
|---|---|---|
| `fit(X)` | *Learns* quantities from the data | scaler learns μ and σ per column; encoder learns the category list |
| `transform(X)` | *Applies* what was learned (no learning) | z = (x − μ)/σ using the stored μ, σ |
| `fit_transform(X)` | fit then transform on the same data | used on training data only |

**Rule:** learn only from training rows; apply to validation, test and app inputs. If validation or test data entered `fit`, their information would leak into the scaler. We never call these by hand: `Pipeline([("prep", ColumnTransformer(...)), ("model", LogisticRegression(...))])` fits the preprocessing and the model together in one `.fit(X_train, y_train)`, and calls `.transform` automatically in `.predict_proba(X_new)`. This guarantees that the app uses exactly the transformation learned in training.

```python
pipe = Pipeline([("prep", ColumnTransformer([
            ("num", StandardScaler(), CONT),
            ("cnt", OneHotEncoder(drop=[1, 3, 0], sparse_output=False), COUNT),
            ("bin", OneHotEncoder(drop="first", sparse_output=False), BIN)])),
         ("model", LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=5000, random_state=42))])
pipe.fit(X_train, y_train)
```

<div class="wrap" markdown="1">
**Key takeaways.** Standardise 5 scores (z = (x − μ)/σ); one-hot the 3 counts (reference dropped) and the 2 Yes/No columns; 10 → 15 columns. All learned inside a Pipeline from fitting rows only.
**Formulas.** z = (x − μ)/σ; x′ = (x − min)/(max − min) (course, not used); K categories → K − 1 columns.
**Revision questions.** (1) Why can't the model use the word "Yes"? (2) What does `fit` store in the scaler? (3) Why drop a category? (4) How do 10 columns become 15? (5) Why fit preprocessing on training data only?
**Professor may ask.** "Standardise or normalise?" We standardised, because the L2 penalty and L-BFGS behave better, and it is less outlier-sensitive. "Why one-hot for Internships if it is a number?" EDA showed non-monotonic effects.
</div>
