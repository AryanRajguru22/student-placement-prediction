# 11. Why Are There Two Thresholds?

This is the most likely "trap" topic in the viva. Learn the one-sentence answer: **the app uses ONE model and ONE probability, and applies TWO classification thresholds to that same probability.**

## 11.1 From probability to label

The model outputs p = P(Placed), a number between 0 and 1. A *label* needs a decision rule: **predict Placed if p ≥ threshold**, otherwise Not Placed (this is `classify()` in `app/prediction.py`). The threshold is a *policy choice* about how to trade two kinds of mistake; it is not something the model learns.

* **0.50** is the conventional value: it equals "z ≥ 0", the course's decision boundary, and treats the classes symmetrically. We keep it as a **reference**.
* **0.35** is our **project (operating) threshold**, chosen by a rule fixed *before* looking: *the grid value from 0.20 to 0.80 (step 0.05) with the highest **validation** F1, ties going to the value closest to 0.50*.

![One probability, two decision rules](masterbook_assets/img/f_thresholds.png)

## 11.2 What lowering the threshold does

| | Threshold 0.50 | Threshold 0.35 | Change |
|---|---|---|---|
| Students labelled Placed | 607 (40.8%) | 729 (49.0%) | +122 |
| TP (placed, found) | 473 | 526 | **+53** |
| FN (placed, missed) | 145 | 92 | **−53** |
| FP (false alarms) | 134 | 203 | **+69** |
| TN | 735 | 666 | −69 |
| Recall / Precision | 0.765 / 0.779 | 0.851 / 0.722 | recall ↑, precision ↓ |

**Why recall rises:** a lower bar admits more students into "Placed", so more of the truly placed pass it. **Why precision falls:** the extra students admitted also include more not-placed ones. This is the **precision–recall trade-off**; moving the threshold slides you along it. **The probability itself never changes**: the student with p = 0.373 has p = 0.373 under both rules.

**Which error is worse?** *FP*: predicting Placed for a student who is not (false comfort). *FN*: predicting Not Placed for one who would be (a missed or discouraged student). The costs depend on how the output would be used, and the project **never defined them**. Illustration only: if a placement cell wanted to find students who might need extra support, it would prefer a threshold that errs toward flagging more people.

## 11.3 The evidence for 0.35, honestly <span class="tag m">REPORTED</span> (`07_threshold_selection.json`)

* Validation F1 at 0.35 = **0.766**; at 0.50 = **0.756**. F1 at 0.30 / 0.35 / 0.40 = 0.760 / 0.766 / 0.764: a **plateau** (0.35, 0.40 and 0.45 are within 0.005 of each other).
* Paired bootstrap difference in F1 (0.35 − 0.50) on validation: **+0.010, 95% interval −0.007 to +0.028, which includes zero**: no conclusive advantage.
* Cross-validated (out-of-fold) F1 was best at 0.40; 0.35 beat 0.50 in 4 of 5 folds.
* On the **test** set: F1 0.781 (0.35) vs 0.772 (0.50), but accuracy is *lower* at 0.35 (0.802 vs 0.812). A paired test interval for that F1 difference was **not computed**; the two separate F1 intervals (0.756–0.804 and 0.744–0.797) overlap almost entirely.

**So:** 0.35 is "the selected operating threshold under the F1-oriented rule"; 0.50 is "practically tied" with it. *Neither is universally optimal.* Never call 0.35 "the best threshold".

## 11.4 The UI example

Typical profile: p = **37.3%**. At 0.35: 0.373 ≥ 0.35 → **Placed**. At 0.50: 0.373 < 0.50 → **Not Placed**. Same probability, different label, because the two rules draw the line in different places. That is why the app shows both side by side and presents the probability as the primary output.

<div class="wrap" markdown="1">
**Key takeaways.** One model → one p → two decision rules. 0.50 conventional; 0.35 chosen by a predeclared validation-F1 rule. Lower threshold → higher recall, lower precision. Neither is proven best.
**Formulas.** label = Placed if p ≥ t; z-threshold: t = 0.35 ⇒ z ≥ −0.619; t = 0.50 ⇒ z ≥ 0.
**Revision questions.** (1) How many models does the app use? (2) Why is recall higher at 0.35? (3) How was 0.35 selected? (4) Why is it not "the best"? (5) What happens to a student at p = 0.40?
**Professor may ask.** "Which threshold would you use in practice?" Depends on the cost of FP vs FN, which must be defined first. "Why is accuracy lower at 0.35?" We trade some correct negatives for more caught positives.
</div>

# 12. Explainability and Feature Contributions

## 12.1 What it is and why we need it <span class="tag s">SUPPLEMENT</span>

**Explainability** = being able to say *why* the model gave a particular estimate. It matters because a number such as 37.3% is much more useful (and checkable) if the user can see which inputs pushed it up or down, and because the course lists "Explainable AI / model interpretability / transparency and accountability" among the ethical considerations of predictive modelling (Lecture 3).

## 12.2 What our app actually does <span class="tag v">VERIFIED</span> (`app/explainability.py`, `app/ui_helpers.py`)

The technique is **coefficient × transformed value**, read directly from the logistic model. It is **not SHAP, not LIME, not permutation importance**: none of those is computed by the app.

1. `x = pipeline.named_steps["prep"].transform(input_row)` gives the 15 processed values.
2. `terms = x * model.coef_[0]`: one term per processed column.
3. Terms are **summed per original feature** (for example the three Projects columns together).
4. Check: `intercept + Σ contributions = z`, the exact log-odds (reconstruction verified to 1e-9 by `verify_app.py`).
5. Each contribution is also shown as an **odds multiplier** e^(contribution).

**Baseline.** Contributions are relative to the reference student: every score at the training average (x = 0 after scaling), every count at its reference level, Yes/No at "No". Hence "Projects = 2 contributes −0.31" means "compared with 3 projects".

**How the UI decides what to show.** `factor_html()` splits the contributions: **≥ +0.005** go under "Factors that raise the model's estimate" (largest first), **≤ −0.005** under "Factors that lower…", and any with |contribution| < 0.005 under "At or very near the baseline". Bars scale to the largest contribution *within that prediction*.

**Our typical-profile example** (z = −0.52): placement training +0.97, extracurricular +0.72, soft skills +0.05, aptitude +0.04, SSC +0.02; workshops −0.32, projects −0.31, HSC −0.03; CGPA (+0.0009) and internships (0) near baseline. Intercept −1.66 + 1.14 = −0.52.

**Related analysis in notebook 07 (not in the app)** <span class="tag m">REPORTED</span>: global coefficients with odds ratios; **permutation importance** on validation data (20 repeats; top three by AUC drop: AptitudeTestScore 0.021, PlacementTraining 0.018, ExtracurricularActivities 0.013); an unpenalised statsmodels logit for inference; and feature-group ablation.

## 12.3 Limitations

* **Association, not causation.** A bar shows how the model *builds its score*, not what would happen if a real student changed that input. Higher CGPA goes with placement in this dataset; it has not been shown to cause it.
* **Baseline dependent.** Change the reference level and the numbers change.
* **Correlated features share credit** (aptitude and HSC, r = 0.56; extracurricular and training overlap 85%).
* **A local explanation of one prediction**: it does not say how reliable the model is.
* For linear models the exact contribution is simple; for the tree models in Chapter 8 it would not be.

<div class="wrap" markdown="1">
**Key takeaways.** Explanations = coefficient × processed value, summed per feature; exact for LR; baseline is the reference student; UI sorts positive/negative with a ±0.005 cut. No SHAP. No causation.
**Formulas.** contribution_f = Σ over columns of f of x_j·b_j; z = b₀ + Σ contributions; multiplier = e^contribution.
**Revision questions.** (1) Is SHAP used? (2) What does "baseline" mean here? (3) Why do contributions sum to z? (4) What does a negative bar mean? (5) Why are explanations not causal?
**Professor may ask.** "Why is it exact?" Because the model is linear in the log-odds. "Why is CGPA near baseline for the typical student?" CGPA 7.7 ≈ the average 7.697, so its scaled value ≈ 0.
</div>

# 13. The Technology Stack, as Actually Used

| Technology | What it is (simple) | Where we use it | Version |
|---|---|---|---|
| **Python** | General-purpose language; the common language of data science | everything | 3.11.9 (3.11+ required) |
| **Pandas** | Library for tables ("DataFrames"): read CSV, filter, group, describe | data cleaning, EDA, building the one-row input (`to_frame`) | 3.0.5 |
| **NumPy** | Fast arrays and maths on whole arrays at once | `x * model.coef_`, exp, bootstrap indices | 2.4.2 |
| **Matplotlib & Seaborn** | Plotting (Seaborn is a nicer layer on Matplotlib) | notebooks and report figures only; **not** in the app | in `requirements.txt` |
| **SciPy / statsmodels** | Statistical tests / inference models | notebooks 04–07 (Mann-Whitney, chi-square, VIF, unpenalised logit); SciPy's `L-BFGS-B` is the optimiser behind `solver='lbfgs'` | 1.17.0 / (notebooks) |
| **scikit-learn** | The machine-learning library | `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `LogisticRegression`, tree models, `train_test_split`, `StratifiedKFold`, metrics | **1.9.1** (pinned) |
| **Streamlit** | Turns a Python script into a web page | the app (`app/app.py`) | 1.64.0 |
| **Joblib** | Saves/loads Python objects, efficient for NumPy data | `joblib.dump` saves the fitted pipeline; `joblib.load` in the app | 1.6.0 |
| **Git & GitHub** | Version control and online hosting of the repository | the project is a git repository, branch `main`, remote `origin` = `github.com/AryanRajguru22/student-placement-prediction` | – |
| Jupyter / nbformat | Notebook files; the scripts `src/build_notebook_*.py` generate them | notebooks 01–08 | `requirements.txt` |

**Not used:** SHAP, LIME, TensorFlow/PyTorch (no neural networks), `GridSearchCV`, a database, cloud hosting.

**How they fit together.** *Training time*: pandas reads CSVs → scikit-learn Pipeline preprocesses and fits → metrics computed → `joblib.dump`. *Inference time*: Streamlit form → validation → pandas one-row DataFrame → `joblib.load`-ed Pipeline → `predict_proba` → NumPy explanation → Streamlit display.

## 13.1 Scikit-learn's grammar

Every object follows the same interface: **estimators** have `fit`; **transformers** have `transform`; **predictors** have `predict` / `predict_proba`. A `Pipeline` chains them and exposes `.named_steps`. This uniformity is why our app can say `pipeline.predict_proba(df)` without knowing what is inside.

## 13.2 Streamlit in five ideas

1. **The whole script re-runs from top to bottom on every user interaction** (a click, a typed value). State must therefore be kept explicitly.
2. **Widgets** return values: `st.number_input`, `st.selectbox`, `st.radio`, `st.button`.
3. **`st.session_state`** keeps values between re-runs (we store the last prediction there, and drop it if any input changes, so a stale result is never shown).
4. **`st.cache_resource`** runs a function once per server session; we use it so the model loads (and its SHA-256 is checked) only once.
5. **Layout**: `st.columns`, `st.container`, `st.expander`; results via `st.markdown`, `st.table`, `st.error`, `st.warning`.

## 13.3 Joblib and the frozen model

`joblib.dump(pipeline, "LR_B_final.joblib")` writes the entire fitted Pipeline (scaler statistics, encoder categories, coefficients). `joblib.load` restores it. Because the saved object only works with the scikit-learn version that made it, the app records the version (1.9.1) and the file's SHA-256 in `model_metadata.json` and refuses to start if either differs.

## 13.4 Git and GitHub

*Git* records snapshots (**commits**) of the project so you can see, compare or restore earlier versions; *branches* are parallel lines of work (ours: `main`); *GitHub* is the website that stores a remote copy. The `.gitignore` deliberately excludes `.venv/`, caches and every file containing the held-out test rows (`DATA_NOTE.md`).

<div class="wrap" markdown="1">
**Key takeaways.** Python + pandas/NumPy for data, scikit-learn for the model, joblib to freeze it, Streamlit to serve it, git/GitHub to version it. Matplotlib/Seaborn/statsmodels only in notebooks.
**Revision questions.** (1) Why joblib? (2) Why is the scikit-learn version pinned? (3) What does `st.cache_resource` do? (4) Which libraries are *not* in the app? (5) What does a `Pipeline` guarantee?
**Professor may ask.** "Why Streamlit over Flask?" Pure-Python UI with no front-end code; enough for a demo. "Do you retrain on every click?" No: frozen model.
</div>

# 14. Application Architecture

## 14.1 Training time versus prediction time

![Training time and prediction time](masterbook_assets/img/g_architecture.png)

**Why the app loads a frozen model instead of training.** Training takes the 8,425 labelled rows and many iterations, would need the dataset (the app never reads any dataset file), and could produce a slightly different model each time. A frozen artifact gives the same answer for the same input every time, starts instantly, and is exactly the model whose test results we reported.

## 14.2 One prediction, step by step <span class="tag v">VERIFIED</span>

| # | What happens | Code (file → function) |
|---|---|---|
| 1 | The user opens the page; the model is loaded **once**: metadata read, SHA-256 and scikit-learn version checked, pipeline loaded, ten feature names confirmed | `app.py` → `get_bundle()` (`@st.cache_resource`) → `prediction.load_bundle()` |
| 2 | The user fills ten fields (all empty at first) or presses **Load typical profile** | `app.py` → `num_input`, `level_input`, `st.radio`; `load_typical_profile` |
| 3 | **Validation** on pressing **Predict Placement Probability**: missing, non-numeric, non-finite, out-of-range or unknown-level values give an error message; nothing is clipped; no prediction is made | `prediction.validate_inputs(raw, metadata)` |
| 4 | Clean values become a one-row DataFrame with the exact column names, counts as `int` | `prediction.to_frame()` |
| 5 | The saved pipeline's `prep` step scales and one-hot encodes | inside `pipeline.predict_proba` |
| 6 | The logistic model computes z (`decision_function`) | `prediction.predict()` |
| 7 | The sigmoid turns z into p (`predict_proba[0, 1]`) | same |
| 8 | Two thresholds give two labels; a descriptive band (0–0.2 "very low" … 0.8–1 "very high") is attached | `classify()`, `probability_band()` |
| 9 | Contributions are computed | `explainability.explain()` |
| 10 | Streamlit renders percentage, labels, band, factor bars, calculation table, inputs used, and the disclaimer | `ui_helpers.result_card`, `factor_bars`, `disclaimer` |

**Invalid values.** Example: CGPA 9.8 → "CGPA: 9.8 is outside the range the model was developed on (6.5 to 9.1); no prediction is made for out-of-range values." The result panel is removed. Why: the model never saw such values, so any number would be extrapolation, and would *look* more reliable than it is. A real student with a 9.8 CGPA is plausible; supporting that needs new data and a re-fit (Chapter 16).

**Other guard rails.** An input edited after a prediction removes the stale result; rare levels (Projects = 0: 22 students) trigger a warning; the entries are never written to disk or logged; `config.toml` binds the server to `localhost`.

**Testing** <span class="tag v">VERIFIED</span>. `python app/verify_app.py` ran 178 checks with 0 failures when the project guide was prepared (27 October 2026): model hash, probabilities equal direct pipeline output, thresholds at boundaries, explanation reconstruction, validation, a headless end-to-end run, metadata consistency, and that no test file was opened. Passing tests show the *code* does what was designed, not that the *predictions* are right for real students.

<div class="wrap" markdown="1">
**Key takeaways.** Load once → validate → DataFrame → pipeline (scale, encode, z, sigmoid) → two thresholds → explanation → render. Frozen, hash-checked artifact. Out-of-range inputs are refused.
**Revision questions.** (1) What does `get_bundle()` check? (2) What does `validate_inputs` do with 9.8? (3) Where is the sigmoid applied? (4) Why not clip 9.8 to 9.1? (5) What is stored in session state?
**Professor may ask.** "Does the app retrain?" No. "Can I trust the model hash check?" It detects a swapped or mismatched file, not a wrong model.
</div>

# 15. Project Walkthrough

Everything below exists at `student-placement-prediction/` (the project root). No file name is invented.

| Path | Purpose |
|---|---|
| `README.md`, `DATA_NOTE.md` | overview; which data files are omitted from GitHub and why |
| `requirements.txt` / `requirements-app.txt` | notebook dependencies / exact pinned versions for the app |
| `run_app.bat` | Windows launcher: creates `.venv` once, installs `requirements-app.txt`, starts Streamlit on <http://localhost:8501> |
| `.streamlit/config.toml` | theme; local-only server; hides stack traces |
| `.gitignore` | keeps test-row files and `.venv` out of git |
| `data/raw/` | the three CSVs exactly as downloaded (`train`, `validation`, `test`) |
| `data/processed/` | `modeling_dataset.csv` (9,912), `train.csv`, `validation.csv`, `test.csv`, a flagged combined file |
| `notebooks/01…08` | 01 data understanding · 02 quality and split · 03 EDA · 04 statistics · 05 preprocessing and baselines · 06 calibration and CV · 07 thresholds, explainability, ablation · 08 final test evaluation |
| `src/build_notebook_01…08.py` | scripts that generate those notebooks (training-time code) |
| `src/*_findings.md` | written findings per phase |
| `models/phase4/` | candidate models (`LR_A`, `LR_B`, `DT_*`, `RF_*`, `GB_*`, `DUMMY_prior`, balanced LR) and `model_configs.json` |
| `models/phase7/` | final refits `LR_B_final.joblib` (**the app's model**), `GB_B_final`, `RF_B_final`, and `final_model_configs.json` |
| `app/app.py` | the Streamlit page |
| `app/prediction.py` | loading, validation, prediction, thresholds |
| `app/explainability.py` | coefficient × value contributions |
| `app/ui_helpers.py` | styling, result card, factor bars, disclaimer |
| `app/model_metadata.json` | frozen settings, SHA-256, ranges, levels, thresholds, headline metrics |
| `app/build_app_metadata.py` | builds that metadata from saved results |
| `app/verify_app.py` | the verification script (178 checks) |
| `app/README.md` | app guide and troubleshooting |
| `results/metrics/`, `results/metrics/modeling/`, `results/figures/` | every saved number and chart |
| `report/` | `final_report.*`, `Student_Placement_Predictor_Complete_Project_Guide.*`, `viva_questions.md`, `figures/`, `guide_assets/` and this book |
| `presentation/` | `final_presentation.pptx`, `slide_notes.md` |
| `design/ui_ux/` | UI design briefs and reviews (documents, no code) |
| `theory ppts/` | the course material |
