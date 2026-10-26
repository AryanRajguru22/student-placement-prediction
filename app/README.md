# Student Placement Probability Predictor (Phase 8A)

An interactive Streamlit app that turns ten student characteristics into a model-based **placement probability**, classifications at two thresholds (the project's 0.35 and the conventional 0.50), and an additive explanation of the estimate. It runs locally on your own computer.

It uses the **frozen Phase 7 Logistic Regression B (`LR_B`)**: the Phase 4 pipeline (preprocessing + L2 logistic regression, C = 1, no class weighting, categorical Internships / Projects / Workshops) **refit on train + validation** and evaluated once on the held-out test set in Phase 7. The app trains nothing, tunes nothing and recalibrates nothing.

## Install and run
Requires **Python 3.11 or newer** (developed and verified on Python 3.11.9, Streamlit 1.64.0, scikit-learn 1.9.1; a clean install from `requirements-app.txt` was also verified on Python 3.13.5). scikit-learn must be exactly 1.9.1, the version the frozen model was fitted with; the app refuses to start otherwise.

### Windows: one step
Double-click `run_app.bat` in the project root (`student-placement-prediction/`), or run it from a terminal. On first run it creates a `.venv` folder, installs `requirements-app.txt` (internet needed once) and starts the app; later runs start immediately.

### Any platform: manual
From the project root:

```
python -m venv .venv
.venv\Scripts\activate            (Windows PowerShell / cmd)     source .venv/bin/activate   (macOS / Linux)
pip install -r requirements-app.txt
streamlit run app/app.py
```

Streamlit prints the address when it is ready: open <http://localhost:8501> in a browser (the manual command does not open a tab for you; `run_app.bat` does). `.streamlit/config.toml` runs the server in headless mode, which also skips Streamlit's first-run e-mail prompt. **Stop it with Ctrl+C in the terminal window.** Launch it from the project root so `.streamlit/config.toml` (light theme, local-only server address, usage statistics off, no error details in the UI) is picked up.

`requirements.txt` in the project root is the full research environment (notebooks, statistics); `requirements-app.txt` is the lean, version-pinned set that is enough for the app and `verify_app.py`.

To check the application layer (model loading, predictions, thresholds, explanation maths, input validation, stale-result handling, error handling and a headless end-to-end UI run):

```
python app/verify_app.py
```

### Troubleshooting
| Symptom | Fix |
|---|---|
| `streamlit` is not recognised / `ModuleNotFoundError` | The virtual environment is not active or dependencies are not installed: activate `.venv` and run `pip install -r requirements-app.txt`. |
| "The frozen model was fitted with scikit-learn 1.9.1 but X is installed" | `pip install scikit-learn==1.9.1` (or reinstall from `requirements-app.txt`). |
| "The frozen model file ... is missing or unreadable" / "does not match the frozen Phase 7 fingerprint" | `models/phase7/LR_B_final.joblib` must be present and unmodified (its SHA-256 is recorded in `app/model_metadata.json`). Restore it from your copy of the project; do not retrain a replacement. |
| "The model metadata file ... is missing or unreadable" | Restore `app/model_metadata.json` (it can be regenerated with `python app/build_app_metadata.py` from the full project, which needs `data/processed/train.csv` and `validation.csv` and the aggregate Phase 7 metric files). |
| Port 8501 already in use | `streamlit run app/app.py --server.port 8502` and open <http://localhost:8502>. |
| Need to open the app from another device | The server listens on `localhost` only, on purpose. Override with `--server.address 0.0.0.0` only on a network you trust; the app has no login. |
| "The prediction could not be computed because of an internal error" | Reload the page and retry. Developer diagnostics (exception type and stack frames, never the entered values) are printed in the terminal that runs Streamlit. |

## Using the app
1. Enter all ten fields (or press **Load typical profile** and edit it). **Clear profile** empties every field. Out-of-range numbers are flagged under the field as you type.
2. Press **Predict Placement Probability**. Nothing is predicted until you press it; a profile with missing or out-of-range values is never sent to the model, and each problem is listed.
3. If you change any input after a prediction, the earlier result is **removed** (with a notice) so it can never be mistaken for the new profile; press Predict again.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit page: inputs, buttons, result, explanation, model information |
| `prediction.py` | Loads the frozen artifact (SHA-256 and scikit-learn version verified), validates inputs, builds the one-row dataframe, `predict_proba`, threshold rule, probability bands |
| `explainability.py` | Per-feature log-odds contributions from the frozen coefficients (Phase 6 methodology) |
| `ui_helpers.py` | CSS/HTML fragments (result card with threshold ruler, factor bars, disclaimer) |
| `model_metadata.json` | Supported input ranges, valid levels, thresholds, artifact fingerprint, dataset facts, Phase 7 aggregate metrics (written by `build_app_metadata.py`) |
| `build_app_metadata.py` | One-off generator of the metadata from the frozen model, train/validation data and aggregate Phase 7 metric files |
| `verify_app.py` | Application-level verification (178 checks) |

Artifacts used: `models/phase7/LR_B_final.joblib` (preprocessing and model in one scikit-learn `Pipeline`; nothing is re-implemented in the app) and `app/model_metadata.json`.

## Inputs
All ten fields are required; missing or out-of-range values are rejected with a message and never defaulted or clipped by the app. Supported ranges and levels are those observed in the data the model was developed on (and documented in Phase 1). The SSC, HSC and Aptitude fields accept decimals (the development data contain whole numbers only): a typed value is validated and used exactly as entered, and is rejected only if it lies outside the supported range.

| Field | Supported values |
|---|---|
| CGPA | 6.5 - 9.1 |
| SSC Marks | 55 - 90 |
| HSC Marks | 57 - 88 |
| Aptitude Test Score | 60 - 90 |
| Soft Skills Rating | 3.0 - 4.8 |
| Internships | 0, 1, 2 |
| Projects | 0, 1, 2, 3 (level 0 is rare: 22 of 8,425 development students, so its estimate is unreliable) |
| Workshops / Certifications | 0, 1, 2, 3 |
| Extracurricular Activities | No / Yes |
| Placement Training | No / Yes |

"Load typical profile" fills the form with the development-data medians / most common levels as a starting point (not a recommendation).

## How the probability is generated
The ten values become a one-row dataframe with the exact column names the model was fitted on and go through the frozen pipeline: the five scores are standardised with the training mean and SD, the three count variables and two binary variables are one-hot encoded (reference levels Internships = 1, Projects = 3, Workshops = 0, Extracurricular = No, Placement Training = No), and the logistic regression returns `P(Placed)`. The probability is shown to one decimal place (more decimals only when it lies within rounding distance of a threshold, so the displayed number never appears to be on the other side of a threshold from the label); nothing is rounded internally.

## The two thresholds
The result shows the classification at **both** thresholds side by side; a profile is "Placed" when probability >= the threshold.
- **0.35 (project threshold):** the *selected operating threshold under the F1-oriented criterion* (Phase 6: highest validation F1, chosen by a rule fixed in advance).
- **0.50 (conventional threshold):** the conventional comparison.

Held-out test results (n = 1,487, one-time Phase 7 evaluation; historical values, not recomputed by the app):

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.35 | 0.802 | 0.722 | 0.851 | 0.781 |
| 0.50 | 0.812 | 0.779 | 0.765 | 0.772 |

0.35 finds more placed students (higher recall) at the cost of more false alarms (lower precision). Neither threshold is claimed to be the best: the right cut-off depends on the relative cost of false positives and false negatives. The probability itself is the primary output.

The **probability interpretation bands** (0-20% very low, 20-40% low, 40-60% moderate, 60-80% high, 80-100% very high) are descriptive labels only. They are not confidence intervals and do not coincide with the classification thresholds (a probability of 37% is "Low" but is classified "Placed" at 0.35).

## The explanation ("Key factors influencing this estimate")
For logistic regression the log-odds are exactly additive: `log-odds = intercept + sum(coefficient x transformed value)`. A feature's contribution is the sum of the terms of its transformed columns, measured **relative to a baseline student** (all scores at the training average, all categories at their reference level). The factor chart shows these actual contributions (blue raises, orange lowers the model's estimate; bar length is relative to the largest contribution within the same prediction; contributions smaller than 0.005 in magnitude are listed as at or very near the baseline). "Show the full calculation" lists every step and the odds multiplier, and the contributions reconstruct the model's log-odds to machine precision. This is the Phase 6 methodology, verified against the saved Phase 6 individual explanations.

## Model, data and evaluation
- **Dataset:** Hugging Face [`Krooz/Campus_Recruitment_CSV`](https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV), licence CC0-1.0. 10,000 supplied rows; 9,912 remain after removing surplus exact duplicates and excluding conflicting-label groups; stratified split 6,938 train / 1,487 validation / 1,487 held-out test. The final model was refit on train + validation (8,425 students).
- **Held-out performance (Phase 7):** ROC-AUC 0.886 (95% CI 0.868-0.902), log loss 0.420, Brier 0.134, expected calibration error 0.016, calibration slope 1.05.
- **Model choice:** logistic regression, gradient boosting and random forest were practically tied on the primary metrics; logistic regression was retained for its simplicity and interpretability, not because it was uniquely better.

## Privacy
Entered profiles are held only in the memory of the local Streamlit session while the page is open (and in your browser's page). The app does not write them to disk, log them, use a database or accounts, or call any external service from its own code, and Streamlit usage statistics are switched off. Installing dependencies needs the internet; running the app does not need it (the app's own code makes no network requests, and the page loads no external fonts or scripts). Developer diagnostics in the terminal contain no entered values. The server listens on `localhost` only.

## Limitations
- The model describes **associations** learned from one observational dataset; contributions are not causal effects, and changing a field does not say what would happen if a real student changed it.
- The observed structure is consistent with a rule-based or constructed dataset, although this is not established by the dataset source. Strong performance on this dataset does not establish equivalent performance on real-world university placement populations, other campuses or other years.
- A probability describes a group of similar students, not a certainty for an individual (about 20% of held-out students were misclassified).
- Reported Phase 7 held-out performance comes from the one-time evaluation of the frozen model; the app does not reproduce or access it.
- **Test-set protection:** no code in the app reads the held-out test split or any per-student test prediction file (`verify_app.py` scans the app source and runs under an audit hook that raises if such a file is opened). The test data were not used for model development, selection, tuning, calibration or threshold selection (earlier phases' descriptive access and the single Phase 7 evaluation are disclosed in the project README and report).
- The app runs locally; it is not deployed anywhere.
