# Current App Audit — Student Placement Probability Predictor

**Scope:** audit only. Nothing in the application, model, data, notebooks, report or `requirements.txt` was modified.
**Method:** every statement below was read from source. Line references are to the files as they exist today (`student-placement-prediction/app/…`). Where something could not be verified from source it is marked **(to verify)**.
**Not done on purpose:** the app was not launched, no packages were installed, the held-out test data was not opened, no metric was recomputed. The only code executed was a read-only call of the app's own `load_bundle` / `predict` / `explain` on one fixed profile (the "typical profile") to obtain illustrative numbers for the design prompt (see §9). No screenshots of the current UI exist in the repository (only report figures under `report/figures/`), so visual observations are derived from the CSS/HTML in `ui_helpers.py`, not from a rendered page.

---

## 0. Current confirmed facts (added 2026-10-17)

§1–§12 below are the **historical audit** of 2026-10-14 (source read, nothing executed except one read-only call of the app's own functions). They are unchanged except where a line is marked *(updated 2026-10-17)*. The facts in this section were **confirmed on 2026-10-17 by execution** and are kept apart from the history:

| Fact | Evidence |
|---|---|
| The application source is still the Phase 8A implementation; no Stitch design is integrated | `app/*.py` and `.streamlit/config.toml` last modified 2026-10-03 (≤ 01:04); `ui_helpers.py` still holds the inline CSS and `result_card`/`factor_bars`; no HTML/CSS/JS/font assets exist in the project outside `report/final_report.html` |
| **The existing verification suite passes 122/122, twice** | Run 1: `python -I -B -X utf8 <guard wrapper> app/verify_app.py`; run 2: `python -B -X utf8 app/verify_app.py`. Both from the project root, both exit code 0, both 122 `PASS` lines and the final line `ALL 122 CHECKS PASSED`, outputs identical apart from timings. `sys.flags.optimize` = 0 (asserts live) |
| Environment | Python 3.11.9 (not a virtualenv), streamlit 1.64.0, scikit-learn 1.9.1 (equals the version the model was fitted with), pandas 3.0.5, numpy 2.4.2, scipy 1.17.0, joblib 1.6.0. Not a Git repository |
| The held-out test files were not opened | The guard hook (independent of the script's own hook) logged every project file read: only the app source, `.streamlit/config.toml`, `data/processed/validation.csv`, `models/phase4/LR_B.joblib`, `models/phase7/LR_B_final.joblib`, `07_individual_explanations.csv`, `07_threshold_selection.json`. It blocked nothing and permitted no write inside the project |
| Nothing in the project changed | Before/after snapshots of all 206 files: no file added, removed or modified (size and mtime); the 47 files under `app/`, `models/`, `.streamlit/`, `design/`, `src/`, `requirements.txt` and `README.md` were also SHA-256 compared (0 mismatches). `data/`, `results/`, `notebooks/`, `report/`, `presentation/` were compared by size and mtime only |
| The 122 checks cover | 5 source scans for test-file references · model load and SHA-256 · LR_B specification · 0.35/0.50 thresholds · 7 manual profiles × 10 checks (parity with the frozen model to < 1e-12, thresholds, bands, explanation reconstruction) · 5 threshold-boundary checks · 6 Phase 6 explanation cross-checks on validation rows · 19 input-validation cases · 13 headless Streamlit `AppTest` checks · 1 audit check |
| Documentation drift found | `app/README.md` still says "118+ checks" and §1 below said "100+"; the root `README.md` (line 176) says 122, which is correct. Not edited here (outside this task's scope) |
| Check weakness (not a failure) | The UI checks "out-of-range aptitude" and "out-of-range HSC" leave other fields empty, so the error shown also lists the missing fields; they pass on the presence of "outside the range" |

**What this establishes.** The pre-redesign baseline: the current Phase 8A app, its frozen model and its explanation layer behave as documented. **What it does not establish:** nothing about any new design — no redesign has been implemented, so none has been verified. Any change to widget types, labels, markers or copy strings listed in §8.1 must update `verify_app.py` in the same change and re-pass the suite.

---

## 1. Architecture and files

| File | Role | Notes |
|---|---|---|
| `app/app.py` (151 lines) | Single Streamlit script: page config, form, validation call, result, model info | `layout="centered"` (l.17) but CSS overrides `.block-container` to `max-width: 1120px` (`ui_helpers.py` l.16) |
| `app/prediction.py` | Loads frozen artifact, **SHA-256 verified** against metadata (l.58-76); `validate_inputs` (l.83-118); `to_frame`; `classify` (`p >= threshold`, l.137-139); `predict` (l.142-150); probability bands (l.33-34) | Authoritative; must not change |
| `app/explainability.py` | Additive per-feature log-odds contributions (l.30-59) | Authoritative; must not change |
| `app/ui_helpers.py` | CSS (l.14-48), header, `result_card`, `factor_bars`, `calculation_table`, `disclaimer` | The only place where presentation lives; the natural redesign target |
| `app/model_metadata.json` | Ranges, levels, defaults, thresholds, fingerprint, Phase 7 aggregate metrics | Read-only data contract |
| `app/verify_app.py` (208 lines) | 122 checks incl. a headless `AppTest` run (l.159-205) *(updated 2026-10-17: was "100+"; 122/122 passed twice, §0)* | **Locks parts of the UI — see §8** |
| `.streamlit/config.toml` | `base = "light"`, primary `#0072B2`, bg `#ffffff`, secondary bg `#f4f6f8`, text `#1d2733`, usage stats off | Only 5 theme keys used |

Dependencies: `streamlit>=1.50` (`requirements.txt`); installed version in this environment is **1.64.0**. No database, auth, API or network call exists. The model is cached with `@st.cache_resource` (`app.py` l.22-24).

## 2. Current screen structure and workflow

One page, one script, linear top-to-bottom (`app.py`):

1. **Header** (`header()`, `ui_helpers.py` l.55-58): title "Student Placement Probability Predictor", one-line subtitle, horizontal rule. Rendered as `<div>`s, not headings.
2. **Student profile** (l.69-93): section title, caption "All ten fields are required…", then an `st.form` with 3 equal columns:
   - *Academic profile*: CGPA, SSC Marks, HSC Marks
   - *Technical profile*: Internships, Projects, Workshops / Certifications, Aptitude Test Score
   - *Employability profile*: Soft Skills Rating, Extracurricular Activities, Placement Training
   - Two buttons: primary **"Predict Placement Probability"** and **"Load typical profile"** (both `st.form_submit_button`).
3. **Error area** (l.98-107): shown below the form only after a submit with problems.
4. **Result** (l.109-126, only after a successful submit): hero card + facts column → rare-level warning(s) → "Key factors influencing this estimate" (custom HTML bars) → expander "Show the full calculation" → expander "Inputs used for this estimate" → disclaimer.
5. **Model information** (l.128-140): two Markdown bullet columns. Always visible, at the bottom, whether or not a prediction exists.
6. **Expander "About the threshold and the limits of this tool"** (l.141-150): collapsed by default; holds the threshold rationale, band caveat, association-not-causation, constructed-dataset caveat, "roughly 1 in 5 misclassified".

**Initial state:** all ten fields empty (`value=None` / `index=None`); no result. The only guidance before the first prediction is the caption and "Load typical profile".

## 3. Verified input specification

Source of truth: `model_metadata.json` (`ranges` l.48-84, `levels` l.85-151), `prediction.py` (`FEATURES`, `LABELS`, `validate_inputs`), widget code `app.py` l.36-57, 73-88.

### 3.1 Continuous (5) — `st.number_input`, `value=None`, **no `min_value`/`max_value` on purpose** (comment l.41-42)

| Field (label) | Key | Allowed range (min–max) | Step / format | Placeholder (median) | "Load typical" value | Training mean |
|---|---|---|---|---|---|---|
| CGPA | `CGPA` | 6.5 – 9.1 | 0.1 / `%.1f` | e.g. 7.7 | 7.7 | 7.697 |
| SSC Marks | `SSC_Marks` | 55 – 90 | 1 / `%d` | e.g. 70 | 70 | 69.14 |
| HSC Marks | `HSC_Marks` | 57 – 88 | 1 / `%d` | e.g. 73 | 73 | 74.44 |
| Aptitude Test Score | `AptitudeTestScore` | 60 – 90 | 1 / `%d` | e.g. 80 | 80 | 79.32 |
| Soft Skills Rating | `SoftSkillsRating` | 3.0 – 4.8 | 0.1 / `%.1f` | e.g. 4.4 | 4.4 | 4.320 |

- `integer_valued` is true for SSC, HSC, Aptitude (metadata). The widget uses an integer step/format for them, but **`validate_inputs` only checks "is a finite number" and `min ≤ v ≤ max` for continuous fields (`prediction.py` l.94-102); it does not reject non-integers.** The design must not invent an integer-only rule; the widget-level behaviour for a typed decimal in a `%d` field is **(to verify)**.
- Ranges are "the range observed in the data the model was developed on" (`range_help`, l.36-37) — they are support limits, not quality judgements.
- Each field currently shows the range twice: in the `help=` tooltip and in a caption beneath (l.45-46).

### 3.2 Count fields (3) — `st.selectbox`, `index=None`, placeholder "Select a level"

| Field | Key | Allowed levels | n per level (of 8,425) | Reference level in model | "Load typical" default |
|---|---|---|---|---|---|
| Internships | `Internships` | 0, 1, 2 | 1,673 / 4,718 / 2,034 | 1 | **1** |
| Projects | `Projects` | 0, 1, 2, 3 | **22** / 3,014 / 2,160 / 3,229 | 3 | **2** |
| Workshops / Certifications | `Workshops/Certifications` | 0, 1, 2, 3 | 3,160 / 2,275 / 2,744 / 246 | 0 | **1** |

- `validate_inputs` requires an integer value contained in the allowed list (l.103-108).
- **Rare-level rule:** `rare_level_threshold_n = 50`. Only **Projects = 0 (22 students)** is below it. Workshops = 3 (246) is *not* flagged. The caption is shown permanently under the Projects selectbox (l.53-56) and a `st.warning` appears in the result when Projects = 0 is used (l.114-117).
- The metadata builder computes the count default as the **median** of the development data (`build_app_metadata.py` l.56), so for counts it is *not* the most common level (the most common Projects level is 3 and the most common Workshops level is 0). The page caption says "medians / most common levels" (`app.py` l.70) — slightly imprecise for counts; accurate copy: "median for scores and counts, most common answer for Yes/No".

### 3.3 Binary fields (2) — `st.radio(["No","Yes"], index=None, horizontal=True)`

| Field | Key | Levels | n (No / Yes) | Reference level | "Load typical" default |
|---|---|---|---|---|---|
| Extracurricular Activities | `ExtracurricularActivities` | No, Yes | 3,510 / 4,915 | No | **Yes** (mode) |
| Placement Training | `PlacementTraining` | No, Yes | 2,276 / 6,149 | No | **Yes** (mode) |

### 3.4 Validation rules (all in `validate_inputs`)

- Every one of the ten fields is required; missing/NaN/blank → `"{Label}: a value is required."`
- Non-numeric/non-finite → `"{Label}: '{v}' is not a valid number."`
- Out of range → `"{Label}: {v} is outside the range the model was developed on ({lo:g} to {hi:g}); no prediction is made for out-of-range values."`
- Invalid level → `"…is not a valid level (allowed: …)"` / `"…is not a valid choice (allowed: …)"`
- Nothing is clipped, rounded or defaulted silently. Errors are collected and shown as **one** `st.error` with a bullet list (l.98-101); the previous result is discarded (`session_state.pop("last")`).

## 4. How the prediction is calculated and displayed

1. Ten validated values → one-row DataFrame with the exact fitted column names; counts cast to int (`to_frame`, l.121-127).
2. Frozen pipeline (`models/phase7/LR_B_final.joblib`): five scores standardised with training mean/SD; three counts + two binaries one-hot encoded (reference levels above); L2 logistic regression (C = 1, no class weighting). `predict_proba` → P(Placed); `decision_function` → log-odds (`predict`, l.142-150). Probabilities are raw — no recalibration.
3. Display rounding only: `pct()` = one decimal (`ui_helpers.py` l.61-62).

### 4.1 Thresholds (verified)

- `thresholds.operating = 0.35` — "selected operating threshold under the F1-oriented criterion (Phase 6, predeclared validation-F1 rule); not claimed to be the best threshold".
- `thresholds.reference = 0.50` — conventional.
- Rule: **Placed iff `p >= threshold`** (inclusive), applied independently to each threshold (`classify`).
- Result card shows the **0.35** classification as the primary "Model classification" pill, and the **0.50** classification as a secondary "Conventional 0.50 threshold" row captioned "threshold choice changes the label, not the probability" (`result_card`, `ui_helpers.py` l.73-78).
- Both thresholds are also drawn as ticks on the gauge with labels "0.35 operating" / "0.50 reference" (l.68-70).
- The two labels **disagree** for any 0.35 ≤ p < 0.50. The current UI never says so explicitly.

### 4.2 Probability bands (display only)

`<0.2` Very low · `<0.4` Low · `<0.6` Moderate · `<0.8` High · else Very high ("estimated probability"). Shown as "Probability interpretation band — descriptive band, not a confidence interval". Because the bands are independent of the thresholds, a probability of 37% reads **"Low"** while being classified **"Placed"** at 0.35 (documented in `app/README.md`; the UI only hints at it in a caption).

## 5. How the explanation is generated and presented

- `explain()` (`explainability.py` l.30-59): `terms = transformed_x × coefficient`; each transformed column is summed back to its raw feature. Output per feature: log-odds contribution, odds multiplier `exp(c)`, value text with its baseline ("training average 79.3" for scores; "reference level 3" for categories). `intercept + Σ contributions` reconstructs the model log-odds exactly.
- The baseline ("reference student") is every score at the training mean and every category at its reference level. **Contributions are relative to that baseline**; they are associations, not causal effects.
- `factor_bars()` (`ui_helpers.py` l.97-115): three groups — "Factors that raise the model's estimate" (c ≥ 1e-12, blue `#0072B2`), "Factors that lower the model's estimate" (c ≤ −1e-12, orange `#D55E00`), and a text-only "At the baseline (no contribution)" list. Diverging bars around a centre line, bar length = |c| / max|c| **within this prediction** (so bar lengths are not comparable across predictions — stated in a footnote). Numeric value shown in **log-odds** with sign (`+0.97`).
- The odds multiplier, the intercept and the total log-odds appear only inside the collapsed "Show the full calculation" table (`calculation_table`, l.118-126; the last row stores the probability string in the "Input" column).
- "Inputs used for this estimate" is an expander containing `st.table` with ten **columns** (one per field) and one row (`app.py` l.124).

## 6. Model information currently available

Displayed (l.128-140): model name, L2 / C = 1 / no class weighting, training data (8,425 students, train + validation), operating and reference thresholds, "raw model output, no recalibration", Phase 7 held-out evaluation on **1,487** students — ROC-AUC 0.886 (95% CI 0.868–0.902), log loss 0.420, Brier 0.134, **calibration slope 1.05** — and the first 16 characters of the artifact SHA-256.

Available in `model_metadata.json` but **not displayed**: **ECE (0.0164)**, calibration intercept (0.041), CIs for log loss (0.393–0.449) and Brier (0.124–0.145), `sklearn_version_at_fit`, per-level counts, reference levels.
Documented in `README.md` / `src/phase7_findings.md` but **not in the metadata or the UI**: test classification table at 0.50 and 0.35 (accuracy 0.812 / 0.802; precision 0.779 / 0.722; recall 0.765 / 0.851; F1 0.772 / 0.781; specificity 0.846 / 0.766; TP/TN/FP/FN 473/735/134/145 and 526/666/203/92), dataset source (Hugging Face `Krooz/Campus_Recruitment_CSV`, CC0-1.0), and the split/evaluation protocol. Surfacing any of these in a redesign is new *static content*, not new computation, but needs to be added to the metadata (or a static text) in the implementation phase with its source cited.

## 7. Existing components and behaviour worth retaining

- Frozen-artifact integrity check with a plain-language failure (`ModelLoadError` → `st.error` + `st.stop`).
- **Nothing assumed:** all fields start empty; nothing clipped or silently defaulted; out-of-range values rejected, never coerced.
- "Load typical profile" as a clearly labelled starting point "not a recommendation".
- Explicit separation of probability / thresholded label / descriptive band, and the "threshold choice changes the label, not the probability" wording.
- Additive explanation grouped into raises / lowers / baseline, with the "associations, not causal effects" footnote, and the full calculation table (strong credibility feature).
- Rare-level warning for Projects = 0.
- Colour-blind-safe blue/orange pair (Okabe–Ito) — and, importantly, **meaning is not carried by colour alone**: sign, section headings and numeric values also encode direction.
- Honest limitations text (association not causation, constructed-dataset caveat, "a probability describes a group of similar students").
- No data leaves the app; no personal data; no persistence.

## 8. Technical constraints (hard and soft)

### 8.1 Hard constraints imposed by `verify_app.py` section 6 (headless `AppTest`)

A redesign that ignores these will fail verification (or must update `verify_app.py` deliberately, as an approved change):

1. **Widget types and keys:** continuous → `number_input`, counts → `selectbox`, binary → `radio`; every widget `key` equals the feature name (e.g. `"Workshops/Certifications"`).
2. **Exact button labels:** `"Predict Placement Probability"` and `"Load typical profile"`.
3. **Marker markup:** the probability is found by searching markdown output for the exact string `<div class="hero-prob">{p:.1f}%</div>`.
4. **Copy strings that must appear in markdown:** `Model classification`, `Conventional 0.50 threshold`, `Factors that raise the model`, `not a guarantee of placement`; error text containing `outside the range`; `a value is required` ×10 in a single error.
5. **Initial state:** all ten fields empty (`value is None`), no hero before the first click.
6. **Behaviour:** CGPA 9.5, Aptitude 95, HSC 40 → an error and **no** hero; "Load typical profile" sets the documented values; the typical profile yields 0.35 ≤ p < 0.50.
7. **Test-set protection:** the verification audit hook raises if any test file is opened; the redesign must not read data files other than the model and metadata.

*(Update 2026-10-17: these locks were exercised and passed on the current source — §0. The resolved design decisions D1 (drop `st.form`) and D2 (counts as radio groups) are specified in `design_brief.md` §13; D2 changes lock 1 and therefore needs the `verify_app.py` section 6 update in the same implementation change.)*

*Consequence for the design:* sliders are unacceptable (they clamp, which contradicts "nothing is clipped" and the out-of-range tests). Replacing `selectbox` with a horizontal `radio`/`segmented_control` for counts is a UX improvement but **requires updating `verify_app.py`** — it must be listed as an explicit approval item, not slipped in.

### 8.2 Streamlit constraints

- **`st.form` freezes widget values until a submit.** Consequences: no live validation, no way to detect "inputs edited since the last estimate", and the stored result (`session_state["last"]`) stays on screen. Inline per-field errors are possible with a form only via a store-errors-in-session-state-and-rerun pattern; without a form they are native (each widget change reruns the script).
- No JavaScript in `st.markdown` / `st.html`; custom HTML cannot take focus, cannot scroll the page to the result, and has no behaviour. `unsafe_allow_html=True` is already used; sanitisation of `role` / `aria-*` attributes and `<style>` is **(to verify in a prototype)**.
- No `position: sticky` or fixed regions without brittle CSS targeting Streamlit's internal DOM.
- Columns collapse to a vertical stack at narrow widths (Streamlit built-in; exact breakpoint **(to verify)**). Nested-column depth limits **(to verify)** for a form-inside-column layout.
- Theme: `.streamlit/config.toml` supports many more keys than the 5 currently used. Confirmed present in **Streamlit 1.64.0**: `font`, `headingFont`, `fontFaces`, `baseRadius`, `buttonRadius`, `borderColor`, `showWidgetBorder`, `baseFontSize`, `chart*Colors`, `metricValueFontSize`, light/dark sections. `requirements.txt` says `>=1.50`, so **the implementation phase must confirm each key exists at the pinned minimum or raise the pin** (not changed here). Web fonts from a CDN would add an outbound request; a system font stack or a self-hosted font avoids it.
- Available widgets on 1.64.0 (confirmed): `segmented_control`, `pills`, `badge`, `metric`, `tabs`, `dialog`, `popover`, `status`, `navigation`, `html`, `toast`, `progress`, `expander`.

## 9. Illustrative reference values (read-only computation with the app's own functions)

For the "typical profile" (CGPA 7.7, SSC 70, HSC 73, Aptitude 80, Soft Skills 4.4, Internships 1, Projects 2, Workshops 1, Extracurricular Yes, Placement Training Yes):

- P(Placed) = **0.3730** → "37.3%", band **Low estimated probability**
- Classified **Placed at 0.35**, **Not Placed at 0.50** → the thresholds disagree and band/classification look contradictory — the ideal stress-test for the result design.
- Intercept −1.661; total log-odds −0.520.
- Raises: Placement Training +0.97 (×2.65), Extracurricular +0.72 (×2.06), Soft Skills +0.05, Aptitude +0.04, SSC +0.02, CGPA +0.00 (0.0009).
- Lowers: Workshops/Certifications −0.32 (×0.72), Projects −0.31 (×0.73), HSC −0.03.
- At baseline: Internships = 1 (reference level).
- Check: intercept + Σ contributions = −0.5195 = model log-odds.

These are used only as **clearly labelled example data** for the design prompt.

## 10. Prioritised usability problems

**P1 — affects correctness of understanding or trust**

1. **Stale result after editing inputs.** Because inputs sit in an `st.form`, editing a field (or pressing "Load typical profile") does not clear or flag `session_state["last"]`; the old probability stays next to new inputs (`app.py` l.95-126). Only the collapsed "Inputs used" expander reveals what was actually scored.
2. **The two-threshold story is under-explained.** Disagreement between 0.35 and 0.50 is never called out; band "Low" vs "Placed" is visually contradictory (§4.2); "operating", "reference" and "Phase 6/7" are project jargon on the main screen (`result_card`, l.69-77; `app.py` l.129, 143).
3. **Disclaimer and limitations are far from the number.** The only in-card caveat is one 0.9 rem line; the full disclaimer is at the very bottom of the result; the substantive limitations are in a collapsed expander under "Model information" (l.141-150).
4. **Contributions are in log-odds** (`+0.97`) with no plain-language reading; the odds multiplier that most users can interpret is hidden in a collapsed table.

**P2 — usability friction**

5. **Result appears below a three-column form**, far from the button on desktop and often below the fold; no scroll or focus move after submit.
6. **Errors are one combined box after the form**, not beside the offending field; on mobile the user may not see it (l.98-101).
7. **Redundant range text** (tooltip + caption) on every numeric field; ranges are not linked to the user's current entry (no "in range" state).
8. **Model information is at the bottom and always on**, before any prediction, so first-time visitors get no explanation of what the tool is, what it's trained on, or how to read the result. No dataset source stated.
9. **Selectboxes for 3-4 option fields** hide the options behind a click; a visible option group would be faster and clearer.
10. **"Inputs used" table is 10 columns × 1 row** — horizontally cramped on narrow screens (l.124).
11. "Load typical profile" is a form-submit button, styled like the primary action's sibling, with no confirmation that it loaded something.
12. ECE (a headline project metric) is not shown; calibration slope is, without explanation. Metrics are listed with no plain-language gloss.

**P3 — visual polish**

13. Flat generic Streamlit look: system sans, default widgets, a single card; hierarchy relies on a 4.6 rem number and thin rules.
14. Brand/identity absent: no icon, no page icon, no visual signature; "Student profile" / "Result" / "Model information" are plain text titles.
15. No dark mode (theme pinned to light) — acceptable, but unused tokens.

## 11. Accessibility and responsive findings (measured contrast, WCAG 2.x relative luminance)

| Item | Finding |
|---|---|
| Orange `#D55E00` text on white (`.fb-val` negative values, 0.92 rem bold ≈ 14.7 px) | **3.87:1 — below AA (4.5:1)** for normal text; 3.54:1 on the `#f3f5f7` track |
| Blue `#0072B2` on white / `#fbfcfd` | 5.19:1 — passes AA |
| Muted `#5b6672` on white / hero / panel | 5.85 / 5.70 / 5.40 — passes |
| Pills: `#00537f` on `#e3f0f9`; `#8f3b00` on `#fbeadf` | 7.11 / 6.42 — pass |
| Bar centre line `#9aa5b1` on white | **2.50:1 — below the 3:1 needed for non-text graphics** |
| Gauge (`.gauge`, ticks) | Pure `<div>`s: **no text alternative, no role/aria**; the probability is in the adjacent big number only |
| Threshold tick labels (`.tick-lab`) | `position:absolute; white-space:nowrap; transform:translateX(-50%)` at 35% and 50% — only 15% of the card apart; "0.35 operating" and "0.50 reference" are ~75-90 px wide, so they **overlap on narrow (phone) widths** |
| Factor bars | Custom HTML; not focusable, no per-row text alternative beyond the visible text; `grid-template-columns: minmax(150px, 34%) 1fr 74px` leaves a very short bar track below ~380 px |
| Headings | Title and section titles are `<div>`s, so no heading structure for screen-reader navigation |
| Hero number | Fixed 4.6 rem (`.hero-prob`) — fits a 2-column card on desktop, large and unconstrained on phones |
| Credit | **Colour is not the only carrier of meaning** (sign, section headings, text values); the blue/orange pair is CVD-safe; visible text labels on every field; native Streamlit widgets give keyboard operation for inputs and buttons |
| Not verified | Touch-target size of Streamlit's number-input steppers on mobile; focus-ring visibility on custom theme; screen-reader announcements of the result (no `aria-live` region exists) |
| Layout | Three columns collapse to one at narrow widths (Streamlit default) — the form becomes a long single column of 10 fields; result then far below |

## 12. Summary

The existing app is **correct, defensible and honest** — its weaknesses are presentational and structural, not mathematical. A redesign should leave `prediction.py`, `explainability.py`, the frozen model and the metadata contract untouched and concentrate on: (1) an explicit, plain-language treatment of probability vs the two classifications; (2) result placement and staleness; (3) per-field, in-context validation; (4) an accessible, responsive result visual; (5) progressive disclosure of methodology and limitations near the number; (6) a coherent visual system. The `verify_app.py` coupling in §8.1 is the main regression risk and the main source of "approve or update" decisions.

*Update 2026-10-17:* the "approve or update" decisions are now resolved in `design_brief.md` v2 §13. The redesign direction (light editorial workspace, horizontal threshold ruler replacing the gauge, navy meter fill) is specified there; this audit remains the evidence base for the *current* app and is not revised. The baseline in §0 is the state any implementation must preserve or deliberately update.
