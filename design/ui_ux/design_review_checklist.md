# Design Review Checklist — Stitch Output

**Version 2 (2026-10-17).** Accept or reject every generated screen against `design_brief.md` v2 (canonical) and `stitch_prompt.md` v2. Each item has a **pass criterion** that can be checked without opinion, a **method**, and a **severity**. Any failing **B** (blocker) item means the screen cannot be approved. **M** (major) items must be fixed or explicitly waived in writing in §11. Mark Pass / Fail / N-A and note the screen.

**Methods.** **V** = look at the screenshot (measure with the frame's pixel size). **H** = inspect the Stitch HTML/CSS (search the text; the HTML is a visual reference, so its Tailwind/Google-Fonts links and incidental scripts are *not* defects — behaviours that Streamlit cannot build are). **C** = compute from hex values or measured sizes. **T** = read the text and compare it to the quoted string. **S** = Streamlit-feasibility judgement against brief §12/§14.

Reviewer: ____________  Date: ____________  Stitch project / screen: ____________

Review order: §1 → §2 → §3 → §5 → §4 → §6–§9 → §10 (screen by screen; do not generate the next screen until the current one passes §1–§3 and §5).

---

## 1. Model truth and fidelity

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 1.1 | Exactly ten input controls, labelled exactly: CGPA, SSC Marks, HSC Marks, Internships, Projects, Workshops / Certifications, Aptitude Test Score, Soft Skills Rating, Extracurricular Activities, Placement Training. No eleventh control; none of Backlogs, Hackathons, Stream/Department, Gender, Name | V, T | B | |
| 1.2 | Visible hints/options match exactly: CGPA 6.5 to 9.1; SSC 55 to 90; HSC 57 to 88; Aptitude 60 to 90; Soft Skills 3.0 to 4.8; Internships 0/1/2; Projects 0/1/2/3; Workshops 0/1/2/3; Extracurricular No/Yes; Placement Training No/Yes. Hint format "Supported range: 6.5 to 9.1" | T | B | |
| 1.3 | On the empty, validation and mobile-form screens no field has a selected/typed value except the ones the screen's brief says are filled; any screen showing the typical profile carries the tag "Example profile" | V, T | B | |
| 1.4 | No `<input type="range">`, no `role="slider"`, no `min=`/`max=` attribute on number inputs, no script that rewrites a typed value (e.g. clamp/`adjustValue`-style) | H | B | |
| 1.5 | Both thresholds appear as 0.35 and 0.50 with "Placed when probability is 0.35 or higher" / "…0.50 or higher" | T | B | |
| 1.6 | The probability is not inside a chip or card of a classification; the sentence "The threshold changes the label, not the probability." is present on the result screen | V, T | B | |
| 1.7 | Searching all text for `optimal`, `best`, `recommended`, `default` finds none applied to a threshold; 0.35 is described as "chosen … by a rule fixed in advance (highest validation F1)" and "Not claimed to be the best threshold" | T | B | |
| 1.8 | Evaluation values exactly: ROC-AUC 0.886 (95% CI 0.868 to 0.902), log loss 0.420, Brier 0.134, ECE 0.016; **exactly four** metrics; title exactly "Results from this project's final test evaluation (1,487 students, evaluated once)"; ROC-AUC never called accuracy | T | B | |
| 1.9 | If the threshold table is shown: accuracy 0.802/0.812, precision 0.722/0.779, recall 0.851/0.765, F1 0.781/0.772, specificity 0.766/0.846 (0.35 / 0.50) | T | B | |
| 1.10 | Example-profile result is exactly: 37.3 %, band Low, Placed at 0.35, Not Placed at 0.50; contributions Placement Training +0.97 ×2.65, Extracurricular +0.72 ×2.06, Soft Skills +0.05 ×1.05, Aptitude +0.04 ×1.04, SSC +0.02 ×1.02, CGPA +0.00 ×1.00, Workshops −0.32 ×0.72, Projects −0.31 ×0.73, HSC −0.03 ×0.97; Internships in the baseline list; **all ten features appear exactly once** | T | B | |
| 1.11 | No per-prediction interval: the words `confidence`, `credible`, `interval`, `CI`, `±` occur only in the ROC-AUC "95% CI" line and in negations ("not a confidence interval") | T, H | B | |
| 1.12 | No advice or causal wording: none of `improve`, `boost`, `increase your chances`, `if you`, `you should`, `will be placed`, `likely`, `at-risk`, `driver`, `impact`, `importance`, `attribution`, `SHAP`; contributions described as associations relative to a baseline | T, H | B | |
| 1.13 | Every number on the screen is in brief §11.1, a layout number (step numbers, 0%/100% scale ends, tab counts) or a marked placeholder ("XX.X"). Method: list all numeric tokens and tick each against §11.1 | T | B | |
| 1.14 | Dataset line "Hugging Face Krooz/Campus_Recruitment_CSV (CC0-1.0)" appears on the Method or Limitations screen | T | M | |
| 1.15 | Within one viewport height of the hero number: "not a guarantee" (hero subline) and the disclaimer containing "not a guarantee of placement" and "causal assessment" | V, T | B | |
| 1.16 | The example (band Low, classified Placed at 0.35) shows the sentence "A 'Low' band can still be 'Placed' at the 0.35 threshold…" | T | B | |
| 1.17 | Threshold-disagreement notice present on the example result; the system sheet shows the agreement variant "Both thresholds give the same label." | T | M | |
| 1.18 | Badges read exactly "Estimate, not a guarantee", "Frozen model · Logistic regression", "Nothing is saved" | T | M | |
| 1.19 | Rare-level notice for Projects = 0 reads with "22 of 8,425 students"; **no** notice for Workshops = 3 | T | M | |

## 2. Content that must be absent

| # | Pass criterion (search result must be empty) | Method | Sev | Result |
|---|---|---|---|---|
| 2.1 | Name / roll number / e-mail / photo / avatar / profile icon / college fields | V, H | B | |
| 2.2 | Login, account, saved profile, history, export, share | V, T | B | |
| 2.3 | Any navigation: top nav, side nav, bottom tab bar, hamburger/kebab menu, breadcrumb | V, H | B | |
| 2.4 | KPI/metric tiles with progress bars, trend arrows or sparklines; dashboard rows of tiles | V | B | |
| 2.5 | Status strips and system telemetry: version, seed, latency, trace/engine IDs, "online", pulsing status dots, hash badge in the header | V, T | B | |
| 2.6 | Peer/cohort comparison, percentiles, rankings, "similar students", benchmark panels | V, T | B | |
| 2.7 | Chatbot, assistant, notifications, gamification, confetti | V | M | |
| 2.8 | Tips, recommended actions, what-if sliders, simulators, "Live test bench" | V, T | B | |
| 2.9 | Radial gauge, dial, arc, dotted ring: no SVG `<circle>`/`<path>` arc used as the probability graphic, no `conic-gradient`/`radial-gradient` | V, H | B | |
| 2.10 | Dark theme: page and card backgrounds all luminance > 0.85 | C | B | |
| 2.11 | ✓/✕ icons, green or red used for Placed / Not Placed; celebratory or alarming treatment | V | B | |
| 2.12 | Stock imagery, mascots, graduation-cap clichés, marketing hero | V | M | |
| 2.13 | A second probability graphic (e.g. a gauge plus a bar for the same number) | V | M | |

## 3. The threshold ruler (the signature element)

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 3.1 | One horizontal track at least 14 px high spanning the result card's content width; 1 px outline `#8A94A3` (≥ 3:1 on white); fill colour of the track `#EEF3F8` | V, C | B | |
| 3.2 | Fill colour is navy `#0F4C75` (not blue `#0072B2`, not cyan, no gradient) | H, C | B | |
| 3.3 | Fill length: (fill width ÷ track width) = 0.373 ± 0.01 on the example | V | B | |
| 3.4 | Two markers at 35 % ± 1 % and 50 % ± 1 % of the track width; 35 % is a **filled** diamond, 50 % a **hollow** diamond (distinguishable in greyscale); numerals "0.35" and "0.50" beside them | V | B | |
| 3.5 | Markers remain visible over the navy fill: a white halo ≥ 1 px around the 2 px ink tick | V, H | B | |
| 3.6 | The 35–50 % span (±1 %) is hatched **and** carries a bracket with the text "Labels differ in this region"; present even when p is outside it (empty state) | V, T | B | |
| 3.7 | Legend row contains "0.35 (35%) project threshold", "0.50 (50%) conventional threshold" and the hatch key "labels differ here" | T | B | |
| 3.8 | Only the numerals "0.35" and "0.50" sit at the markers; scale labels are only "0%" and "100%"; **no text at the fill end** | V | B | |
| 3.9 | No two text boxes on the ruler overlap at 1440, 390 and 320 px (compare bounding boxes) | V | B | |
| 3.10 | The text equivalent sentence beneath matches the case: "Estimated probability 37.3%: above the 0.35 threshold, below the 0.50 threshold." | T | B | |
| 3.11 | Band tag is neutral: border `#8A94A3`/text `#475467`; no blue/orange fill, border or text | V, C | B | |
| 3.12 | Two classification chips have equal size, padding and font weight; glyphs ■ (Placed) and ● (Not Placed) plus text; Placed uses `#0A5C7A` on `#E6F2F8`, Not Placed `#B54708` on `#FFF4E5` | V, C | B | |
| 3.13 | Empty-state ruler shows markers and hatched region and **no fill and no number** | V | B | |
| 3.14 | Header scale rule, empty-state ruler and bar centre-line ticks are the only uses of the ruler motif | V | M | |

## 4. Visual hierarchy and style

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 4.1 | Hero numeral is the largest text on the screen: ≥ 72 px on the 1440 frame (≥ 2× the 32 px H1) and ≥ 52 px on the 390 frame (larger than the 28 px H1 and every other text) | V, H | B | |
| 4.2 | Top-to-bottom order in the result card: eyebrow → hero → ruler/legend/sentence → disclaimer → how-to-read note → two classification cards → band row; explanation tabs follow | V | B | |
| 4.3 | Page background `#F6F8FB` (± 3 per channel); card background `#FFFFFF`; cards have a 1 px `#DDE4EC` border | H, C | B | |
| 4.4 | No decorative shadow: `box-shadow` absent or `none` on cards, buttons, inputs, chips, tabs (focus ring and native overlays excepted); no `shadow-*` utility classes on them | H | B | |
| 4.5 | No `gradient`, `backdrop-filter`, `blur`, `glow`, `drop-shadow` anywhere | H | B | |
| 4.6 | Display serif is used on exactly two elements: the H1 and the hero numeral; every other element is Inter (or a system sans); no monospace except a SHA-256 string | H | B | |
| 4.7 | Hero digits are lining (all digits the same height) — screenshot of "37.3%" | V | M | |
| 4.8 | Every text size ≥ 13 px: no `font-size` < 13 px, no Tailwind `text-[9px…12px]` or `text-xs` | H | B | |
| 4.9 | No uppercase text: no `uppercase`/`text-transform: uppercase`; letter-spacing ≤ 0.02 em | H | M | |
| 4.10 | Colour roles: blue only in contribution bars, Placed chip and raise-group heading; orange only in bars, Not Placed chip and lower-group heading; red only in errors; navy in primary button, ruler fill, hero, focus ring | V, H | B | |
| 4.11 | Header contains only: H1, subtitle, three badges, step strip, scale rule | V | M | |
| 4.12 | Spacing: ten sampled gaps/paddings are all multiples of 4 px; card padding 24 px (16 on phones) | V, H | M | |
| 4.13 | The same component (chip, notice, field, tab, card) looks identical on every screen | V | M | |
| 4.14 | "Not generic" test: the screen has **fewer than two** of these markers — dark canvas, glow, gradient button, radial gauge, KPI tile row, icon sidebar, uppercase mono labels, status dots, glass cards | V | B | |
| 4.15 | Squint test (view at 25 %): the hero numeral and the ruler are the two dominant forms | V | M | |

## 5. Fields, validation and states

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 5.1 | Every field has a visible text label (not placeholder-only), a visible hint and a reserved error slot directly beneath the control | V | B | |
| 5.2 | Count fields (Internships, Projects, Workshops) are visible option groups with segments ≥ 44 × 44 px (not dropdowns); Extracurricular and Placement Training are No/Yes groups | V | B | |
| 5.3 | Number fields have +/- steppers and typed entry; no slider | V, H | B | |
| 5.4 | Buttons read exactly "Predict Placement Probability" (primary), "Load typical profile" (secondary), "Clear profile" (tertiary); no "Compute/Run/Simulate" | T | B | |
| 5.5 | Predict is enabled on the validation screen: no `disabled` attribute, no disabled colour (`#DDE4EC` fill), label unchanged | V, H | B | |
| 5.6 | Validation screen (CGPA = 9.5, HSC empty) shows the two inline messages **verbatim**, each with icon + 2 px red border + text, under the right fields | T, V | B | |
| 5.7 | Validation summary at the top of the Result card reads "Please correct the following before a prediction can be made:" followed by the same two items; no probability, ruler fill, chips or explanation tabs | T, V | B | |
| 5.8 | Each range hint appears once per field (no repeat in the summary row, status table or caption) | V | M | |
| 5.9 | Stale screen: banner "Inputs changed. Press Predict Placement Probability to update."; tag "From previous inputs"; changed field tagged "Prior: 7.7 → Now: 8.2" | T | B | |
| 5.10 | Stale result is distinguished by outline/hatch + tag, **not opacity**: no `opacity-*` below 100 on result text; every text ≥ 4.5:1 on its background; numbers unchanged from the original result | H, C | B | |
| 5.11 | Empty screen: all ten fields blank; Result card shows the descriptive sentence beginning "Enter a student profile and press Predict Placement Probability…", the empty ruler, and no numbers | V, T | B | |
| 5.12 | Model-load screen: headline "The prediction model could not be loaded."; sentence "The model file could not be loaded or did not pass its integrity check. No prediction is shown."; collapsed "Technical details"; no form, tiles, retry, timers, "remediation", IDs | V, T | B | |
| 5.13 | Typical-profile toast text (system sheet) reads "Typical profile loaded. Edit any value, then press Predict Placement Probability." | T | M | |
| 5.14 | The system sheet shows default, hover, focus, selected and error states for the number input and the option group | V | M | |

## 6. Responsive layout

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 6.1 | Desktop (1440 frame): two panes, widths 5/12 : 7/12 (± 5 %), tops aligned; content width ≤ 1200 px, centred | V | B | |
| 6.2 | Hero numeral and the whole ruler (including legend) lie within the first 900 px of the 1440 frame | V | B | |
| 6.3 | Phone frame is 390 px wide; one column; no horizontal overflow at 390 or 320 (document scroll width ≤ viewport width) | V, H | B | |
| 6.4 | On phones Predict is full width ≥ 48 px high; Load and Clear are full width ≥ 44 px, stacked beneath | V | B | |
| 6.5 | Classification cards: side by side at ≥ 640 px, stacked below 640 px | V | B | |
| 6.6 | Phone contribution rows are two lines (name + values, then bar); the bar track is ≥ 120 px wide | V | B | |
| 6.7 | Phone tab bar scrolls horizontally (cut-off last tab visible); tables become key–value lists | V | M | |
| 6.8 | Buttons, option segments and tabs are ≥ 44 × 44 px on phones (native stepper size is checked at implementation: §10.8) | V | B | |
| 6.9 | Hero numeral 52 px on phones; legend wraps; ruler never scrolls | V | B | |
| 6.10 | No `position: fixed` or `sticky` element on any screen | H | B | |
| 6.11 | Long strings ("Workshops / Certifications", error sentences) wrap without clipping | V | M | |

## 7. Accessibility

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 7.1 | Every text/background pair ≥ 4.5:1 (reference: ink 15.7/16.7 on canvas/white, muted 7.7, brand on white 9.1, white on brand 9.1, raise-ink on tint 6.5, lower-ink on tint 5.0, error on tint 6.1) — compute any colour not in the brief | C | B | |
| 7.2 | Non-text ≥ 3:1: control borders `#8A94A3` on **white** (3.07), ruler outline, marker ticks, bar centre line, focus ring | C | B | |
| 7.3 | `#D55E00` is used for graphics only; no orange text lighter than `#B54708` | H, C | B | |
| 7.4 | Meaning never by colour alone: each blue/orange element also has a sign, group heading, text label or shape | V | B | |
| 7.5 | Exactly one `h1`; one `h2` per section; `h3` for card titles; no skipped level | H | M | |
| 7.6 | Each input has an associated `<label>`/`aria-label`; hint and error are adjacent (same container) | H | M | |
| 7.7 | A visible focus style (2 px `#0F4C75` outline, 2 px offset) is defined for buttons, inputs, option segments and tabs | H, V | B | |
| 7.8 | Ruler and bars have an adjacent visible text equivalent; decorative parts are `aria-hidden` | H | M | |
| 7.9 | Errors identified by text + icon + border, naming the field and the allowed range | V, T | B | |
| 7.10 | CGPA, SSC, HSC, ROC-AUC, ECE are expanded in visible text at first use | T | M | |
| 7.11 | At 200 % zoom no content is clipped or overlaps (render the HTML at 200 %) | V | M | |
| 7.12 | Tooltips are never the only place for essential information (ranges, thresholds, disclaimers) | V, H | M | |

## 8. Motion (annotations on the system sheet; Stitch screens are static)

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 8.1 | Every CSS `animation-duration` / `transition-duration` (plus delay) in the HTML ≤ 450 ms | H | B | |
| 8.2 | The sheet's motion table matches brief §7: hero fade 150 ms; ruler fill 400 ms; bars 220 ms with ≤ 20 ms stagger (last ends ≤ 400 ms); hover/focus 100–120 ms; errors, stale, tabs not animated | T | B | |
| 8.3 | A `prefers-reduced-motion: reduce` rule (or the annotation "reduced motion removes all animation") is present | H, T | B | |
| 8.4 | No `infinite`, `animate-pulse`, `animate-ping`, `animate-spin` > 300 ms, parallax or shimmer; no count-up of the probability | H | B | |
| 8.5 | No hover transforms (`scale-*`, `translate-*`) on cards or buttons | H | M | |

## 9. Truthful methodology and copy

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 9.1 | Method tab states only: logistic regression, L2 (C = 1), no class weighting, standardised scores, one-hot counts/binaries, 8,425 students (train + validation), raw probabilities / no recalibration, SHA-256 prefix `58e8b04e9666b13b…` | T | B | |
| 9.2 | The word "calibrated" is not used as a claim about a person's probability; ECE is described as "average gap between predicted and observed rates across ten groups of students; lower is better" | T | B | |
| 9.3 | Evaluation caveat present verbatim: "These describe performance on one held-out sample from one dataset. They do not guarantee performance on other students, campuses or years." | T | B | |
| 9.4 | If a reference baseline is shown it is the project's (log loss 0.679, Brier 0.243); never 0.693 or 0.250; no other reference values | T | B | |
| 9.5 | Limitations list contains the six agreed points (association; constructed-data caveat; group not individual and ~1 in 5 misclassified; ranges rejected not adjusted; Projects = 0 rare; top-decile under-prediction 0.946 vs 0.905 not recalibrated) | T | M | |
| 9.6 | Terminology: "project threshold (0.35)" and "conventional (0.50)" in primary UI; none of "operating", "reference threshold", "Phase 6/7", "fingerprint" outside Method/Technical details | T | M | |
| 9.7 | Contributions are called "contribution(s)" / "factors"; groups use the exact three headings; the footnote and the odds-multiplier aid are present | T | B | |

## 10. Streamlit feasibility and regression

| # | Pass criterion | Method | Sev | Result |
|---|---|---|---|---|
| 10.1 | Every element maps to a native widget or static HTML/CSS (number input, radio group, button, columns, bordered container, tab, expander, table, key–value list, chip, ruler, bar). Anything else is listed and justified | S | B | |
| 10.2 | Nothing requires changing `prediction.py`, `explainability.py`, the model, preprocessing, thresholds or `classify` | S | B | |
| 10.3 | Nothing requires a database, auth, API, file storage or network call | S | B | |
| 10.4 | New static content (ECE row, threshold table, dataset source) is traceable to `README.md` / `src/phase7_findings.md` / `model_metadata.json` | S | M | |
| 10.5 | Fonts: no runtime CDN in the implementation plan; self-hosted or system fallback stack named | S | M | |
| 10.6 | Plan preserves `verify_app.py` locks: widget keys = feature names; the two button labels; `<div class="hero-prob">{p:.1f}%</div>`; ★ strings of brief §11.2; empty start; out-of-range cases (CGPA 9.5, Aptitude 95, HSC 40) → error, no hero | S | B | |
| 10.7 | Plan includes the approved `verify_app.py` section 6 update for counts as radio groups (D2) and the session-state stale detection for dropping `st.form` (D1) | S | B | |
| 10.8 | At implementation: native stepper size and first-press behaviour on an empty field verified; fallback (hidden steppers on touch widths) chosen if poor | S | M | |
| 10.9 | At implementation: `requirements.txt` pin raised to the version tested (D7) | S | M | |
| 10.10 | At implementation: `python app/verify_app.py` passes in full (baseline 122 checks; updated count documented), exit 0, test-file protection intact; keyboard walk-through; screenshots at 320 / 390 / 768 / 1440 px reviewed; brief §14 items resolved or fallbacks taken | S | B | |

## 11. Decision record

**Resolved on 2026-10-17 (see `design_brief.md` §2 and §13):**

| Item | Decision |
|---|---|
| A–J (direction) | A new set; B docs first; C horizontal ruler + disagreement region; D navy fill; E motion ≤ 450 ms + reduced motion; F display serif for hero numeral and H1; G Predict enabled, never runs on invalid input; H hairlines, no shadows; I light theme; J large central number, no gauge |
| D1 | Drop `st.form` |
| D2 | Counts as radio option groups (+ `verify_app.py` update) |
| D3 | Two-pane 5/12 : 7/12, stacked fallback |
| D4 | Static ECE row, threshold table, dataset source (documented values only) |
| D5 | Serif for hero + H1 only; Inter or system sans; no CDN |
| D6 | "Clear profile" included |
| D7 | Target 1.64.0; raise the pin in the implementation change |

Waivers (item, reason, who approved): ______________________________________________

**Overall verdict:** ☐ Approve screen for the next stage  ☐ Approve with listed changes  ☐ Send back to Stitch (failing items): ______________________
