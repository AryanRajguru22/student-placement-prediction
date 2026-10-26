# Google Stitch Prompts — Student Placement Probability Predictor

**Version 2 (2026-10-17).** Derived from `design_brief.md` v2 (canonical). If this file and the brief disagree, the brief wins. Review every generated screen with `design_review_checklist.md` before generating the next one.

## How to use (staged — do not skip steps)

1. **Create a new Stitch project** (recommended: the existing project's dark design system and two dark reference images steered the first set toward the wrong look; keep the old project as an archive and do not edit it).
2. **Apply the design-system document in Part 1** as the project's *only* design system (light). Do not apply the dark "Placement Intelligence" system. The front matter is untested against Stitch's importer: if it is rejected, apply only the prose sections plus the colour and type values, then confirm in the first screen that the serif appears only on the title and the hero numeral.
3. **Send the Main Prompt (Part 2).** It is self-contained and generates **one screen only**: the desktop result state. Review it against the checklist (sections 1–6 at minimum). Send corrections until it passes.
4. **Only then** send the screen prompts in Part 3, one at a time, in order: S2 → S9. Review each.
5. Stitch's HTML is a **visual reference**, not code to ship; Tailwind/Google-Fonts links inside it are expected and are not a defect. The Streamlit implementation is a separate, later stage.

Everything in the prompts comes from the verified implementation and project documents. The example numbers are the real model output for a hypothetical "typical profile" and are labelled as an example inside the prompts.

---

## Part 1 — Design-system document (apply to the new project; light only)

```
---
name: Editorial Analytical Instrument
colorMode: LIGHT
colors:
  canvas: '#F6F8FB'
  surface: '#FFFFFF'
  tint: '#EEF3F8'
  ink: '#0B1F33'
  ink-muted: '#475467'
  brand: '#0F4C75'
  brand-hover: '#0B3B5C'
  raise: '#0072B2'
  raise-ink: '#0A5C7A'
  raise-tint: '#E6F2F8'
  lower: '#D55E00'
  lower-ink: '#B54708'
  lower-tint: '#FFF4E5'
  error: '#B42318'
  error-tint: '#FEF3F2'
  control-border: '#8A94A3'
  line: '#DDE4EC'
typography:
  display-hero:      { fontFamily: Newsreader, fontSize: 72px, fontWeight: '600', lineHeight: 76px, letterSpacing: -0.01em }
  display-hero-mobile: { fontFamily: Newsreader, fontSize: 52px, fontWeight: '600', lineHeight: 56px, letterSpacing: -0.01em }
  headline-lg:       { fontFamily: Newsreader, fontSize: 32px, fontWeight: '600', lineHeight: 40px, letterSpacing: -0.01em }
  headline-md:       { fontFamily: Inter, fontSize: 20px, fontWeight: '600', lineHeight: 28px }
  headline-sm:       { fontFamily: Inter, fontSize: 16px, fontWeight: '600', lineHeight: 24px }
  body-lg:           { fontFamily: Inter, fontSize: 16px, fontWeight: '400', lineHeight: 24px }
  body-md:           { fontFamily: Inter, fontSize: 14px, fontWeight: '400', lineHeight: 20px }
  body-sm:           { fontFamily: Inter, fontSize: 13px, fontWeight: '400', lineHeight: 18px }
  label-md:          { fontFamily: Inter, fontSize: 14px, fontWeight: '600', lineHeight: 20px }
  label-sm:          { fontFamily: Inter, fontSize: 13px, fontWeight: '600', lineHeight: 18px }
rounded: { ruler: 2px, input: 8px, card: 12px, chip: 9999px }
spacing: { xs: 4px, sm: 8px, md: 12px, lg: 16px, xl: 24px, 2xl: 32px, 3xl: 48px }
---

## Brand and style
A premium, light, editorial analytical workspace for a university machine-learning project. Calm, precise, quantitative. Restraint is the luxury: whitespace, hairlines, one navy accent, one serif numeral and one signature device, the horizontal threshold ruler. It is NOT a generic AI dashboard, admin template, college portal, or neon/glassmorphism interface. No dark theme, gradients, glow, blur, glass, decorative shadows, radial gauges or dials, stock imagery, mascots or celebratory effects.

## Colour
Canvas #F6F8FB, white cards (#FFFFFF) with a 1 px #DDE4EC hairline. Ink #0B1F33; secondary text #475467. Brand navy #0F4C75 is used for the primary button, the probability ruler fill, the hero numeral, links and focus rings. Blue (#0072B2, text #0A5C7A on #E6F2F8) means "toward Placed" and orange (#D55E00 for bar fills only; text #B54708 on #FFF4E5) means "toward Not Placed"; they are used only for contribution bars, the Placed / Not Placed chips and group headings. The probability band is neutral (outlined, #475467). Red (#B42318 on #FEF3F2) is used only for input errors. Form-control borders are #8A94A3 on white surfaces. Never red/green, never tick/cross icons for classification. Meaning is never carried by colour alone.

## Typography
Display serif (Newsreader; alternatives Source Serif 4, then Fraunces), lining figures, used ONLY for the hero probability numeral and the page title. Everything else is Inter with tabular numerals. Minimum text size 13 px. Sentence case everywhere: no uppercase, no letter-spaced micro-labels, no monospace.

## Surfaces and shape
White cards on the canvas with 1 px hairlines and NO box-shadow. Radius: cards 12, inputs and buttons 8, chips fully rounded, ruler 2. 4-px spacing base. Left-aligned text, generous whitespace.

## Components
Primary button: solid #0F4C75, white text, 8 px radius, 48 px high on mobile / 40 px desktop, no shadow, hover #0B3B5C. Secondary: white, 1 px #8A94A3 border, ink text. Tertiary: text only. Inputs: white, 1 px #8A94A3, 8 px radius, native-style +/- steppers for number fields; focus = 2 px #0F4C75 outline with 2 px offset. Option groups: a row of at least 44 x 44 px segments, selected = navy outline and tint. Cards: white, hairline, 12 px radius, 24 px padding. Chips: pill, text plus a neutral shape glyph (square = Placed, circle = Not Placed). Notices: info (tint), caution (orange tint), error (red tint) always with icon and text. NO sliders, NO sidebar, NO top navigation.

## Motion (annotations only; Stitch screens are static)
Nothing animates longer than 450 ms. Ruler fill 400 ms; hero fade 150 ms; contribution bars 220 ms with at most 20 ms stagger; hover/focus 100-120 ms; errors and stale states do not animate; no count-up, no looping, no pulsing; prefers-reduced-motion removes all of it.
```

---

## Part 2 — MAIN PROMPT (paste into Stitch; generates the desktop result screen only)

===== BEGIN MAIN PROMPT =====

TASK
Design the DESKTOP RESULT SCREEN (1440 px wide frame, content width at most 1200 px, LIGHT theme only) of "Student Placement Probability Predictor": a university machine-learning portfolio project that turns ten student characteristics into an estimated probability of campus placement, two threshold-based classifications, and an explanation of which inputs shaped the estimate. Generate ONLY this one screen now; other screens will be requested later, so build reusable components (header, field card, option group, chip, notice, ruler, contribution row, tabs).

VISUAL GOAL
A premium, distinctive, light editorial analytical workspace. Calm, precise, quantitative. NOT a generic AI dashboard, NOT an admin-panel template, NOT a college portal or marketing page, NOT neon, dark or glassmorphism. Restraint is the luxury: whitespace, 1 px hairlines, one navy accent, one serif numeral, and one signature device, the horizontal threshold ruler. Use the project's design system (light). Hero numeral and page title in a display serif (Newsreader; lining figures); everything else in Inter. No gradients, glow, blur, glass, drop shadows on cards, radial gauges or dials, stock imagery, icons-as-decoration, status strips, version or latency badges, avatars, top navigation, sidebar, KPI tiles or progress bars. No uppercase or letter-spaced micro-labels; no monospace; nothing smaller than 13 px.

HARD CONSTRAINT: IT MUST BE BUILDABLE IN STREAMLIT
The design will be implemented in Python Streamlit with light CSS. Use only: page header, headings, text, number inputs with +/- steppers, option groups (a short row of 3-4 visible choices), No/Yes choice groups, primary/secondary/tertiary buttons, columns and bordered cards, tabs, expanders, tables, key-value lists, badges/chips, info/caution/error notices, toasts, and static HTML/CSS graphics (a horizontal probability ruler and horizontal bars). Do NOT design: sliders, dials or gauges of any kind, maps, 3D, carousels, blocking modals, hover-only content, sticky or floating panels, multi-page navigation, JavaScript behaviour, or charts that need data not listed below. ONE page, ONE workspace.

DO NOT FABRICATE
Use only the facts, fields, ranges, labels and numbers given in this prompt. Do not invent inputs, outputs, metrics, charts, statistics, dataset facts, user counts, features, navigation items, system identifiers, or any claim about performance. Where this prompt gives a number, use exactly that number. If you need a number that is not given, use a clearly marked placeholder such as "XX.X". The screen shows the real model output for a hypothetical "typical profile"; mark it with a small visible tag "Example profile".

PRODUCT FACTS
- Model: Logistic Regression (L2 regularised), trained on 8,425 students from a public dataset (Hugging Face "Krooz/Campus_Recruitment_CSV", CC0-1.0). Frozen: nothing is retrained in the app. Probabilities are raw model output, not recalibrated. The app itself does not save or log the values entered.
- Outputs: (1) the estimated probability that a student is Placed; (2) a classification at the 0.35 threshold; (3) a classification at the conventional 0.50 threshold; (4) a descriptive probability band; (5) an additive explanation: each input's contribution to the model's log-odds relative to a baseline student.
- The probability is the primary output; a classification only applies a cut-off to it. Rule: Placed when probability is greater than or equal to the threshold, applied separately for 0.35 and 0.50.
- 0.35 ("project threshold") was selected in the project by a rule fixed in advance (highest validation F1). It finds more placed students but raises more false alarms than 0.50. It is NOT claimed to be best or optimal. 0.50 is the conventional threshold, shown for comparison. Never call either "recommended", "optimal" or "best".
- Probability bands (descriptive only, NOT confidence intervals, independent of thresholds): below 20% Very low, 20-40% Low, 40-60% Moderate, 60-80% High, 80% and above Very high ("estimated probability").
- Explanations are associations learned from one observational dataset, not causal effects. No advice, no tips, no "how to improve", no what-if simulator, no confidence or credible interval on the prediction, no SHAP, no "importance" or "impact" wording.

INPUTS (exactly ten; all required; all EMPTY on first load; nothing pre-filled)
Group "Academic": 1 CGPA (number, step 0.1, supported range 6.5 to 9.1); 2 SSC Marks (number, step 1, 55 to 90); 3 HSC Marks (number, step 1, 57 to 88).
Group "Technical": 4 Internships (choose one of 0, 1, 2); 5 Projects (choose one of 0, 1, 2, 3; note beside it: "Projects = 0 is rare in the development data (22 of 8,425 students); the estimate for it is unreliable."); 6 Workshops / Certifications (choose one of 0, 1, 2, 3); 7 Aptitude Test Score (number, step 1, 60 to 90).
Group "Employability": 8 Soft Skills Rating (number, step 0.1, 3.0 to 4.8); 9 Extracurricular Activities (No / Yes); 10 Placement Training (No / Yes).
There are no other inputs (no Backlogs, Hackathons, department, gender, name, etc.). Number fields are typed fields with +/- steppers and a visible hint under each ("Supported range: 6.5 to 9.1"); NO sliders (they would clamp values; the product never clips or silently corrects input). Count fields are visible option buttons (at least 44 x 44 px each), not dropdowns. Section helper line: "All ten fields are required. Supported ranges are the ranges observed in the data the model was developed on."
Buttons (exact labels): primary "Predict Placement Probability" (never "Compute", "Run" or "Simulate"); secondary "Load typical profile" with helper text "Median for scores and counts, most common answer for Yes/No. A starting point, not a recommendation."; tertiary "Clear profile".
The Predict button is always enabled. Typical profile used for this screen: CGPA 7.7, SSC Marks 70, HSC Marks 73, Aptitude Test Score 80, Soft Skills Rating 4.4, Internships 1, Projects 2, Workshops / Certifications 1, Extracurricular Activities Yes, Placement Training Yes. On THIS screen the ten fields show these typical-profile values (they were filled with "Load typical profile").

LAYOUT (this screen)
A. Header: H1 "Student Placement Probability Predictor" (display serif); subtitle "Estimates the probability that a student is placed from ten academic, technical and employability characteristics."; three outline text badges "Estimate, not a guarantee", "Frozen model · Logistic regression", "Nothing is saved"; a one-line step strip "1 Enter profile → 2 Read the estimate → 3 See why"; a thin static hairline scale rule with tick marks under it (the only decoration).
B. Left pane (5/12): "Student profile": the three grouped white cards with the ten fields, then the three buttons.
C. Right pane (7/12), top-aligned with the left pane: "Result" card, then "What shaped this estimate" tabs.
D. Full-width below: "About the model" with tabs Method, Evaluation, Thresholds, Limitations and data; show the Evaluation tab open.
The hero number and ruler must be visible without scrolling at 1440 x 900.

RESULT CARD (top to bottom; example profile)
1. Eyebrow (14 px, sentence case, muted): "Estimated probability of placement". Centred stack: the hero number "37.3%" in the display serif, navy, 72 px, lining figures — the largest element on the screen — and below it "Model estimate for students with these characteristics — not a guarantee." Small tag "Example profile".
2. THE THRESHOLD RULER (the signature element; a horizontal bar, NOT a radial gauge): a 14 px high track spanning the card width with a 1 px #8A94A3 outline and #EEF3F8 fill; scale labels "0%" at the left end and "100%" at the right end only; small minor ticks every 10% with the 50% tick longer; the NAVY (#0F4C75) fill runs from 0 to 37.3% with a flat end and no label at the fill end. Two engraved vertical markers cross the track at exactly 35% and 50% (2 px ink line with a thin white halo so they stay visible over the fill): a FILLED diamond above the 35% marker with the numeral "0.35", and a HOLLOW diamond above the 50% marker with the numeral "0.50". The span between 35% and 50% is drawn with diagonal hatching on the track, plus a small bracket beneath the scale labelled "Labels differ in this region". Only the numerals "0.35" and "0.50" sit at the markers; all longer text goes in a legend row beneath that wraps on narrow screens: "◆ 0.35 (35%) project threshold · ◇ 0.50 (50%) conventional threshold · ▨ labels differ here". Beneath the legend, one plain sentence: "Estimated probability 37.3%: above the 0.35 threshold, below the 0.50 threshold." No labels may overlap.
3. Disclaimer strip directly under the ruler block (neutral, left border): "This probability is an estimate produced by the trained model for students with the supplied characteristics. It is not a guarantee of placement and should not be interpreted as a causal assessment."
4. One info notice (merged "how to read"): "The two thresholds disagree for this profile (probability between 0.35 and 0.50). That is expected near a cut-off; the probability itself is the primary output. The threshold changes the label, not the probability."
5. Two classification cards of equal size and weight, side by side:
 - Title "Model classification", sub "at the 0.35 project threshold". Chip "■ Placed" (blue tint, text #0A5C7A). Text "Placed when probability is 0.35 or higher." Note "Chosen in the project by a rule fixed in advance (highest validation F1). It finds more placed students but raises more false alarms. Not claimed to be the best threshold."
 - Title "Conventional 0.50 threshold". Chip "● Not Placed" (orange tint, text #B54708). Text "Placed when probability is 0.50 or higher. Shown for comparison."
6. Band row: a neutral outlined tag "Interpretation band: Low estimated probability" (NOT blue or orange) with the caption "A descriptive label for the size of the probability — not a confidence interval and independent of the thresholds. A 'Low' band can still be 'Placed' at the 0.35 threshold: the band describes the size of the probability, the classification applies a cut-off."

WHAT SHAPED THIS ESTIMATE (tabs "Factors" default, "Full calculation", "Inputs used")
Factors tab: exactly three groups with these headings: "Factors that raise the model's estimate" (blue), "Factors that lower the model's estimate" (orange), "At the baseline (no contribution)". Each row: label; small subtext comparing the value with the baseline; a horizontal bar diverging from a centre line (with small end ticks echoing the ruler); the log-odds contribution with sign; and the odds multiplier as a second figure. Show ALL ten features, exactly once:
 Raise: Placement Training — Yes (reference level No) — +0.97 — ×2.65 | Extracurricular Activities — Yes (reference level No) — +0.72 — ×2.06 | Soft Skills Rating — 4.4 (training average 4.3) — +0.05 — ×1.05 | Aptitude Test Score — 80 (training average 79.3) — +0.04 — ×1.04 | SSC Marks — 70 (training average 69.1) — +0.02 — ×1.02 | CGPA — 7.7 (training average 7.7) — +0.00 — ×1.00
 Lower: Workshops / Certifications — 1 (reference level 0) — −0.32 — ×0.72 | Projects — 2 (reference level 3) — −0.31 — ×0.73 | HSC Marks — 73 (training average 74.4) — −0.03 — ×0.97
 Baseline (text list): Internships = 1 (reference level 1)
Bars are scaled to the largest contribution in this prediction (Placement Training). Orange appears as bar fill only; orange text uses #B54708. Always-visible footnote: "Contributions are measured against a baseline student — training-average scores and reference categories. Bar length is relative to the largest contribution in this prediction. These are associations learned by the model, not causal effects." One-line aid: "An odds multiplier of ×2.65 means the odds of 'Placed' are multiplied by 2.65 relative to the baseline, within the model."

ABOUT THE MODEL (Evaluation tab shown)
Title exactly: "Results from this project's final test evaluation (1,487 students, evaluated once)". Present four metrics as a quiet key-value list (NOT dashboard tiles, no progress bars, no icons, no arrows): ROC-AUC 0.886 (95% CI 0.868 to 0.902) — "how well the model ranks placed students above non-placed ones; this is not accuracy"; Log loss 0.420 — "penalises confident wrong probabilities; lower is better"; Brier score 0.134 — "mean squared error of the probabilities; lower is better"; Calibration error (ECE) 0.016 — "average gap between predicted and observed rates across ten groups of students; lower is better". Caveat: "These describe performance on one held-out sample from one dataset. They do not guarantee performance on other students, campuses or years." Do not add any other metric and never describe ROC-AUC as accuracy.

EXPLICITLY OUT OF SCOPE — DO NOT ADD
Student name, roll number, e-mail, photo, avatar or any personal field; login, accounts, saved profiles, history, export, sharing; batch upload; peer comparison, percentiles, rankings; chatbot or assistant; improvement tips or advice; what-if sliders; a confidence or credible interval on the prediction; "Recommended", "Optimal" or "Best" on a threshold; any accuracy claim; navigation, sidebar, avatar, footer telemetry, status strips; dark theme; radial gauge; marketing imagery.

OUTPUT
One desktop screen, light theme, hairline surfaces, navy ruler fill, serif hero numeral, the three-way distinction clear at a glance: the probability first, then its two classifications and how to read them, then why, then how far to trust it.

===== END MAIN PROMPT =====

---

## Part 3 — Screen prompts (send one at a time, only after the previous screen passes the checklist)

Each prompt assumes the previous screen exists in the project. They restate the facts they need.

**S2 — Desktop, empty first load**
"Using exactly the same design system and components as the desktop result screen, design the DESKTOP FIRST-LOAD EMPTY STATE. All ten fields are empty (no value selected anywhere), each showing its label and visible range hint (CGPA: 6.5 to 9.1; SSC Marks: 55 to 90; HSC Marks: 57 to 88; Aptitude Test Score: 60 to 90; Soft Skills Rating: 3.0 to 4.8; Internships 0/1/2; Projects 0/1/2/3; Workshops / Certifications 0/1/2/3; Extracurricular Activities No/Yes; Placement Training No/Yes). The Predict button is enabled. The Result card is a designed empty state: the EMPTY ruler (outline track, the two threshold markers at 0.35 and 0.50, the hatched region, NO fill), a muted dash where the number will be, and the text 'Enter a student profile and press Predict Placement Probability to see an estimated probability, how it compares with two classification cut-offs, and which inputs shaped it.' No numbers, no example tag, no explanation tabs. Keep the About the model section collapsed to its tab bar."

**S3 — Desktop, validation error**
"Same design system. Design the DESKTOP VALIDATION ERROR STATE after the user pressed Predict Placement Probability with CGPA = 9.5 and HSC Marks empty (other fields filled with the typical profile). The Predict button is still enabled. Inline messages sit directly under the failing fields, each with an icon, a 2 px red border and text (never colour alone): 'CGPA: 9.5 is outside the range the model was developed on (6.5 to 9.1); no prediction is made for out-of-range values.' and 'HSC Marks: a value is required.' At the top of the Result card a red-tint error notice: 'Please correct the following before a prediction can be made:' followed by the same two items as a list. No probability, no ruler fill, no explanation tabs are shown. Show each range hint once only. Use only real fields; do not add any other field or status text."

**S4 — Desktop, stale result**
"Same design system. Design the DESKTOP STALE STATE: a prediction was made for the typical profile (37.3%, Placed at 0.35, Not Placed at 0.50), then CGPA was changed from 7.7 to 8.2 and Predict has not been pressed. Above the old result, a caution banner: 'Inputs changed. Press Predict Placement Probability to update.' The CGPA field carries a 'Changed' tag with the text 'Prior: 7.7 → Now: 8.2'. The old result card is tagged 'From previous inputs' and distinguished by a dashed outline and a hatched border — do NOT lower its opacity; all text must stay at least 4.5:1 contrast. Show the old numbers unchanged (37.3%, same ruler, same chips). Do not show any new probability, any benchmark, or any extra stale indicators beyond the banner, the tags and the outline."

**S5 — Mobile (390 px) form and result**
"Same design system. Design the MOBILE (390 px wide) screen as one long stacked page: header (title, subtitle, three badges, step strip), then the three grouped field cards with all ten inputs (typical-profile values), option groups with segments at least 44 x 44 px, then a full-width 48 px primary 'Predict Placement Probability', with 'Load typical profile' and 'Clear profile' stacked beneath at 44 px. Then the result in this order: eyebrow, hero '37.3%' (52 px, serif, navy, centred), the horizontal ruler with only the numerals '0.35' and '0.50' at the markers and the legend wrapping below, the sentence, the disclaimer strip, the merged how-to-read notice, the two classification cards STACKED, the neutral band tag, and the Factors tab with each contribution row on two lines (name and values first, full-width bar beneath). No bottom bar, no navigation, no sliders, no horizontal scroll at 390 or 320 px. Tag it 'Example profile'."

**S6 — Desktop, About the model: Thresholds and Limitations**
"Same design system. Design the desktop 'About the model' section twice, stacked: (1) the Thresholds tab: a comparison table for the same held-out test set — Accuracy 0.802 at 0.35 vs 0.812 at 0.50; Precision 0.722 vs 0.779; Recall 0.851 vs 0.765; F1 0.781 vs 0.772; Specificity 0.766 vs 0.846 — and the sentence '0.35 finds more placed students but raises more false alarms. Neither threshold is universally better: the right cut-off depends on the relative cost of false positives and false negatives, which this project has not defined.' (2) the Limitations and data tab: six plain bullets — associations not causation; one observational dataset whose structure is consistent with a rule-based or constructed origin (not established by the source), so real-world equivalence is not shown; a probability describes a group of similar students, not a certainty (roughly 1 in 5 held-out students misclassified at either threshold); supported ranges are those observed in the development data and out-of-range values are rejected, never adjusted; Projects = 0 is rare (22 of 8,425) and its estimate is unreliable; a small under-prediction in the top decile (observed 0.946 vs predicted 0.905) was not recalibrated — plus the source line 'Dataset: Hugging Face Krooz/Campus_Recruitment_CSV (CC0-1.0).' No other numbers, no charts."

**S7 — Mobile, About the model**
"Same design system. Design the MOBILE (390 px) 'About the model' section: a horizontally scrolling tab bar (Method, Evaluation, Thresholds, Limitations and data) with the Evaluation tab open: the exact title, the four-row key-value metric list (no tiles, no progress bars) and the caveat. Then the Thresholds tab on phones as stacked key-value rows (each metric with its 0.35 and 0.50 values on two lines). No fixed bottom bar."

**S8 — Model could not load**
"Same design system. Design the FULL-PAGE MODEL-LOAD FAILURE state: one calm full-width card with the headline 'The prediction model could not be loaded.', the sentence 'The model file could not be loaded or did not pass its integrity check. No prediction is shown.', and a collapsed 'Technical details' disclosure. No form, no navigation, no status tiles, no retry buttons, no timers, no 'self-remediation', no identifiers. Desktop and mobile variants."

**S9 — Design-system sheet**
"Same design system. Produce a LIGHT design-system sheet: colour tokens with roles and the two contrast-checked pairs for text (#0B1F33 and #475467 on white); type scale (serif hero 72/52 and H1 32/28 only; Inter for everything else; nothing below 13 px); spacing and radius; hairline-only surfaces; components with default / hover / focus / selected / error states (number input, option group, No/Yes group, primary/secondary/tertiary buttons, chip with square and circle glyphs, notices, tabs, contribution row, band tag); the two 'how to read' notice variants — disagree ('The two thresholds disagree for this profile (probability between 0.35 and 0.50). That is expected near a cut-off; the probability itself is the primary output.') and agree ('Both thresholds give the same label.'), each followed by 'The threshold changes the label, not the probability.'; a toast reading 'Typical profile loaded. Edit any value, then press Predict Placement Probability.'; the ruler anatomy with labelled parts (track, navy fill, filled and hollow diamond markers, hatched 'labels differ' region, legend row); and a motion annotation table (ruler fill 400 ms, hero fade 150 ms, bars 220 ms with at most 20 ms stagger, hover 100-120 ms, nothing longer than 450 ms, no animation for errors or stale states, reduced motion removes all). No sliders, no gauge, no dark variant, no live simulator, no invented components."

## Refinement prompts (use after a review, as needed)

**R1 — Hierarchy and ruler:** "Make the 37.3% numeral clearly the dominant element. On the ruler: navy fill only, filled diamond 0.35 and hollow diamond 0.50 with the numerals at the markers and everything else in the legend row, the hatched 35%-50% region with its bracket label, no label at the end of the fill, nothing overlapping at 390 px. Remove any gauge, arc, glow, gradient or shadow. Do not add any recommendation, confidence interval or advice text."

**R2 — Explanation and contrast:** "In the Factors tab keep the three exact group headings, show all ten features once, diverging bars around a centre line, sign, log-odds value and odds multiplier on every row. Orange text must be #B54708 (the #D55E00 orange is for bar fills only). Centre lines, ruler outline and control borders must reach 3:1 contrast and text 4.5:1. Keep the footnote about baseline and 'associations, not causal effects' visible."

**R3 — Strip anything invented:** "Remove every element that is not in the brief: navigation, avatars, status strips, version, seed, latency or identifier badges, benchmark or peer panels, confidence intervals, extra metrics, extra fields (Backlogs, Hackathons, etc.), simulators and any uppercase or monospace micro-labels. Replace any number that was not given with a placeholder or remove it."
