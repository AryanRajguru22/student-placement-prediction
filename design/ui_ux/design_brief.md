# UI/UX Design Brief — Student Placement Probability Predictor

**Version 2 — final pre-generation specification (2026-10-17).** Supersedes version 1 (2026-10-14). This is the **canonical source of truth** for the redesign.
**Status:** specification only. No application code, model, data or Stitch screen has been changed by this document.
**Authority order when documents disagree:** (1) the code and `app/model_metadata.json` (facts); (2) this brief (design decisions); (3) `stitch_prompt.md` (a derivation of this brief); (4) `design_review_checklist.md` (acceptance tests derived from this brief); (5) `stitch_review.md` and `current_app_audit.md` (history and evidence, cited as *Audit §n* / *Review §n*).
**Baseline:** the Phase 8A app passed its 122-check suite (`app/verify_app.py`) on 2026-10-17 in two independent runs (exit 0). That is the pre-redesign baseline; it says nothing about any new design (*Audit §0*).
**Streamlit facts** were verified on the installed **1.64.0**; `requirements.txt` still says `>=1.50` (see D7, §13).

---

## 1. Product positioning and users

**What it is.** A single-student *probability estimator* on a frozen, documented logistic-regression model (`LR_B`, refit on 8,425 students, evaluated once on 1,487 held-out students). It returns a **probability**, two **threshold-based classifications** and an **additive explanation** of how the ten inputs built that probability.

**What it should feel like.** A calm, precise, quantitative instrument with an editorial finish: the reader sees *this is a model; here is exactly what it was given; here is what it says; here is why; here is how far to trust it.* Not a college portal, not career advice, not a generic form, not an "AI dashboard".

**Positioning line (copy basis).** "An explainable probability model of campus placement, built as a university machine-learning project. It estimates — it does not decide, advise or guarantee."

**Credibility levers (all true in the code; surface them, never invent others):** frozen and fingerprint-verified model; probability shown to one decimal and never rounded internally; explicit refusal of out-of-range input; exact additive explanation; evaluation with confidence intervals and calibration; honest limitations.

| User | Context | Needs |
|---|---|---|
| Examiner / viva panel (primary) | Opens the app during a demo or marking | Grasp the tool in under a minute; see both thresholds and the explanation; find metrics and limitations quickly; nothing overclaimed |
| Project author presenting | Live demo, projector, sometimes a phone | Fast path (Load typical profile → Predict), result visible without scrolling, large legible numbers |
| Peer student / curious user | "What does the model say for a profile like this?" | Clear ranges, plain-language reading, reassurance nothing is saved |
| Portfolio visitor | Skims screenshots or a short session | A polished, coherent system; evidence of rigour |

Use cases: enter a hypothetical profile; read an estimate; understand why; compare the two cut-offs; inspect model and evaluation facts; read limitations. Non-use-cases: advising a real student, ranking, screening, record keeping, "how do I raise my chances" (§15).

## 2. Final design direction

**One-sentence direction.** *A premium, distinctive, light editorial analytical workspace: warm-neutral hairline surfaces, one navy accent, a serif display numeral, and a single signature device — the horizontal threshold ruler.*

**Authoritative decisions (A–J)** — settled; do not reopen without a new requirement:

| | Decision | Where specified |
|---|---|---|
| A | Generate a **new** Stitch set; do not edit the existing dark screens | `stitch_review.md` §12 |
| B | Finalize documents **before** generating | this set |
| C | **Horizontal threshold ruler** explains 0.35 vs 0.50 and marks the disagreement region; **no radial gauge** | §5.4 |
| D | **Navy** (`--brand`) fills the probability meter | §5.4, §6.1 |
| E | Motion restrained and purposeful; **no animation longer than 450 ms**; reduced-motion honoured | §7 |
| F | A distinctive **display font** may be used for the hero probability (and page title); body text and controls stay highly readable | §6.2 |
| G | **Predict stays available** with errors; **never runs** on invalid input; inline + accessible error messaging | §9 |
| H | **Hairline borders, no decorative card shadows** | §6.3 |
| I | **Light theme** for release 1 | §6.1 |
| J | Keep a **large central probability number**; remove the radial gauge | §5.3, §5.4 |

**Signature device — the ruler.** The probability ruler is the product's one recognisable motif: hairline axis, ticks, navy fill, two engraved shape-coded threshold markers, a hatched "labels differ here" region. It reappears only in three places: the header's static scale rule, the empty-state outline of the result panel, and the centre-line ticks of the contribution bars. Nothing else is decorative.

**Explicitly not:** dark/obsidian/cyan, glassmorphism, glow, gradients, radial gauges or dials, neon, monospace "terminal" labels, uppercase letter-spaced micro-labels, status strips, telemetry, top navigation, avatars, KPI cards with progress bars, stock imagery, confetti or any celebratory/alarming treatment of a classification.

## 3. Information architecture

**One page, one workspace; no navigation.** Two panes on wide screens (profile left, result right) with tabs only for secondary content; it degrades to a single stacked column. Multi-page navigation was rejected (the explanation is meaningless without the inputs and result beside it).

```
┌───────────────────────────────────────────────────────────────────────┐
│ A. HEADER   H1 · purpose · 3 badges · 3-step strip · static scale rule │
├───────────────────────────────┬───────────────────────────────────────┤
│ B. STUDENT PROFILE (5/12)     │ C. RESULT (7/12)                      │
│  3 grouped cards, 10 fields   │  empty state ⇄ result                 │
│  [Predict] [Load] [Clear]     │  hero · ruler · note · 2 cards · band │
│                               │ D. WHAT SHAPED THIS ESTIMATE (tabs)   │
├───────────────────────────────┴───────────────────────────────────────┤
│ E. ABOUT THE MODEL (full width, tabs)                                  │
└───────────────────────────────────────────────────────────────────────┘
```
Narrow screens: A → B → C → D → E. Panes are top-aligned; at 1440 × 900 the hero and ruler are visible without scrolling. **Fallback:** if nesting (form pane + tabs inside the result pane) misbehaves in the prototype, use the stacked order on all widths — same design system, different arrangement (D3).

## 4. User journey
1. **Land.** Header and empty state say what the tool does and what will appear.
2. **Fill.** *Load typical profile* or type ten values; ranges are visible; an invalid entry is explained inline as soon as it commits.
3. **Predict.** Invalid or missing → summary plus inline messages and no result; valid → result.
4. **Read.** Hero → ruler → one "how to read" note → two classification cards → band.
5. **Understand.** Factors, full calculation, inputs used.
6. **Calibrate trust.** Disclaimer beside the number; About the model one click away.
7. **Iterate.** Editing any field marks the old result "From previous inputs" until Predict is pressed again.

## 5. Screen specification

### 5.1 A — Header
- `h1` "Student Placement Probability Predictor" (display serif, §6.2).
- Subtitle: "Estimates the probability that a student is placed from ten academic, technical and employability characteristics."
- Badges (text only, outline, no icons): **Estimate, not a guarantee** · **Frozen model · Logistic regression** · **Nothing is saved** (the app itself does not save or log the values entered; they exist only for the browser session).
- 3-step strip, one line of static text: "1 Enter profile → 2 Read the estimate → 3 See why".
- A static, `aria-hidden` hairline scale rule (0–100 ticks) under the strip. No navigation, avatar, status, version, latency or telemetry.

### 5.2 B — Student profile
Section heading "Student profile" with one helper line: "All ten fields are required. Supported ranges are the ranges observed in the data the model was developed on."

Three grouped cards (same grouping as the current app):
- **Academic:** CGPA, SSC Marks, HSC Marks
- **Technical:** Internships, Projects, Workshops / Certifications, Aptitude Test Score
- **Employability:** Soft Skills Rating, Extracurricular Activities, Placement Training

| # | Field (label) | Widget key | Control | Visible hint (always shown) |
|---|---|---|---|---|
| 1 | CGPA | `CGPA` | number input, step 0.1, native +/− | Supported range: 6.5 to 9.1 |
| 2 | SSC Marks | `SSC_Marks` | number input, step 1 | Supported range: 55 to 90 |
| 3 | HSC Marks | `HSC_Marks` | number input, step 1 | Supported range: 57 to 88 |
| 4 | Internships | `Internships` | option group **0 · 1 · 2** | — |
| 5 | Projects | `Projects` | option group **0 · 1 · 2 · 3** | "Projects = 0 is rare in the development data (22 of 8,425 students); the estimate for it is unreliable." (neutral note beside the options) |
| 6 | Workshops / Certifications | `Workshops/Certifications` | option group **0 · 1 · 2 · 3** | — |
| 7 | Aptitude Test Score | `AptitudeTestScore` | number input, step 1 | Supported range: 60 to 90 |
| 8 | Soft Skills Rating | `SoftSkillsRating` | number input, step 0.1 | Supported range: 3.0 to 4.8 |
| 9 | Extracurricular Activities | `ExtracurricularActivities` | choice group **No · Yes** | — |
| 10 | Placement Training | `PlacementTraining` | choice group **No · Yes** | — |

These ten are the **only** inputs (no Backlogs, Hackathons, department, gender, name or anything else).
Rules: all ten required; **all start empty** (no default selected); **no sliders** (they clamp); no `min_value`/`max_value` on number inputs (out-of-range is reported, never blocked or coerced); steppers are the native ones and never clamp; each field = visible label + control + visible range/option hint + an error slot directly beneath. Controls sit on white surfaces (`--control-border` fails 3:1 on the canvas).

Actions (order, left→right on desktop; stacked full-width on phones): primary **Predict Placement Probability**; secondary **Load typical profile**; tertiary **Clear profile**. Helper under *Load typical profile*: "Median for scores and counts, most common answer for Yes/No. A starting point, not a recommendation." After loading: toast "Typical profile loaded. Edit any value, then press Predict Placement Probability." *Clear profile* empties the ten fields and removes any result and errors.

### 5.3 C — Result panel
**Empty state (first load, after Clear):** the panel is designed, not blank: the **empty ruler** (outline track, the two threshold markers and the hatched region, no fill), a muted placeholder "–" where the number will be, and: "Enter a student profile and press Predict Placement Probability to see an estimated probability, how it compares with two classification cut-offs, and which inputs shaped it." No sample numbers.

**Result, top → bottom** (this order is the same on phones):
1. **Eyebrow** (Inter 14/600, sentence case, muted): "Estimated probability of placement".
2. **Hero number**, **central**: the eyebrow, numeral and subline form a centred stack on the card axis. Display serif, navy, 72 px desktop / 52 px phone, lining figures, e.g. `37.3%`. Markup keeps `<div class="hero-prob">{p:.1f}%</div>` (one decimal; display rounding only). Subline: "Model estimate for students with these characteristics — not a guarantee."
3. **Ruler** (§5.4) with legend and a one-sentence text equivalent.
4. **Disclaimer strip**, directly under the ruler block (not at the bottom): "This probability is an estimate produced by the trained model for students with the supplied characteristics. It is not a guarantee of placement and should not be interpreted as a causal assessment."
5. **"How to read this" note** (one merged notice, not three): the agreement/disagreement sentence plus "The threshold changes the label, not the probability."
   - Thresholds disagree (0.35 ≤ p < 0.50): "The two thresholds disagree for this profile (probability between 0.35 and 0.50). That is expected near a cut-off; the probability itself is the primary output."
   - Otherwise: "Both thresholds give the same label."
6. **Two classification cards**, equal size and weight (side by side ≥ 640 px, stacked below):
   - *Model classification* — sub "at the 0.35 project threshold". Chip **■ Placed** / **● Not Placed** (text + neutral shape; never ✓/✕, never red/green). Text "Placed when probability is 0.35 or higher." Note: "Chosen in the project by a rule fixed in advance (highest validation F1). It finds more placed students but raises more false alarms. Not claimed to be the best threshold."
   - *Conventional 0.50 threshold*. Chip. Text "Placed when probability is 0.50 or higher. Shown for comparison."
   Both chips use the same size and weight; Placed = blue tint/ink, Not Placed = orange tint/ink (direction colours, §6.1).
7. **Band row** — a neutral outlined tag (muted ink; **never blue/orange**): "Interpretation band: Low estimated probability". Caption: "A descriptive label for the size of the probability — not a confidence interval and independent of the thresholds." When the band is Very low/Low and the 0.35 label is Placed, add: "A 'Low' band can still be 'Placed' at the 0.35 threshold: the band describes the size of the probability, the classification applies a cut-off."
8. **Rare-level notice** (only when Projects = 0): neutral notice naming the field and "22 of 8,425 students". Workshops = 3 (n = 246) is **not** flagged.

**Stale state:** see §9.

### 5.4 The threshold ruler (replaces the radial gauge everywhere)
A single static horizontal instrument, HTML/CSS only.

| Element | Specification |
|---|---|
| Track | Full card width minus padding; 14 px high; fill `--tint`; 1 px outline `--control-border` (3.07:1 on white) so its extent is perceivable; 2 px radius (instrument, not pill) |
| Scale | 0 % at the left end, 100 % at the right end (labels 13 px, muted). Minor ticks every 10 %; the 50 % tick is long and coincides with the 0.50 marker. No other numeric labels on the scale |
| Fill | **Navy `--brand`** from 0 to p; flat end-cap; **no gradient, no glow, no text label at the fill end** (a label there would collide with the 0.35 marker when p ≈ 0.35) |
| Markers | Two vertical ticks crossing the track, 2 px `--ink` with a 1.5 px white halo (so they stay visible over the navy fill — ink on navy is only 1.8:1). A **filled ◆** at 0.35 and a **hollow ◇** at 0.50 sit above the track with the short numerals "0.35" and "0.50" (13 px). Positions are exactly 35 % and 50 % of the track |
| Disagreement region | The span 35 %–50 % is drawn as a **45° hatched** segment of the track (hatch behind the fill) **and** as a bracket under the scale labelled "Labels differ in this region". Shape + text, never colour alone. Always visible, whatever p is |
| Legend row | Beneath the ruler, wrapping on phones: "◆ 0.35 (35%) project threshold · ◇ 0.50 (50%) conventional threshold · ▨ labels differ here". Long names live only in the legend |
| Text equivalent | One sentence adjacent to the ruler, by case. p < 0.35: "Estimated probability 12.0%: below both the 0.35 and 0.50 thresholds." 0.35 ≤ p < 0.50: "Estimated probability 37.3%: above the 0.35 threshold, below the 0.50 threshold." p ≥ 0.50: "Estimated probability 63.0%: at or above both thresholds." (numbers illustrative) |
| Phones | Only the numerals "0.35"/"0.50" sit at the markers (≈ 28 px wide each; markers are ≥ 40 px apart even at 320 px); everything longer is in the legend; the ruler never scrolls horizontally |
| Accessibility | Decorative parts `aria-hidden`; the text equivalent carries the meaning; `role="img"` + `aria-label`, or `<meter>`, only if Streamlit's sanitiser allows (§14) |

Forbidden here: arcs, dials, needles, conic/radial gradients, dotted rings, particle effects, count-up, pulsing, a second probability graphic, a confidence band.

### 5.5 D — "What shaped this estimate" (tabs in the result pane)
- **Factors (default).** Three groups with these exact headings: *Factors that raise the model's estimate* (blue), *Factors that lower the model's estimate* (orange), *At the baseline (no contribution)* (text list, e.g. "Internships = 1 (reference level 1)"). **Every one of the ten features appears exactly once**, so the rows reconstruct the total. Each row: label; subtext comparing value with baseline ("Workshops / Certifications = 1 (reference level 0)", "HSC Marks = 73 (training average 74.4)"); a diverging bar around a centre line (end ticks echo the ruler); the **log-odds contribution with sign** (`+0.97`) **and** the **odds multiplier** (`×2.65`). Bar length is relative to the largest contribution in this prediction. Text sign + group heading carry direction in addition to colour. Orange appears as text only in `--lower-ink`.
  Always-visible footnote: "Contributions are measured against a baseline student — training-average scores and reference categories. Bar length is relative to the largest contribution in this prediction. These are associations learned by the model, not causal effects."
  One-line aid: "An odds multiplier of ×2.65 means the odds of 'Placed' are multiplied by 2.65 relative to the baseline, within the model."
  Do not rank by "importance", do not suggest what to change.
- **Full calculation.** Identity once above the table: "log-odds = intercept + sum of (transformed value × coefficient); probability = 1 / (1 + e^−log-odds)". Table: baseline intercept → ten rows (contribution, odds multiplier) → total log-odds → probability.
- **Inputs used.** Vertical key–value list of the ten scored values (two columns on desktop, one on phones). Replaces the 10-column table.

### 5.6 E — About the model (full-width tabs: Method · Evaluation · Thresholds · Limitations and data)
1. **Method.** Logistic regression, L2 regularised (C = 1), no class weighting; scores standardised, counts and Yes/No one-hot encoded; trained on 8,425 students (train + validation) from a public dataset (Hugging Face `Krooz/Campus_Recruitment_CSV`, CC0-1.0); raw probabilities, no recalibration. A collapsed *Technical details*: "Model file verified by SHA-256 fingerprint 58e8b04e9666b13b…".
2. **Evaluation.** Title exactly **"Results from this project's final test evaluation (1,487 students, evaluated once)"**. A quiet **key–value metric list** — four rows, *not* dashboard tiles, no progress bars, no icons, no trend arrows: ROC-AUC **0.886** (95% CI 0.868 to 0.902) — "how well the model ranks placed students above non-placed ones; this is not accuracy"; Log loss **0.420** — "penalises confident wrong probabilities; lower is better"; Brier score **0.134** — "mean squared error of the probabilities; lower is better"; Calibration error (ECE) **0.016** — "average gap between predicted and observed rates across ten groups of students; lower is better". Caveat: "These describe performance on one held-out sample from one dataset. They do not guarantee performance on other students, campuses or years." The 95% CI belongs to the evaluation metric (project bootstrap); it is **not** an interval for any single prediction.
3. **Thresholds.** Table, same test set — Accuracy 0.802 (0.35) vs 0.812 (0.50); Precision 0.722 vs 0.779; Recall 0.851 vs 0.765; F1 0.781 vs 0.772; Specificity 0.766 vs 0.846. Optional counts (TP/TN/FP/FN): 526/666/203/92 vs 473/735/134/145. Sentence: "0.35 finds more placed students but raises more false alarms. Neither threshold is universally better: the right cut-off depends on the relative cost of false positives and false negatives, which this project has not defined." Becomes stacked key–value lists on phones.
4. **Limitations and data** (six bullets): associations, not causation; one observational dataset whose structure is consistent with a rule-based or constructed origin (not established by the source), so real-world equivalence is not shown; a probability describes a group of similar students, not a certainty (roughly 1 in 5 held-out students misclassified at either threshold); supported ranges are those observed in development data and out-of-range values are rejected, never adjusted; Projects = 0 is rare (22 of 8,425) and its estimate is unreliable; small under-prediction in the top decile (observed 0.946 vs predicted 0.905) was not recalibrated. Source line: "Dataset: Hugging Face Krooz/Campus_Recruitment_CSV (CC0-1.0)."

All figures in §5.6 exist in `README.md` / `src/phase7_findings.md` / `model_metadata.json` (checked 2026-10-17); they are static strings, never recomputed by the app (D4).

## 6. Visual system

### 6.1 Colour (light only — D/I)
| Token | Hex | Use | Contrast (WCAG 2.x, from hex) |
|---|---|---|---|
| `--canvas` | `#F6F8FB` | page background | — |
| `--surface` | `#FFFFFF` | cards, inputs | — |
| `--tint` | `#EEF3F8` | ruler track, table header, subtle panels | ink 14.95 |
| `--ink` | `#0B1F33` | body text, headings, markers | 15.69 on canvas · 16.69 on white |
| `--ink-muted` | `#475467` | captions, band tag, secondary text | 7.69 on white · 6.89 on tint |
| `--brand` | `#0F4C75` | primary button, **ruler fill**, hero numeral, links, focus ring | 9.09 on white; white on it 9.09; vs `--tint` 8.1 |
| `--brand-hover` | `#0B3B5C` | button hover/active | white text 11.70 |
| `--raise` | `#0072B2` | contribution bars "toward Placed" (graphics only) | 5.19 on white |
| `--raise-ink` / `--raise-tint` | `#0A5C7A` / `#E6F2F8` | "Placed" chip, raise-group header text | 6.52 on tint |
| `--lower` | `#D55E00` | contribution bars "toward Not Placed" (graphics only, **never text**) | 3.87 on white |
| `--lower-ink` / `--lower-tint` | `#B54708` / `#FFF4E5` | "Not Placed" chip, lower-group values | 4.99 on tint · 5.43 on white |
| `--error` / tint | `#B42318` / `#FEF3F2` | validation only | 6.57 / 6.05 |
| `--control-border` | `#8A94A3` | input, option-group and ruler-track borders (on white) | 3.07 on white |
| `--line` | `#DDE4EC` | decorative card hairlines | decorative only |
| `--focus` | 2 px `--brand` outline, 2 px offset | every interactive element | 9.09 |

Amendment to v1: the meter/ruler fill is **navy**, so blue/orange keep one meaning only — direction of a contribution or class. Rules: never colour alone (sign + heading + text/shape); no red/green for classification; red is for input errors only; band is neutral.
**Dark theme: out of scope for release 1.** No dark tokens are specified. The Stitch project's dark "Placement Intelligence" design system must not be used.

### 6.2 Typography (F)
- **Display serif — hero numeral and H1 only.** An open-licence (SIL OFL) editorial serif with **lining figures** and a weight of 500–600. First choice **Newsreader**; alternatives **Source Serif 4**, then **Fraunces** (with the "wonk" axis off). Fallback stack: `Georgia, "Times New Roman", serif`. Letter-spacing −0.01em. If the chosen face defaults to old-style figures, switch to lining figures or choose another — `37.3%` must read as evenly sized digits.
- **Everything else: Inter** (tabular numerals `tnum` in tables, lists, bars, field values) or the system sans stack. No monospace except the SHA-256 string in Technical details. No second display use of the serif (H2/H3, buttons, labels, tabs stay Inter).
- **Self-host or system fonts; no runtime CDN request** (D5, §12).

| Role | Size / weight |
|---|---|
| Hero numeral | serif `clamp(3.25rem, 9vw, 4.5rem)` = 52 → 72 px / 600, navy |
| H1 | serif 32 px desktop, 28 px phone / 600 |
| H2 section | Inter 20 px / 600 |
| H3 card title | Inter 16 px / 600 |
| Body | Inter 16 px / 400 |
| Secondary / caption | 14 px / 400; **floor 13 px for any text** |
| Eyebrow / labels | Inter 14 px / 600, sentence case, **no uppercase, no letter-spacing tricks** |

### 6.3 Surfaces, borders, shape (H)
White cards on `--canvas`; **1 px `--line` hairline; `box-shadow: none`** on cards, buttons, inputs, chips and tabs. Depth = tonal layering + hairlines. Allowed shadows: only the browser focus ring and Streamlit's own functional overlays (select menus, toasts) — never a decorative card shadow. Radius: cards 12, inputs/buttons 8, chips 999, ruler 2. No gradients, glass, blur or glow anywhere. Spacing on a 4-px base (4 · 8 · 12 · 16 · 24 · 32 · 48); card padding 24 (16 on phones). Left-aligned text, prose ≤ 70 characters wide, generous whitespace (this, with the serif and the ruler, is the "editorial" feel).
Icons: one outline set, 16–20 px, always beside a text label; used for error/notice states only.

### 6.4 Components (each specified with default / hover / focus / selected / error / disabled states in the Stitch design-system sheet)
Header · grouped field card · number input with hint · option group (segmented look, ≥ 44 px segments, native radio semantics) · choice group (No/Yes) · primary / secondary / tertiary buttons · classification card with chip · ruler · notice (info / caution / error) · diverging contribution row · tab bar · key–value list · data table · disclosure · toast · band tag.

## 7. Motion (E)
Principle: motion confirms that a **new result** arrived; it never decorates. **No animation, delay included, runs longer than 450 ms; the whole result sequence has finished within 450 ms of the result rendering.**

| Moment | Motion | Duration |
|---|---|---|
| New result appears | hero numeral opacity 0 → 1 (**no count-up**: intermediate numbers are probabilities the model never produced) | 150 ms ease-out |
| Ruler fill | width 0 → p, ticks and markers static | 400 ms, `cubic-bezier(.2,.8,.2,1)` |
| Contribution bars | grow from the centre line; stagger ≤ 20 ms per row, ≤ 10 rows | 220 ms each; last one ends ≤ 400 ms |
| Hover / focus / selection | border or background colour change | 100–120 ms |
| Tabs, disclosures | instant (or ≤ 120 ms opacity) | — |
| Errors, stale state, validation | **no motion** — appear immediately | — |

Rules: nothing loops or pulses; no shimmer, parallax, glow animation or spinner longer than 300 ms (prediction is near-instant; show a loading hint only if it exceeds 300 ms); animation plays **only for a freshly computed result**, not on reruns and not on the stale view. **`@media (prefers-reduced-motion: reduce)` sets `animation: none; transition: none`** so every element appears in its final state immediately. (v1 limited motion to ≤ 200 ms; E raises the ceiling to 450 ms and the ruler fill is the only 400 ms item.) Stitch cannot animate: the design-system sheet documents this table as annotations.

## 8. Layout and responsive behaviour
| Width | Behaviour |
|---|---|
| **≥ 1024 px** | Content ≤ 1200 px wide, centred; two panes 5/12 : 7/12 (balanced: dense controls on the left, airy result on the right), top-aligned; numeric fields in two-column sub-grids; E full width. Hero + ruler visible without scrolling at 1440 × 900 |
| **640–1023 px** | Single column; profile groups two-up where they fit; classification cards side by side |
| **< 640 px** | One column; every field full width; option groups wrap to two rows if needed; **Predict full width 48 px high**, Load and Clear stacked beneath at 44 px; classification cards stacked; contribution rows two-line (name + values, then full-width bar); tab bar scrolls horizontally; tables become key–value lists |

No horizontal page scroll at 320 px. Touch targets ≥ 44 × 44 px (native stepper size **to verify**; if poor, steppers may be hidden on touch widths and typed entry stays primary). No fixed/sticky bars. Streamlit's own column stacking breakpoint is 640 px; stacking at 1024 px needs a CSS media query (§14).

## 9. States and validation behaviour (G)
**Rule:** the UI never re-implements validation. Every message comes from `validate_inputs()` (single source of truth); `predict()` is called only when it returns no errors.

| State | Behaviour |
|---|---|
| **Empty** | All ten blank, nothing assumed, no errors shown, Predict enabled, designed empty result (§5.3) |
| **Typical profile loaded** | Fields filled; toast; any earlier result marked stale (not silently cleared) |
| **Field edited, valid** | Neutral; no success decoration required |
| **Field edited, invalid** (out of range, once the value commits) | Inline message immediately under that field, existing wording: "CGPA: 9.5 is outside the range the model was developed on (6.5 to 9.1); no prediction is made for out-of-range values." 2 px `--error` border + icon + text (never colour alone). Predict remains enabled |
| **Field empty** | No message until a Predict attempt; then "HSC Marks: a value is required." persists until the field is filled |
| **Predict pressed with any error** | Nothing is computed. A summary notice at the **top of the result panel** (first thing under the buttons on phones): "Please correct the following before a prediction can be made:" + one list item per problem (all missing fields in one list); inline messages on every failing field; previous result removed; panel shows no number |
| **Predict pressed, valid** | Result (§5.3, §5.5); motion per §7 |
| **Success, thresholds disagree / agree** | Note per §5.3 #5; chips differ / match |
| **Success, Projects = 0** | Rare-level notice; estimate still shown |
| **Stale** (any input changed since the last result) | Banner above the old result: "Inputs changed. Press Predict Placement Probability to update." Old result tagged **"From previous inputs"** and shown with an outline/hatched border and a muted tag — **never lowered opacity**; all text stays ≥ 4.5:1; changed fields carry a "Changed" tag with "Prior: 7.7 → Now: 8.2" style text. The old numbers are unaltered and never presented as current |
| **Computation error** | "The prediction could not be computed." + short cause; no stack trace; inputs kept |
| **Model load / integrity failure** | One calm full-width card: headline "The prediction model could not be loaded."; "The model file could not be loaded or did not pass its integrity check. No prediction is shown."; collapsed *Technical details* with the app's own message. No form, no tiles, no retry/replica/remediation claims (matches `st.stop()`) |

## 10. Accessibility (target WCAG 2.2 AA)
- Text ≥ 4.5:1; UI boundaries and graphics ≥ 3:1 (§6.1); type floor 13 px; reflow at 320 px; 200 % zoom without clipping.
- Real heading hierarchy (one `h1`, `h2` per section, `h3` per card); landmarks where Streamlit allows.
- Every field: visible label (not placeholder-only), visible range/option hint, error message adjacent in DOM order and naming the field and allowed range. Abbreviations (CGPA, SSC, HSC, ROC-AUC, ECE) expanded at first use in visible text; tooltips are never the only carrier of essential information.
- Keyboard: everything reachable, visible 2 px focus ring, order profile → actions → result → explanation → about; focus never hidden.
- Errors use the native alert role (`st.error`); results are announced via a polite live region only if the sanitiser allows `aria-live` (§14).
- Non-colour encoding everywhere (sign, heading, text, shape). Ruler/bars have adjacent text equivalents; decoration is `aria-hidden`.
- Motion per §7 including reduced motion.

## 11. Content truthfulness and data integrity
### 11.1 Fact register — the only numbers the design may show
| Fact | Value |
|---|---|
| Thresholds | 0.35 (project, selected by predeclared validation-F1 rule), 0.50 (conventional); Placed iff p ≥ threshold |
| Bands | < 20 % Very low · 20–40 % Low · 40–60 % Moderate · 60–80 % High · ≥ 80 % Very high ("estimated probability") |
| Training data | 8,425 students (train + validation); held-out test 1,487, evaluated once |
| Ranges | CGPA 6.5–9.1 · SSC 55–90 · HSC 57–88 · Aptitude 60–90 · Soft Skills 3.0–4.8 · Internships 0/1/2 · Projects 0/1/2/3 · Workshops 0/1/2/3 · Extracurricular No/Yes · Placement Training No/Yes |
| Rare level | Projects = 0 (22 of 8,425); rare-level threshold n < 50 |
| Evaluation | ROC-AUC 0.886 (95% CI 0.868–0.902) · log loss 0.420 · Brier 0.134 · ECE 0.016 · calibration slope 1.05 (optional detail) |
| Threshold comparison | §5.6 #3 values |
| Fingerprint | `58e8b04e9666b13b…` (first 16 hex characters) |
| Typical profile (example only) | CGPA 7.7, SSC 70, HSC 73, Aptitude 80, Soft Skills 4.4, Internships 1, Projects 2, Workshops 1, Extracurricular Yes, Placement Training Yes → **37.3 %**, band Low, **Placed @ 0.35, Not Placed @ 0.50**; intercept −1.66, total log-odds −0.52; contributions Placement Training +0.97 (×2.65), Extracurricular +0.72 (×2.06), Soft Skills +0.05 (×1.05), Aptitude +0.04 (×1.04), SSC +0.02 (×1.02), CGPA +0.00 (×1.00), Workshops −0.32 (×0.72), Projects −0.31 (×0.73), HSC −0.03 (×0.97); Internships at baseline |
| Misclassification | roughly 1 in 5 held-out students at either threshold |

Anything not in this table is a placeholder (`XX.X`), clearly marked, never a number that looks real. Mock screens that show result numbers carry a visible tag **"Example profile"** (the profile is hypothetical; the numbers are the real model output for it). The shipped product has no such tag.

### 11.2 Strings that must appear verbatim (also locked by `verify_app.py` where marked ★)
Buttons ★ "Predict Placement Probability", ★ "Load typical profile", "Clear profile". ★ `Model classification`, ★ `Conventional 0.50 threshold`, ★ `Factors that raise the model` (full: "Factors that raise the model's estimate"), "Factors that lower the model's estimate", "At the baseline (no contribution)", ★ `not a guarantee of placement` (disclaimer), ★ `outside the range` (error text), ★ `a value is required`, "Please correct the following before a prediction can be made:", "The threshold changes the label, not the probability.", the Evaluation title (§5.6 #2), "The prediction model could not be loaded."

### 11.3 Forbidden language and claims
"Compute…/Simulate…" as button labels; "optimal", "best", "recommended" or "default-correct" for a threshold; "likely", "at-risk", "marginal", "will be placed"; any per-prediction confidence/credible interval (the ROC-AUC CI is the only interval, §5.6); "SHAP", "feature importance", "attribution", "driver", "impact", "boost", "penalty", "improve", "increase your chances"; "calibrated" as a claim about one person's probability; "Platt", "posterior", "inference engine", "telemetry" and similar invented machinery; accuracy claims; ROC-AUC called accuracy; any peer/cohort comparison; anything implying the probability is a guarantee, a cause or a verdict.
Terminology: "operating threshold" → "project threshold (0.35)"; "reference threshold" → "conventional (0.50)"; "Phase 6/7" → "this project's final test evaluation"; "artifact fingerprint" → "Model file verified (SHA-256 …)" inside Technical details; "log-odds contribution" is kept and paired with "odds multiplier".

## 12. Implementation constraints (hold in any implementation)
1. **Frozen logic untouched:** `models/phase7/LR_B_final.joblib`, `prediction.py`, `explainability.py`, preprocessing, thresholds and `classify` are not modified. Presentation lives in `ui_helpers.py`, `app.py` and `.streamlit/config.toml`.
2. **`verify_app.py` locks** (*Audit §8.1*): widget keys = feature names; button labels; `<div class="hero-prob">{p:.1f}%</div>`; the ★ strings; empty-start; CGPA 9.5 / Aptitude 95 / HSC 40 → error and no hero; typical profile yields 0.35 ≤ p < 0.50. Dropping `st.form` (D1) is compatible; changing counts from `selectbox` to a radio group (D2) **requires the approved update of `verify_app.py` section 6 in the same change** (`at.selectbox(key=…)` → `at.radio(key=…)`), after which the full suite must pass.
3. **No clamping:** no sliders, no `min_value`/`max_value`; all values go through `validate_inputs`.
4. **No new data access:** the app reads only the model and `model_metadata.json`; never the test split or per-student test predictions; no network.
5. **New static content** (ECE row, threshold table, dataset source) is added as sourced static strings/metadata, never recomputed.
6. **Streamlit-feasible only:** native widgets + sanitised HTML/CSS; no JavaScript, custom components, sticky/fixed layers, dials or hover-only content. The Stitch HTML is a **visual reference only** — its Tailwind/Google-Fonts links and incidental scripts are not shipped.
7. **No runtime CDN** (fonts or scripts); the display serif and Inter are self-hosted (Streamlit `fontFaces` + static serving) or fall back to system stacks.
8. **Regression:** `python app/verify_app.py` passes in full before and after (baseline 122/122), plus the manual checks in `design_review_checklist.md` §10.

## 13. Decisions D1–D7 — resolved (2026-10-17)
| # | Original question (brief v1 §13) | Resolution | Rationale | Affected sections |
|---|---|---|---|---|
| D1 | Drop `st.form` (enables inline validation and a stale-result flag) vs keep it? | **Drop it.** Keep an explicit Predict button; prediction is never live | Decision G requires inline validation with Predict always available; a form freezes values until submit, so inline errors and "from previous inputs" are impossible. Logic is untouched; `verify_app.py` interactions are unchanged | §5.2, §5.3, §9, §12.2 |
| D2 | Counts as visible option group instead of `selectbox` (needs `verify_app.py` update)? | **Yes**, as a horizontal `st.radio` (`index=None`) styled as a segmented group; `st.segmented_control` only if AppTest supports it. The `verify_app.py` section 6 update ships in the same implementation change | Visible 0–3 choices are faster, give 44 px touch targets (J/mobile goal), and reuse the widget class already used for Yes/No, so the test change is mechanical | §5.2, §8, §12.2, §14 |
| D3 | Two-pane vs stacked desktop? | **Two-pane 5/12 : 7/12**, stacked fallback if the prototype fails | The brief asks for balanced input and result panels with the result visible beside the form | §3, §8 |
| D4 | Add ECE row, threshold trade-off table and dataset source as static content? | **Yes**, documented values only (§11.1), as static strings or metadata fields with the source cited | They are already in `README.md`/`phase7_findings.md` (checked); no recomputation, no new data access | §5.6, §11.1, §12.5 |
| D5 | Self-hosted Inter vs system stack (and display face)? | **Display serif for hero numeral and H1 only; Inter (self-hosted) or system sans for everything else; no CDN.** Order: Newsreader → Source Serif 4 → Fraunces → Georgia | Decision F allows a display face for the hero; confining it to two roles keeps controls and body text maximally readable and the page restrained | §6.2, §12.7 |
| D6 | Optional "Clear profile" button? | **Include** as a tertiary button | With empty-start, no form and stale-result logic, users need a one-click reset; it adds no new logic and no `verify_app.py` lock | §5.2, §9 |
| D7 | Streamlit minimum-version pin? | **Target and test on 1.64.0**; raise `requirements.txt` to the version the implementation is verified on (e.g. `streamlit>=1.64`) in the implementation change, not now | Theme keys (`fontFaces`, radius, border) and widgets were confirmed only on 1.64.0; the 122-check baseline ran on it. Any key missing at a lower version is a defect | §12.8, §14 |

Resolved by the authoritative direction without new questions: stale-result handling (stale, not cleared), the form-vs-no-form trade-off, motion ceiling (E), shadows (H), meter colour (D), theme (I), gauge (C, J).

## 14. Open technical items (verify in a prototype; none block Stitch generation)
1. Streamlit HTML sanitiser: `aria-*`, `role`, `<style>` with `@keyframes` and `@media (prefers-reduced-motion)`.
2. CSS media query to stack the columns at < 1024 px (Streamlit's default breakpoint is 640 px) and `layout="wide"` with a 1200 px cap.
3. Nested columns + tabs depth in the result pane.
4. Native stepper touch-target size and behaviour of the first +/− press on an empty field; fall back to hidden steppers on touch widths.
5. `st.radio(horizontal=True, index=None)` styled as a segmented group; AppTest support if `st.segmented_control` is preferred.
6. Red border on an invalid number input via the `st-key-<widget key>` container class.
7. Self-hosting fonts: `[[theme.fontFaces]]` plus static serving; confirm lining figures in the chosen serif.
8. Animation triggers only on a newly computed result and does not replay on rerun.
9. Theme keys and `st.toast` at the pinned minimum version (D7).

## 15. Explicitly excluded
- Personal or identifying data (name, roll number, e-mail, photo, avatar, college, free text); accounts, login, saved profiles, **prediction history**, export or sharing, analytics beyond Streamlit defaults (usage stats are off), databases, APIs, background jobs.
- Batch/CSV upload, ranking, peer percentiles, leaderboards, cohort dashboards, "similar students".
- Improvement advice, tips, recommended actions, counterfactuals, **what-if sliders or simulators**.
- Any per-prediction interval or uncertainty not produced by the app; any invented metric, chart, distribution or feature-importance display.
- Radial gauge or any dial; dark theme; sidebar; top navigation or tab bar; status strips; version/seed/latency/trace badges; KPI tiles with progress bars; gradients, glass, glow, decorative shadows; uppercase or monospace micro-labels.
- Threshold labels "Recommended/Optimal/Best"; accuracy claims; celebratory or alarming treatment of the classification.
- Chatbot, assistant, notifications, social sharing, gamification; retraining, recalibration, alternative models, changing thresholds from the UI; multi-page navigation; multilingual UI.

**Section map from version 1 (for older cross-references):** v1 §3 → §3 · §5 → §5 · §6–§7 → §2, §6 · §8 → §6.2–§6.4 · §9 → §8 · §10 → §9 · §11 → §10 · §12 → §12 · §13 → §13 · §14 → §15.
