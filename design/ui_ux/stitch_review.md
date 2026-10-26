# Stitch Review — "Placement Probability Analytics Workspace"

**Project:** `projects/16130822196033598061` (Stitch, `TEXT_TO_UI_PRO`, last updated 2026-10-13)
**Reviewed:** 2026-10-14
**Status:** review and recommendations only. Nothing in Stitch was edited, no variants were generated, no application code, model, data, notebook, report or `requirements.txt` was touched, and no generated HTML was placed in the repository.
**Update 2026-10-17:** §0–§11 below are the historical assessment written on 2026-10-14 and are **not revised** — the old seven screens did not implement the brief. The decisions this review left open (§10 A–J, brief D1–D7) are now **resolved in §12**, and the corrected specification is `design_brief.md` v2. Section numbers cited as *Brief §n* in §0–§11 refer to brief **v1**; the brief's closing section map converts them.
**Reference documents:** `current_app_audit.md` (cited *Audit §n*), `design_brief.md` (*Brief §n*), `stitch_prompt.md`, `design_review_checklist.md` (*Checklist n.n*).

---

## 0. Verdict

**Do not edit the existing screens. Generate a new set, from a tightened prompt, on top of the project's own *light* design system, and carry over only the structural and copy ideas listed in §5.**

The seven screens look finished, but they do not implement the brief. In short:

1. **They do not implement the written brief.** The brief and the Stitch project's own design document (`designTheme.designMd`, "Clinical Analytical Inference", light, navy `#0F4C75`, blue/orange pair) call for a calm light instrument. Every screen was rendered as a **dark "obsidian / electric cyan / glass" theme** with neon glow, a radial gauge and monospace "terminal" labels. The project's theme tokens and the screens' actual Tailwind tokens are different palettes (§2). Dark theme is out of scope for release 1 in the brief (Brief §7, Checklist 2.8). **Context:** the project also holds two reference images uploaded by you (a dark glassmorphism SaaS dashboard and a neon radial "67%" gauge, §1), which very likely steered Stitch to this look. So the theme is a genuine open decision rather than a Stitch error alone (§10, decision I).
2. **Every screen contains fabricated or wrong content**, including a per-prediction 95% CI, wrong evaluation numbers, an invented "Platt-scaled" calibration, invented ranges and fields (Backlogs, Hackathons), invented infrastructure and a "live parameter test bench" (§4). Several Checklist **[BLOCKER]** items fail on every screen.
3. **It breaks hard constraints:** sliders (mobile), clamping steppers with the wrong ranges, pre-filled fields, wrong primary-button label, multi-page navigation, an avatar, JavaScript behaviour, CDN dependencies.
4. **It is the generic AI-dashboard look**, not a distinctive one: dark navy + cyan glow + radial gauge + uppercase mono micro-labels + pseudo-technical jargon.

The **structure and copy** of a few states are good and worth keeping (stale state, validation summary, mobile result copy, evaluation tiles, two-pane desktop result). Because the *visual layer, content layer and component set* all need to change, an edit pass would be a rewrite; a fresh generation from a stricter prompt is cheaper and safer (§9).

---

## 1. What was inspected

Retrieved with `list_screens` and `get_project`; screenshots and HTML for all seven screens were downloaded to the session scratchpad (not the repo) and read. Screenshots were read visually; HTML was analysed statically (tokens, markup, scripts, class names). **Nothing was run in a browser**, so interactions are judged from the HTML/JS, and contrast ratios are computed from the Tailwind colour tokens in the HTML, not measured on pixels.

| # | Stitch title | Device flag | Frame (px) | Inspected |
|---|---|---|---|---|
| 1 | Placement Intelligence – Analytical Prediction Workspace (**main**) | DESKTOP | 2560 × 3788 | screenshot + HTML |
| 2 | Placement Intelligence – Design System Specification | DESKTOP | 2560 × 7894 | screenshot + HTML |
| 3 | Mobile – Profile Form and Result State | MOBILE | 780 × 6956 (390 @2×) | screenshot + HTML |
| 4 | Mobile – About the Model Evaluation | **DESKTOP** | 2560 × 2212 | screenshot + HTML |
| 5 | Placement Intelligence – Validation Error State | DESKTOP | 2560 × 2844 | screenshot + HTML |
| 6 | Placement Intelligence – Stale Result State | DESKTOP | 2560 × 3642 | screenshot + HTML |
| 7 | Placement Intelligence – Model Load Failure State | DESKTOP | 2560 × 2130 | screenshot + HTML |

Notes:
- **Screen 4 is titled "Mobile" but is a 1280-px-wide desktop frame** (mobile layout stretched across it), so its mobile behaviour cannot be verified. It must be regenerated at 390 px.
- **The mobile form + result is the only mobile frame**; there is no mobile validation, stale or error state, and no mobile design-system view.
- The project also holds ~10 *hidden* screen instances (earlier iterations) that `list_screens` does not return. Not reviewed, except the two below.
- **Two uploaded reference images** (`image.png`, screens `5332846485232339791` and `5332846485232338709`, read-only `get_screen`): (a) a dark "Glassmorphism Dashboard UI" template for a generic SaaS analytics product (KPI tiles, area charts, donut, glowing panels); (b) a neon radial gauge reading "67%" with a dotted particle ring, plus a dark amber-gradient hero card. These match the dark glass look, the glowing radial gauge and the dotted ring on the generated screens. Whoever uploaded them wanted a premium, dark, glassy feel; the generated output follows that, not the written brief.
- Not present at all: a desktop **empty first-load state** (required state 1 in the prompt) and a desktop **"About the model" tabbed section** (replaced by a collapsed accordion).

**Review order followed:** main workspace → design system → mobile layouts → error/stale/load-failure states.

---

## 2. Cross-cutting findings (apply to every screen)

### 2.1 The screens do not use the project's design system

`get_project` returns `designTheme.designMd` = **"Clinical Analytical Inference", `colorMode: LIGHT`**, canvas `#F6F8FB`, surface `#FFFFFF`, brand `#0F4C75`, blue `#0072B2` / orange `#D55E00`, hairline `#DDE4EC`. The generated HTML instead ships its own Tailwind config: `background #0c1421`, `surface-container #18202d`, `primary-container #00f2fe` (electric cyan), plus `JetBrains Mono` for "data" text. The design-system screen itself calls this "Dark Obsidian Mode · Cybernetic Cyan Accent · Glass Tactical Dark · glassmorphism interaction states". So the Stitch project currently contains **two contradictory design systems**, and the screens follow the one that matches the uploaded dark reference images (§1) rather than the written brief. Which of the two the product should follow is decision I in §10.

### 2.2 The project's own design document contains traps

`designMd` is not clean either and must be fixed before regenerating (it may also have encouraged some of the fabrication — plausible, not proven):

| In `designMd` | Conflicts with | Fix |
|---|---|---|
| "**Sliders**" component spec | No sliders: they clamp (Brief §5.2, Checklist 1.4) | delete |
| "**Sidebar Allocation** … parametric feature-input dock (`st.sidebar`)" | Brief §3.3/D3: two-pane workspace, no sidebar | delete or decide explicitly |
| "direct **confidence intervals** rather than theatrical…" | No per-prediction CI (Brief §14, Checklist 1.11) | delete |
| `body-sm` 12 px, `label-sm` 11 px | 13 px floor (Brief §8, Checklist 3.4) | raise to 13 px |
| "avoids ambient drop shadows" | Brief §8: hairline + very soft shadow | pick one (recommend hairline-only; drop the shadow) |
| "SHAP / Feature Attribution" wording | The explanation is exact additive log-odds, not SHAP | use "contribution" |

### 2.3 Content fabrication is systemic (details in §4)

The prompt said "Do not fabricate … use a placeholder such as XX.X". The screens invent metrics, system identifiers, fields, ranges and behaviours on **all seven** screens.

### 2.4 Hard-constraint violations (Brief §12, Checklist §5, §8)

| Constraint | Finding | Where |
|---|---|---|
| Light theme only | Dark everywhere | all |
| No sliders / no clamping | `<input type="range">` for Aptitude and Soft Skills; desktop steppers clamp via `adjustValue()` to **wrong** limits: CGPA 6.0–10.0, HSC 50–100, Aptitude 40–100, Soft skills 1.0–5.0 (validation screen: HSC 40–100, Aptitude 0–100) — different from the supported 6.5–9.1 / 57–88 / 60–90 / 3.0–4.8 | main, mobile form, validation |
| All ten fields start empty | Every screen except validation is fully pre-filled with the typical profile; validation shows invented values (Hackathons 3, Aptitude 78) | all |
| Exact labels `Predict Placement Probability` / `Load typical profile` | Primary button is **"Compute Placement Probability"** (also "Compute Probability", "Simulate Score"); "Load typical profile" appears only on the mobile form; desktop uses "Baseline" and "Typical". String `Predict Placement Probability` appears on **no** screen (Checklist 8.1 BLOCKER) | main, stale, valerr, design |
| Single workspace page, no navigation | Top nav (Prediction Workspace / Evaluation Metrics / Methodology & Limitations) + **avatar/profile icon**; mobile fixed bottom tab bar (Predictor / Metrics / **Data** / Model) + kebab menu + "ONLINE" status (Checklist 2.1, 2.6) | all |
| No JavaScript behaviour / sticky layers | Inline JS for steppers, segmented toggles, compute click, "JS Evaluator Active" bench; fixed bottom bar | main, design, mobile |
| No runtime CDN | Tailwind CDN, Google Fonts (Inter, JetBrains Mono, Material Symbols) on all screens (Brief §12.8) | all |
| Predict stays enabled and refuses (Brief §4) | Validation screen **disables** the button ("Inference Blocked — Correct Errors") | valerr |
| No what-if / simulator | "Live Parameter Test Bench" with a made-up logit formula and "Simulate Score" | design |

### 2.5 Accessibility, from the HTML (Brief §11)

| Item | Finding |
|---|---|
| Headings | `<h1>` absent on main, stale, validation and load-failure screens (present on design, mobile form, mobile about). No heading structure on the gauge/result. |
| ARIA / roles | 0–4 `aria-` attributes per file, **no `role`**, no `<meter>`; the radial gauge and bars have no text alternative beyond adjacent text. |
| Labels | 5 numeric inputs but 1 `<label>` per desktop screen → inputs not programmatically labelled. |
| Reduced motion | No `prefers-reduced-motion` rule anywhere. Infinite `animate-pulse` / `animate-ping` on status dots (looping motion; Brief §11 forbids). |
| Type floor | `text-[9px]`, `[10px]`, `[11px]`, `[12px]` utilities are present on most screens (floor is 13 px). Most micro-labels are uppercase, letter-spaced mono. |
| Text contrast (computed from tokens) | Main text passes comfortably (`#dbe2f5` on `#0c1421`, `#b9cacb` on `#18202d` = 9.6:1, muted `#849495` on `#18202d` = 5.2:1, cyan `#00f2fe` on `#0c1421` = 13.3:1). |
| **Non-text contrast** | Card/control borders `#3a494b` on `#18202d` = **1.74:1**; bar tracks `#2d3543` on `#18202d` = **1.33:1** — both below the 3:1 needed for UI boundaries and graphics. Input borders are `outline-variant` at 40% opacity (mobile) — lower still. |
| **Stale dimming** | Stale screen uses `opacity-75/50/30`: body text at 50% ≈ **4.2:1**, at 30% ≈ **2.4:1** — the "previous result" becomes partly unreadable. Dimming must stay ≥ 4.5:1 for text. |
| Colour alone | Mostly OK: signs, group headings and text accompany colour in factor rows. But the desktop `Not Placed` chip is *grey* while `Placed` is cyan (de-emphasis rather than neutral peer), and the band tag reuses the orange of "toward Not Placed" (§3.1). |
| Meter labels overlap | The overlapping tick-label bug flagged in Audit §11 **is reproduced on the main screen**: "Profile (37.3%)" sits on top of the 0.35 label, next to "0.50 Std". |

### 2.6 Goal scorecard

| Goal | Rating | Basis |
|---|---|---|
| Distinctive, premium identity | **Fail** | Visually dramatic, but the *archetypal* generic "AI dashboard" look (§2.7). Not distinctive for this product. |
| Less generic AI-dashboard styling | **Fail** | Radial gauge, neon glow, glass, gradient CTA, uppercase mono labels, pulsing status dots, fake "system" telemetry. |
| Attractive but restrained motion | **Fail** (not designed) | No motion spec; only decorative pulse/ping loops, glow and `scale-95` on click. No reduced-motion handling. |
| Minimal unnecessary information | **Fail** | 3 header rows (status strip + chips + seed/eval-set/cut-off), top nav, avatar, footer telemetry, 10+ system labels, duplicated ranges, a "Comparative Benchmark Segment" and posterior tiles (§4). |
| Placement probability as main focus | **Partial** | 37.3% is the largest element on main and mobile; but it is wrapped in a decorative radial gauge with extra numbers (`p = 0.3734`, `logit −0.518`, a CI) competing for attention. |
| Distinguish probability / band / 0.35–0.50 classifications | **Partial** | Content is present and mostly well worded (esp. mobile). Visual encoding is inconsistent and the band is colour-coded like a classification (§3.1). |
| Accessible contrast / responsive | **Partial / Fail** | Text contrast fine; non-text contrast, type floor, heading/label semantics, reduced motion, overlapping labels and the unverifiable mobile frames fail. |

### 2.7 Why it reads as generic

Deep navy canvas, cyan glow on every active element, a circular progress-arc with a dotted outer ring, glassy cards with backdrop blur, uppercase letter-spaced monospace micro-labels, numbered section prefixes ("01 · ACADEMIC FOUNDATIONS"), invented system vocabulary ("Inference Core: L2_LOGIT_V2.1", "Candidate Feature Vector", "Epistemic Uncertainty Bounds Active", "Zero-surrogate policy enforced"), status dots and latency counters. These are the stock signals of a generated "ML dashboard"; they also signal engineering cosplay, which undermines a project whose credibility rests on restraint and honesty.

---

## 3. Screen-by-screen review

### 3.1 Screen 1 — Analytical Prediction Workspace (main desktop)

**Layout:** header + nav; status strip; left "Candidate Feature Vector" (three numbered cards + Compute button); right hero gauge, meter, two threshold cards, "Boundary discrepancy note" + band tag, "Attribution Decomposition", collapsed "Model Diagnostics & Rigor Specifications", academic-notice bar, footer.

**What works**
- **Two-pane desktop works**: the hero 37.3% is visible beside the form without scrolling (Brief D3 confirmed as a good direction).
- Three grouped input cards matching the brief's Academic/Technical/Employability grouping; counts as visible option buttons; Yes/No pairs; the Projects = 0 rarity note is present and correctly worded ("22 of 8,425").
- Meter has a plain-language sentence beneath: "Estimated probability 37.3%: above 0.35 project threshold, below 0.50 conventional threshold." Good text equivalent.
- Two equal threshold cards plus a "discrepancy note" explaining why the labels differ (`probability is the primary continuous metric`).
- Factor rows carry label, sign, log-odds **and** odds multiplier (e.g. `+0.97 (×2.65 odds)`), grouped raise / lower; the rows that are shown (Placement Training, Extracurricular, Soft Skills, Aptitude; Workshops, Projects, HSC) match the audit (§9 of the audit). **The set is incomplete:** SSC +0.02 and CGPA +0.00 are missing from the raise group and there is no "At the baseline (no contribution)" group (Internships = 1), so the rows do not reconstruct the total log-odds.
- Active option state is unambiguous; Inter + tabular figures for the hero.

**What feels generic, cluttered or unnecessary**
- Neon radial gauge with dotted ring and glow; the arc encodes probability by angle, which is harder to read than position on a scale, and it cannot carry two threshold markers — so a *second* linear meter is added underneath (two probability graphics for one number).
- **Header overload:** three rows before any content (frozen/SHA badge, dataset badge, nav, profile avatar, then "Inference core / Decision matrix calibrated (ECE 0.016) / Optimal F1 cut-off 0.350 / Seed / Eval set"). None of this helps a first-time reader; most of it is invented.
- Top navigation + avatar (out of scope) and a footer of "Latency / Accuracy / ROC-AUC".
- Jargon: "Candidate Feature Vector", "Inference Stream", "Statistical Attribution Analytics", "Threshold ALPHA / BETA", "Pop Mean", "Benchmark target", "Scored Deterministic".
- Compute CTA is a glowing cyan gradient bar — loudest element after the number, competing with it.
- A CGPA progress bar under the field (`value/10`) implies a score-out-of-10 scale that has nothing to do with the supported range.

**Fabricated / incorrect (BLOCKERS)** — see §4 for the full table: `95% CI [0.321, 0.428]`; `OPTIMAL F1 CUT-OFF` (the word "optimal" is forbidden, Checklist 1.7); `EVAL_SET: N=1,685 (20%)` (true: 1,487); `Sensitivity 89.2% / Specificity 71.4%` and `74.1% / 88.6%` (true recall/specificity: 0.851/0.766 and 0.765/0.846); footer `Accuracy 94.2% · ROC-AUC 0.961` (true ROC-AUC 0.886, accuracy 0.802/0.812); `p = 0.3734 · logit −0.518` (the app's value is 0.3730 / −0.5195); `Seed #0x8F94B1`; `Latency 14 ms`; button "Compute …"; fields pre-filled; wrong stepper limits.

**Hierarchy / three-way distinction**
- Probability: strong. **Band** ("Low Prob (20%–40%)") is rendered in amber — the same hue as "toward Not Placed"/"lowering" — so it reads as a *negative classification*, the exact conflation the brief wants to avoid. **Classification chips**: "Classified: Placed" in cyan, "Not Placed" in grey — unequal weighting; no shape glyph.
- The two cut-off cards are below the meter and partly below the fold (card titles cut at the scroll line in the screenshot), so the disagreement is not understood at first glance.

**Accessibility / responsive:** overlapping meter labels (§2.5); no heading `h1`; mono 10–12 px labels everywhere; 1.7:1 card borders; gauge has no text alternative.

**Recommended changes:** replace the radial gauge by a single horizontal probability ruler with both cut-offs (§7); collapse header to title + three badges + 3-step strip; remove nav/avatar/footer telemetry; recolour band neutral; use exact strings "Model classification — at the 0.35 project threshold" / "Conventional 0.50 threshold"; rename CTA to **Predict Placement Probability**; start empty; use only supplied ranges.

### 3.2 Screen 2 — Design System Specification

**What works**
- It contains a useful set of *specimens* worth recreating: colour tokens with roles, type scale including hero 72 px/H1 32/H2 20, state variants (default / focus / range violation), button hierarchy (primary / secondary / ghost), geometric class chips ("Square mark"/"Circle mark" — a good neutral-shape idea), a diverging attribution bar, an error callout and a methodology callout.
- Stepper focus and error states are shown (cyan border; red border + icon + message — not colour-only).

**What is generic / wrong**
- Documents the **dark glassmorphism** theme; explicitly "Glass Tactical Dark", `0dp Elevation`, glow tokens. Not the project's design system (§2.1).
- Specimen content is invented and sometimes impossible for this product: fields **Backlogs, Internships (0–5), Academic Stream (Computer Sci / IT / Electronics)** that do not exist in this model, a "range violation" demo on "Backlogs Count 14 > 10 max", and CGPA example values (8.42, 8.80) unrelated to the typical profile; `Z = +2.44`; `+1.601 logit` with "Interquartile range 1.12"; hero `0.7842` / gauge `78%`; "SHAP Decomposition"; "Singular Matrix Inversion Failure / multicollinearity / Convergence … Reverting automatically to cohort prior weights" (the app does none of that; it never falls back to anything); "Department of Applied Statistics".
- **"Live Parameter Test Bench"** with a fabricated equation `logit = −4.2 + 0.85 × CGPA + 0.62 × Intern − 1.10 × Backlogs`, "Likely Placed (◆)", "JS Evaluator Active", "Simulate Score": a what-if simulator with invented coefficients — explicitly out of scope (Brief §14, Checklist 2.5) and potentially misleading (made-up coefficients next to a real model).
- Hero metric shown as `0.7842` in 72 px mono — a probability shown as a decimal, while the app shows a percentage with one decimal.
- "WCAG AAA / AA Validated" claim: true only for text on the dark canvas; borders and tracks fail 3:1 (§2.5).
- "Likely Placed" is a new, predictive-sounding label that the product must not use.

**Recommended changes:** regenerate as a *light* token sheet from the brief palette (§7), with only the specimens listed above; use real fields for state demos (CGPA, one option group); show the three-tier output hierarchy (probability / cut-off labels / band) as components; drop the simulator, JS, SHAP, glow and glass specimens.

### 3.3 Screen 3 — Mobile: Profile Form and Result (390 px)

This is the **closest screen to the brief**, particularly the result copy.

**What works**
- Header copy and three badges are exactly the specified strings ("Estimate, not a guarantee", "Frozen model – logistic regression", "No data stored") and a 3-step strip.
- Result sequence matches the spec: "Example values" tag → hero 37.3% → "Model-based estimate … not a guarantee." → ruler with legend (◆ 0.35 project threshold / ◇ 0.50 conventional threshold) → "Summary: Estimated probability 37.3%: above the 0.35 threshold, below the 0.50 threshold." → disclaimer → **Model classification / Placed** card with the exact "chosen in advance … not claimed to be the best threshold" copy → **Conventional 0.50 threshold / Not Placed** → "The threshold changes the label, not the probability." → agreement note → band card → tabs *Factors / Full calculation / Inputs used* → exact headings "Factors that raise/lower the model's estimate" → "At the baseline (no contribution)" → odds-multiplier aid.
- Cards stack; tabs are horizontal; factor rows are two-line (name + value, then bar); primary button full width; secondary buttons paired.
- Segmented option groups for counts; separate Yes/No groups; Projects = 0 note.

**What fails / is generic**
- **Sliders** for Aptitude (`0–100`, shown "80 / 100") and Soft Skills (shown "1.0 – 5.0", slider 1.0–5.0) — wrong ranges, clamping control (BLOCKER).
- **Wrong levels:** Internships `0 1 2 3+` (true 0–2), Projects `0 1 2 3 4+` (true 0–3). Workshops uses a **dropdown** ("1 workshop") — inconsistent with the other two count fields. CGPA shown as a plain text box "7.7" with no steppers and no range hint; SSC/HSC "Score %" with no range hint; "CGPA (10pt)" label invented.
- All fields pre-filled; button label "Compute Placement Probability".
- **Ruler is wrong:** a gem icon at 0.50; the filled 0.35 marker exists only in the legend, not on the bar (Checklist 3.9 fails). Radial gauge with "ESTIMATED PROB" label clipped.
- **Factor values are different from the desktop screen for the same "typical profile" and are fabricated:** CGPA 7.70 (mean 7.00) +0.68 ×1.97, Internships +0.42, SSC −0.51, Aptitude −0.24, intercept −1.08, Extracurricular "No" at baseline. Real (Audit §9): Placement Training +0.97, Extracurricular +0.72, Soft Skills +0.05, Aptitude +0.04, SSC +0.02, CGPA +0.00, Workshops −0.32, Projects −0.31, HSC −0.03, Internships at baseline, intercept −1.661.
- Band card copy invents a claim: "quantitative performance features place them in the **lower quartile of predicted student distributions**" (no such analysis exists in the project) and calls 0.35 a "tuned sensitivity mark".
- "0.35 project threshold (**empirically derived**)", "0.50 conventional **balanced** threshold" — wording drift from the approved copy.
- Disclaimer appears *before* the classification cards, and the "threshold changes the label" note sits after them in a separate box; the agreement note is a third box. Three adjacent notices say related things.
- Fixed bottom tab bar (Predictor / Metrics / **Data** / Model), kebab menu, "ONLINE" status, header "v2.4-frozen"; a "Data" section does not exist.
- Mono uppercase badges (`3 attributes`, `1 completed`, `2 completed`), `10–12 px` labels, border `outline-variant/40`.

**Recommended changes:** keep the result copy and order; merge the three notices into one "How to read" block; show both cut-offs as shape-coded ticks on the ruler; remove sliders; correct ranges/levels; real contributions only; remove bottom nav; per-field range hint; numeric steppers (native Streamlit).

### 3.4 Screen 4 — "Mobile" About the Model (Evaluation) — actually a 1280-px frame

**What works**
- Title is exactly **"Results from this project's final test evaluation (1,487 students, evaluated once)"**; four tiles with the right headline numbers (ROC-AUC 0.886 with CI 0.868–0.902; log loss 0.420; Brier 0.134; ECE 0.016); the one-line plain meanings are correct, and "this is not accuracy" is included. "Generalization boundary" caveat copy is right. Pill tabs (Method / Evaluation / Thresholds / Limitations) are the right IA. "10 equal-mass bins" is accurate (the project's ECE uses 10 equal-count bins).

**What is wrong / unnecessary**
- Each tile has a **filled progress bar** whose scale is arbitrary (ROC-AUC bar ≈ 0.89 of full; log-loss bar ≈ 0.42 of the track; Brier ≈ 0.26) — it makes "lower is better" metrics look like scores out of 1 and invites false comparison across tiles. Remove (or replace with a labelled reference marker).
- **Undocumented baselines:** "**ref: 0.693 uninformative**" for log loss and "**null benchmark: 0.250**" for Brier. These are legitimate coin-flip values (ln 2 and 0.25 for always predicting 0.5), but they are not the project's documented reference, which is **0.679** and **0.243** for always predicting the training rate (`src/phase7_findings.md`). Showing a different baseline from the one in the report would give a viva panel two reference points. Use the project's values or drop the reference.
- "Fixed seed baseline (#0x8F94B1)", "Deterministic out-of-sample split", "Δ ≤ 1.6% empirical drift" (not a project quantity), "v2.4-eval", "L2_LOGIT_V2.1", "ECE 0.016" badge in the header.
- Same fixed bottom bar (Predictor / Metrics / Data / Model).
- Not a mobile layout (1280 px); mono type everywhere.

**Recommended changes:** regenerate at 390 px; four tiles without bars; delete invented references or use the project's 0.679/0.243 only if the reference note is wanted; Thresholds tab as stacked key–value rows; Limitations as the six agreed bullets; dataset source line (Hugging Face `Krooz/Campus_Recruitment_CSV`, CC0-1.0) — currently absent from every screen.

### 3.5 Screen 5 — Validation Error State

**What works (best state design in the set)**
- **Triple-layer error pattern is right in spirit:** red "2 Validation Errors" chip on the form header, inline messages under the field (icon + red border + text, not colour only), and a summary at the top of the result panel — "Please correct the following before a prediction can be made:" with a numbered list, using the **exact** sentences for `CGPA: 9.5 is outside the range the model was developed on (6.5 to 9.1); no prediction is made for out-of-range values.` and `HSC Marks: a value is required.`
- "Inference Blocked / No Diagnostic Output Generated" empty result panel makes it unambiguous that no number is shown; no stale result is shown.
- Field-level "Required" marker and `--` placeholder for the empty field.

**What is wrong / unnecessary**
- **Fabricated fields:** "History of Academic Backlogs" (No (0)/Yes (1+)), "Hackathons (0–12)", "Aptitude Score 0–100", "Internships 0–8", "HSC 40–100 %" — the screen shows a different data model from the app. The "Allowed domain" table below repeats these wrong ranges (CGPA 6.50–9.10 is right; HSC and Internships are wrong).
- Range text is shown **three times** for the same field (label caption "Allowed development range", inline caption "Allowed: 55 to 90 %", status row/table) plus the summary (Audit §10 #7 redundancy).
- Predict is **disabled** ("Inference Blocked — Correct Errors"); the brief keeps it enabled and explains (Brief §4 step 2).
- Jargon: "Guardrail intercept", "Extrapolation safeguard: ACTIVE", "Link function gradient saturation prevented", "Epistemic Safeguards", "Parametric Constraints".
- The "About the Model & Bounds" panel is out of place here.
- Pre-filled CGPA shows "9.50" (3 digits clipped in a 56-px box).

**Recommended changes:** keep summary + inline pattern and the wording; show errors for **real** fields only; one range hint per field; keep Predict enabled; remove the status table and jargon strips.

### 3.6 Screen 6 — Stale Result State

**What works (the strongest idea in the set)**
- Top banner in caution style: "**Inputs changed — press Compute Placement Probability to update.**" plus a one-line explanation, tagged "STALE STATE"; a "1 unsaved change" chip; the changed field carries a **MODIFIED** tag and "Prior: 7.40 → Now: 8.20"; the old result is dimmed and tagged "From previous inputs … Out-of-sync snapshot"; the primary button repeats which input changed. The old result is never presented as current. This answers Audit §10 P1 #1 directly.

**What is wrong / unnecessary**
- **Fabrication again:** `95% CI [0.318, 0.431]`; "Classification: **At-Risk / Marginal**" (new label); "Unplaced Posterior P(Y=0|X) 0.627 — Predicted majority label"; "**Comparative Benchmark Segment — similar profiles in validation split (N = 340), 39.1% historic cohort placement rate, Calibration delta −1.8%**" (a peer-comparison panel that is explicitly out of scope and invented); "**Platt-Scaled Sigmoid**" under Calibration Integrity (the model is **not** recalibrated; raw probabilities); "Engine Hash #9d2a014e" (real: `58e8b04e9666b13b…`); "Cached output frozen (#INF_88301B)", "Last eval: 4m ago", "EST_LATENCY 14ms"; the stale attribution list ("Communication Assessment (68/100)", "Capstone Technical Projects", "Model Bias Intercept −0.68", CGPA −0.42) matches neither the app nor the other screens.
- CGPA edited to 8.20 with an extra `+0.80 std dev` caption and a CGPA progress bar.
- Dimming via `opacity-30/50/75` drops body text below 4.5:1 (§2.5).
- The status strip repeats "DESYNC DETECTED · L2_LOGIT_V2.1 · Cached output frozen" — three overlapping stale cues on top of the banner and the result tag.

**Recommended changes:** keep the banner/MODIFIED/Prior→Now/"From previous inputs" pattern with the real button label; dim by lowering saturation and adding a hatch/outline instead of opacity (keep text ≥ 4.5:1); delete the strip, benchmark panel, posteriors, new labels; show the old result's numbers (37.3%, both labels) unaltered.

### 3.7 Screen 7 — Model Load Failure State

**What works**
- Headline "**The prediction model could not be loaded.**" and a collapsed **Technical details** disclosure match the brief; the form is not shown (matches `st.stop()` behaviour); the statement that no synthetic probability is shown is a good principle ("The system will never supply synthetic or unverified probability scores").

**What is wrong / unnecessary**
- Almost everything else is **invented infrastructure** the app does not have: "Environment: production-node-04", "Trace ID 0x8F9B2", "Pipeline block #0x4A", "100% Protected", "Strict Fail-Closed", "**Self-Remediation Protocol: the deployment pipeline automatically queries cold-tier model replicas every 300 seconds … Replica query in 04:12 / Cold cache fallback ready**" (false; there is no retry, replica or timer), "Retry Handshake", "Inspect Model Card" (no such feature), "Contact institutional administrator", "Zero-surrogate policy enforced", "Audited".
- Three status tiles and two explanatory cards for one error: heavy and alarming; the brief asks for a calm card with a short plain message.
- Top nav still present on a state where there is no app.

**Recommended changes:** one full-width card: headline, one or two plain sentences ("The model file could not be loaded or did not pass its integrity check. No prediction is shown."), collapsed "Technical details" (file name, hash mismatch/scikit-learn mismatch, as the app reports), no retry/remediation claims; no nav, no tiles.

---

## 4. Fabrication and fidelity register (Checklist §1, §2, §5)

Verified against `model_metadata.json`, `src/phase7_findings.md`, `README.md` and the audit. "Shown" = what Stitch rendered.

| Item | Shown | Verified | Screens |
|---|---|---|---|
| Per-prediction confidence interval | `95% CI [0.321, 0.428]` / `[0.318, 0.431]` | **None exists** (Brief §14; band is not a CI) | 1, 6 |
| Evaluation set size | `N=1,685 (20%)` | **1,487** (Phase 7 test) | 1, 6 |
| ROC-AUC / accuracy in footer | `ROC-AUC 0.961`, `Accuracy 94.2%` | ROC-AUC **0.886**; accuracy 0.812 @0.50 / 0.802 @0.35 | 1 |
| Sensitivity / specificity | `89.2% / 71.4%` (0.35), `74.1% / 88.6%` (0.50) | recall **0.851** / spec **0.766** (0.35); **0.765 / 0.846** (0.50) | 1 |
| Cut-off wording | "**OPTIMAL F1 CUT-OFF 0.350**" | forbidden; "selected … not claimed best" | 1, 5, 6 |
| Calibration method | "**Platt-Scaled Sigmoid**" | none; raw probabilities, no recalibration | 6 |
| Probability / log-odds precision | `p = 0.3734`, logit `−0.518`/`−0.519` | 0.3730, −0.5195 (audit §9) | 1, 6 |
| Log-loss / Brier references | `0.693 uninformative`, `0.250 null` | valid coin-flip baselines but **not the project's documented ones**: **0.679**, **0.243** (training-rate predictor) — *undocumented baseline, not a false number* | 4 |
| Intercept | `−0.68` (stale), `−1.08` (mobile) | **−1.661** | 3, 6 |
| Contributions for the typical profile | mobile: CGPA +0.68, Internships +0.42, SSC −0.51, Aptitude −0.24 … | Training +0.97, Extracurricular +0.72, Soft +0.05, Aptitude +0.04, SSC +0.02, CGPA +0.00; Workshops −0.32, Projects −0.31, HSC −0.03 | 3, 6 (desktop main is correct for the rows visible) |
| Model hash | `#9d2a014e` | `58e8b04e9666b13b…` | 6 |
| Input ranges | CGPA 6.0–10 / HSC 50–100 / Aptitude 0–100, 40–100 / Soft 1–5 / HSC 40–100 / Internships 0–8 / Hackathons 0–12 | CGPA 6.5–9.1, SSC 55–90, HSC 57–88, Aptitude 60–90, Soft 3.0–4.8 | 1, 3, 5 |
| Levels | Internships 0–3+, Projects 0–4+, "Backlogs", "Hackathons", Academic Stream | Internships 0/1/2; Projects 0/1/2/3; no such fields | 2, 3, 5 |
| Peer/segment statistic | "similar profiles N = 340; 39.1% cohort rate; calibration Δ −1.8%" | does not exist; out of scope | 6 |
| Predictive labels | "Likely Placed", "At-Risk / Marginal", "lower quartile of predicted student distributions" | none exist | 2, 3, 6 |
| Infrastructure | seed, latency, trace ID, node, replica queries, retry handshake, "Last eval 4m ago" | none exist | all |
| What-if simulator | "Live Parameter Test Bench", fabricated logit equation | out of scope | 2 |
| Dataset source line | absent | Hugging Face `Krooz/Campus_Recruitment_CSV`, CC0-1.0 should appear (Method / Limitations) | all |
| Disclosure of "not causal / constructed data / Projects=0" | partly (Projects note on 1, 3; "not a guarantee" on 3) | needs to be present near the result on desktop | 1 |

**Correct and worth preserving verbatim:** hero 37.3% and typical-profile labels (Placed @0.35, Not Placed @0.50, band Low); the desktop contribution rows visible on main; the four evaluation headline values; the exact validation/summary sentences; the Projects = 0 note; "The threshold changes the label, not the probability"; the disagreement-note copy; the disclaimer string.

**Checklist [BLOCKER] items failing:** 1.1 (fields differ on 2/5), 1.2 (ranges/options wrong on 1/3/5), 1.3 (not empty), 1.4 (sliders; clamping), 1.7 (word "optimal"), 1.8/1.9 (wrong evaluation numbers on 1), 1.11 (CI), 2.1/2.2/2.6 (avatar, nav), 5.1 (JS, fixed bar, CDN, gauge), 8.1 (button label), 8.3 (copy strings on desktop screens). Checklist 3.x–7.x partly fail as scored in §2.

---

## 5. What is worth keeping (strongest existing elements)

| Keep | From | How to carry it over |
|---|---|---|
| Two-pane desktop: form left, hero + result right, hero visible at first glance | 1 | Brief D3 layout, light theme |
| Three grouped input cards; counts as option groups; Yes/No pairs; Projects = 0 note | 1, 3 | exact ranges/levels only |
| **Stale state**: banner + MODIFIED/Prior→Now + dimmed "From previous inputs" result | 6 | real label; readable dimming |
| **Validation pattern**: summary at top of result panel + inline messages (icon + border + text), exact error sentences, "no result shown" | 5 | real fields; keep Predict enabled |
| Mobile result copy and sequence (hero → ruler + legend → summary → disclaimer → two cards → agreement note → band → tabs) | 3 | move disclaimer directly under hero block; merge adjacent notices |
| Log-odds **and** ×odds-multiplier on each factor row; odds-multiplier aid; sign + group heading + text | 1, 3 | use the real contributions |
| Mobile factor rows reflowing to two lines | 3 | keep |
| Geometric (non-✓/✕) class marks, neutral shape glyphs | 2 | square = Placed, circle = Not Placed, equal weight |
| Evaluation tab title, four tiles with plain-language meanings, generalisation caveat | 4 | drop progress bars and invented references |
| Model-load headline + collapsed technical details; "never show a synthetic probability" principle | 7 | strip all invented infrastructure |
| Inter with tabular numerals; hero ≥ 72 px | 1, 2 | brief type scale |

---

## 6. Consistency with the application audit and `design/ui_ux/` documents

| Audit/brief item | Stitch status |
|---|---|
| Audit P1 #1 stale result | **Addressed** (screen 6) — needs D1 (drop `st.form`) to be implementable (Brief §3.3) |
| Audit P1 #2 two-threshold story | **Mostly addressed** with explicit disagreement note (1, 3); band/label contradiction explained but visually confused (§3.1) |
| Audit P1 #3 disclaimer near the number | Mobile: yes (below hero); desktop: only the footer "Academic notice" — **not addressed** on screen 1 |
| Audit P1 #4 log-odds in plain language | Partly (aid on mobile; desktop only a small mono note under the bars) |
| Audit P2 #5 result far from form | **Addressed** by two-pane |
| Audit P2 #6 errors far from fields | **Addressed** (screen 5) |
| Audit P2 #7 redundant range text | **Worse** on 5 (range ×3) |
| Audit P2 #8 model info at bottom, no explanation | Replaced by a collapsed accordion (1) / unverified tabs (4) |
| Audit P2 #9 selectbox for 3–4 options | Addressed except Workshops on mobile |
| Audit P2 #10 10-column "inputs used" table | Not shown on any screen (tab exists on 3; content not rendered) |
| Audit P2 #11 Load typical feedback (toast) | Not shown |
| Audit P3 identity / hierarchy | Identity changed but generic; see §2.7 |
| Audit §11 overlapping tick labels | **Reproduced** on screen 1 |
| Audit §8.1 `verify_app.py` locks | Widget types/keys: counts as non-select groups = needs D2 update; button labels **do not match**; `hero-prob` marker implementable; copy strings present only on screen 3 |
| Brief §6 "no gradients, glass, glow" | **Violated** by every screen |
| Brief §7 palette | **Not used**; dark tokens instead |
| Brief §9 breakpoints | Only one mobile frame; "About" mislabelled |
| Brief §10 states | Empty, mobile errors, mobile stale, mobile load-failure, model-computation error: **missing** |
| Brief §14 exclusions | Violated: nav/avatar, benchmark panel, CI, simulator, "Likely Placed" etc. |

---

## 7. Recommended visual and motion improvements (for the new generation)

### 7.1 Identity (distinctive without leaving the brief)

Keep the brief's light clinical base and give it **one signature device**, so the product has a recognisable look without decorative effects:

- **The probability ruler as the brand motif.** A precise horizontal scale: hairline axis, ticks every 5 % (long ticks at 0/50/100), the **fill in brand navy** `#0F4C75` (a neutral, so that blue/orange keep their "toward Placed / toward Not Placed" meaning for contribution bars and chips — a one-line amendment to Brief §7 `--raise` "bars/meter fill"), and the two cut-offs as **engraved shape-coded ticks** (filled ◆ 0.35, hollow ◇ 0.50) with a legend row below, never labels that collide. Echo it in small as the divider under section headings and in the contribution bars (same tick language), so the whole product reads as one measuring instrument.
- **The "disagreement zone".** Lightly tint the span between 0.35 and 0.50 on the ruler and label it "labels differ here". It shows exactly why 37.3% is "Placed" at 0.35 and "Not Placed" at 0.50, uses only the two existing thresholds, and replaces three explanatory boxes with one picture plus one sentence. (Distinctive, and honest.)
- **Typography:** Inter with tabular figures for all UI (self-hosted via `fontFaces` or the system stack). Optionally (decision, Brief D5) one display face used **only for the hero numeral** — gives a premium feel at no cost in clutter. Drop JetBrains Mono entirely: monospace labels are the biggest contributor to the "terminal" look.
- **Surface treatment:** white cards on `#F6F8FB`, 1 px hairlines, 12 px radius, no glow, no glass, no gradient CTA. Primary button: solid navy. Premium comes from spacing, alignment, numeral scale and restraint.
- **Information density:** header = H1, one-line purpose, three badges, three-step strip (the mobile screen already has this). Delete status strips, nav, avatar, footers.

### 7.2 Three-way distinction (probability / band / classification)

| Layer | Visual encoding | Never |
|---|---|---|
| **Probability** (primary) | hero numeral 72/52 px + position on the ruler; eyebrow "Estimated probability of placement" | sit inside a decorative gauge; add a CI or extra decimals |
| **Classification at a cut-off** (derived) | two equal cards "Model classification — 0.35 project threshold" and "Conventional 0.50 threshold"; chip with text + neutral shape glyph (■ Placed / ● Not Placed); same size, same weight; line "Placed when probability ≥ 0.35" | use grey for one and colour for the other; call either recommended/optimal |
| **Band** (descriptive) | small neutral outlined tag next to the ruler "Interpretation band: Low — describes the size of the probability, not a confidence interval; independent of the cut-offs" | use blue or orange (reserved for direction) |

One merged "How to read this" note (disagreement or agreement sentence + "the threshold changes the label, not the probability") replaces three stacked notices.

### 7.3 Motion (attractive but restrained)

Current brief limits motion to "≤ 200 ms meter fill". Recommendation (needs approval, new decision **D8**):

| Moment | Motion | Duration |
|---|---|---|
| New result appears | ruler fill grows from 0 to p, ease-out; ticks fixed | 350–450 ms, once |
| Hero numeral | opacity fade-in only (no count-up: intermediate numbers would show probabilities the model never produced) | 150 ms |
| Contribution bars | grow from the centre line, 25–30 ms stagger, ≤ 10 rows | 250 ms |
| Stale transition | previous result changes to the "From previous inputs" state | 150 ms |
| Hover / focus / selection | border-colour and background change | 100–120 ms |
| Error messages | appear without motion | — |

Rules: no looping or pulsing indicators, no glow animations, no parallax/shimmer, no spinner longer than 300 ms; **`prefers-reduced-motion: reduce` removes all of it** (result simply appears). **Streamlit caveat (to verify in a prototype):** CSS `@keyframes` inside an injected `<style>` is feasible; JS is not. Because Streamlit reruns re-render the result on every widget change, the animation must be tied to a freshly computed result (e.g. class applied only when a new prediction is stored), not replayed on every rerun, and not on the stale view.

### 7.4 Accessibility and mobile fixes to apply in the next generation

1. All text ≥ 13 px; no uppercase/letter-spaced micro-labels for essential information.
2. Non-text contrast ≥ 3:1 for card edges where they carry meaning, control borders `#8A94A3` on white, ruler track vs fill, factor-bar track, centre line.
3. One `h1`; `h2` per section; `h3` per card. Every input has a visible `<label>`-equivalent (native Streamlit widgets) plus one range hint and one error slot.
4. Ruler: text equivalent sentence adjacent (already good), legend instead of positioned labels; `role="img"` + `aria-label` and `<meter>` flagged **to verify** in Streamlit's HTML sanitisation.
5. Stale view: dim by outline/hatch and a tag, keep text ≥ 4.5:1.
6. Mobile 390 px **and** 320 px: single column; Predict full width; Load/Clear stacked beneath; cards stacked; tabs scroll horizontally; tables as key–value lists; no fixed bottom bar; touch targets ≥ 44 px (verify Streamlit steppers).
7. Frames to produce for mobile: form (empty), result, validation error, stale, evaluation tab, load failure. The current "Mobile – About" frame is desktop-width.

---

## 8. Implementation constraints (what Streamlit can and cannot take)

**Do not assume generated HTML drops into Streamlit.** The Stitch HTML is a Tailwind-CDN page with inline JS. Streamlit renders Python widgets; custom HTML via `st.markdown`/`st.html` has no JS, no Tailwind, and unreliable support for `position: fixed/sticky`, `backdrop-filter`, custom `aria-*`/`role`, and `<style>` scoping (Audit §8.2; **sanitisation of `aria`/`role` and `<style>` still to verify in a prototype**).

| Design element in Stitch | Streamlit reality | Verdict |
|---|---|---|
| Tailwind CDN, Google Fonts, Material Symbols | external requests at runtime; Tailwind classes meaningless | **replace**: theme keys + own CSS; self-host Inter or system stack; icons via native Material icon support in widgets or inline SVG (**version to verify**) |
| Dark glass/glow theme | needs `[theme.dark]`; backdrop blur brittle; out of scope | **drop** |
| Radial gauge | static SVG/conic-gradient possible, but a dial; cannot carry two cut-offs clearly | **replace by horizontal ruler** (static HTML/CSS) |
| Steppers with JS clamp | `st.number_input` has native +/−; **never set `min_value`/`max_value`** (clamps); validation lives in `prediction.py` | use native widget, no JS |
| Segmented option groups | `st.radio(horizontal=True)` / `st.segmented_control` / `st.pills`; **changing counts from `selectbox` is D2** and needs `verify_app.py` section 6 update | approval item |
| Sliders | clamp; forbidden | **remove** |
| Inline per-field errors, "MODIFIED / Prior→Now", stale banner | need widgets **outside `st.form`** (D1) and a comparison with `session_state["last"]["clean"]` | approval item; presentation/flow only |
| Top navigation, avatar, bottom tab bar | multi-page navigation out of scope; fixed bars brittle | **drop** |
| Tabs, expanders, tables, bordered containers, columns | native | fine (two-pane + tabs nesting depth **to verify**) |
| Result card, ruler, bars, chips, notices | static HTML/CSS in `ui_helpers.py` | fine; must keep `<div class="hero-prob">{p:.1f}%</div>` |
| Toasts ("typical profile loaded") | `st.toast` | fine |
| Click `scale-95` effect, "JS Evaluator", live bench | JS | **drop** |
| Fonts JetBrains Mono / Material Symbols | extra downloads | **drop** |

**Visual design vs application behaviour — keep them separate.**

| Visual layer (what Stitch should deliver) | Behaviour layer (stays in the app; not designed in Stitch) |
|---|---|
| Palette, type scale, spacing, radius, hairlines | `validate_inputs`, ranges, rare-level rule, error wording |
| Ruler, bars, chips, cards, notices, tabs as *look* | `predict`, `classify` (`p ≥ threshold`), bands, `explain` |
| States as pictures: empty, result, errors, stale, load failure | When states occur: session state, rerun rules, form-vs-no-form, `st.stop()` |
| Copy strings (as given) | `verify_app.py` locks: keys = feature names, button labels, `hero-prob`, copy strings |
| Motion spec | Where animation is triggered (fresh result only) |

Nothing in this review requires changing `prediction.py`, `explainability.py`, the model, thresholds or metadata. New static content needed later (ECE tile, threshold table, dataset source) is already documented in `README.md`/`phase7_findings.md` (Brief D4).

---

## 9. Prioritised change list

**P0 — must be fixed in the new generation (blockers)**

1. **Light theme per the brief** (Brief §7) unless decision I (§10) reopens dark; either way delete neon glow, glass blur, gradient CTAs and decorative gradients, and restore a design document and screens that use the *same* palette.
2. **Use only supplied facts:** no CI, no invented metrics/IDs/fields/ranges/labels/infrastructure/peer data/simulator; unknown numbers as `XX.X`; every example screen tagged "Example values".
3. **Correct inputs:** exactly ten fields, supported ranges, levels 0/1/2 and 0/1/2/3, all empty, **no sliders**, no clamping, native-style number fields with steppers and one range hint each.
4. **Exact labels:** `Predict Placement Probability`, `Load typical profile`, optional `Clear profile`; exact strings for classification cards, headings, disclaimer, errors.
5. **Single workspace:** remove top nav, avatar, bottom tab bar, status strips, footer telemetry, "ONLINE" etc.
6. **Fix the Stitch design document** (§2.2): remove sliders, sidebar dock, "confidence intervals", 11–12 px label sizes; resolve shadows.
7. **No "optimal / best / recommended / likely / at-risk"**; remove "Platt", "SHAP", "posterior", "quartile" claims.

**P1 — hierarchy, clarity, accessibility**

8. Replace radial gauge + second meter with **one horizontal ruler**, shape-coded cut-offs, legend row, "disagreement zone".
9. Three-layer encoding (§7.2): band neutral, chips equal weight with ■/● glyphs.
10. Merge the three adjacent notices; put the disclaimer immediately under the hero/ruler block on desktop too.
11. Contribution rows: centre-line diverging bars on both desktop and mobile; real values; exact group headings; keep odds-multiplier aid; footnote always visible.
12. Empty state, stale state (readable), validation state (Predict enabled, one range hint), load failure (calm, short) — all on real fields.
13. Contrast: UI boundaries ≥ 3:1; stale text ≥ 4.5:1; type floor 13 px; headings and labels.
14. Evaluation tab: tiles without progress bars; add Thresholds table and Limitations/data tab in the same visual language; dataset source line.
15. Regenerate **mobile** frames at 390 px (form empty, result, errors, stale, evaluation, load failure); verify at 320 px.

**P2 — polish and motion**

16. Signature ruler motif repeated in bars and dividers; optional display face for the hero numeral (D5).
17. Motion spec per §7.3 with `prefers-reduced-motion` handling (D8).
18. Hover/focus/selected/error state specimens on a *light* token sheet; geometric class marks.
19. Toast or caption confirming "Typical profile loaded: median scores and counts, most common Yes/No answers."
20. Header trimmed to H1 + subtitle + three badges + three-step strip.

---

## 10. Final recommendation — edit or generate a new variant?

**Generate a new set; do not edit these screens, and do not use `generate_variants` on them.**

- Edits (`edit_screens`) keep the layout skeleton and mostly the visual language. Here the skeleton is serviceable (two-pane, cards, tabs) but the *visual language, content, component set and about half the copy* are wrong. Converting dark → light, removing the gauge/nav/avatar/sliders/JS, replacing every number and string, and fixing ranges would touch effectively every element on every screen, with no guarantee Stitch will not re-introduce the glow or invented metrics on the next pass. Variants of the current screens would inherit the same dark system.
- The good parts (§5) are structural and copy-level and can be transcribed into a tightened prompt.
- A fresh generation lets us put the corrected design document in place **first**, so the new screens inherit the right tokens.

**Proposed sequence (after your approval; nothing below has been done):**

1. Fix or replace the Stitch design system (`update_design_system` or create from a corrected design document based on the brief palette; remove sliders/sidebar/CI; 13 px floor).
2. Generate the **desktop result** first with a stricter prompt: the existing `stitch_prompt.md` plus the lessons from this review (explicit "light theme only; no navigation; no avatar; no gauge; ruler with disagreement zone; no figures other than those listed; fields empty in the empty state"). Review it against `design_review_checklist.md` before generating anything else.
3. Generate the remaining desktop states (empty, validation, stale, load failure, About tabs) and the 390-px mobile set only after the desktop result passes.
4. Keep the current project's screens as an archive (do not delete); apply the new set in the same or a new project, as you prefer.

**Decisions I need from you before any Stitch change**

| # | Decision | My recommendation |
|---|---|---|
| A | New generation instead of editing | Yes |
| B | Fix the Stitch design document (§2.2) first | Yes |
| C | Ruler with "disagreement zone" instead of a radial gauge | Yes |
| D | Meter fill in brand navy (amend Brief §7 `--raise` use) | Yes |
| E | Motion upper bound ≤ 450 ms for the meter fill (Brief says ≤ 200 ms) | Yes, with reduced-motion handling |
| F | Optional display face for the hero numeral (Brief D5) | Optional; Inter-only is acceptable |
| G | Keep Predict enabled with errors (Brief §4) rather than disabled | Keep enabled |
| H | Shadow: hairline only vs hairline + very soft shadow | Hairline only |
| I | **Theme: light per the brief, or reopen a dark theme** (your uploaded references are dark glass + neon radial gauge) | **Light for release 1.** Your goal "less generic AI-generated dashboard styling" argues against these references: dark glass + glow + radial gauge is the look most generated dashboards share. If you want dark, treat it as a deliberate brief change: a restrained dark variant (no glow/glass/gradient, solid surfaces, same ruler motif) needs `[theme.dark]` config, its own contrast pass (non-text ≥ 3:1) and a second set of states, so I would add it after the light set is approved rather than first. |
| J | Replace the radial gauge (as in your reference #2) by a horizontal ruler | Yes — a dial cannot carry two cut-offs and is a stock "AI dashboard" element; keep the *emphasis* (large central numeral) from the reference, not the dial |
| — | Existing D1–D7 from the brief remain open | unchanged |

---

## 11. Limits of this review

- Screenshots were reviewed visually (some downscaled); small text was cross-checked against the HTML. No browser run, no interaction testing, no measured rendering.
- Contrast ratios are computed from Tailwind token hex values in the generated HTML, not from pixel sampling.
- Hidden/earlier screen instances in the project were not reviewed.
- Streamlit feasibility statements follow the audit and brief (verified there on 1.64.0) and carry their "to verify" flags (HTML sanitisation of `aria`/`role`/`<style>`, nested columns + tabs, stepper touch targets, minimum-version theme keys, Material icon support at the pinned version).

---

## 12. Resolution — final decisions (2026-10-17)

This section is the only part of the file written after the review. It does not change the assessment above. Nothing in Stitch was contacted, edited or generated; the existing project and its seven screens are untouched and remain an archive.

### 12.1 Decisions A–J (§10) — all resolved
| # | Decision | Resolution | Specified in (`design_brief.md` v2) |
|---|---|---|---|
| A | New generation instead of editing | **Yes** — new set, ideally in a **new Stitch project** (the old project holds the dark design system and two dark reference images that steered the first set) | §2 |
| B | Fix the design system / documents first | **Yes** — documents finalized; a corrected light design-system document is in `stitch_prompt.md` Part 1 and replaces `designMd` (its sliders, sidebar dock, "confidence intervals", 11–12 px sizes and SHAP wording are removed) | §2, `stitch_prompt.md` |
| C | Ruler with disagreement zone instead of a gauge | **Yes** — horizontal ruler, filled ◆ 0.35 and hollow ◇ 0.50 markers, hatched 35–50 % region with a labelled bracket | §5.4 |
| D | Meter fill in brand navy | **Yes** — blue/orange keep one meaning (direction) | §5.4, §6.1 |
| E | Motion ceiling | **Ceiling 450 ms**, reduced-motion honoured; ruler fill 400 ms | §7 |
| F | Optional display face | **Yes, for the hero numeral and H1 only** (Newsreader → Source Serif 4 → Fraunces; lining figures); all else Inter | §6.2 |
| G | Predict enabled with errors | **Yes** — enabled, never executes on invalid input; inline + summary messages | §9 |
| H | Shadow | **Hairline only**, no decorative card shadows | §6.3 |
| I | Theme | **Light for release 1**; no dark tokens specified | §2, §6.1 |
| J | Replace the radial gauge | **Yes** — large central numeral kept, dial removed | §5.3, §5.4 |
| D1–D7 | Brief decisions | Drop `st.form`; counts as radio groups (+ `verify_app.py` update); two-pane 5/12 : 7/12; static ECE/threshold/source content; serif hero + H1, no CDN; "Clear profile" included; target and pin Streamlit 1.64.0 in the implementation change | brief §13 (full question / resolution / rationale / sections) |

### 12.2 Where the final specification differs from this review's recommendations
| This review said | Final specification | Why |
|---|---|---|
| §7.3 bars 250 ms with 25–30 ms stagger | Bars 220 ms, stagger ≤ 20 ms, last ends ≤ 400 ms | The review's numbers reach ~520 ms with ten rows; E caps everything at 450 ms |
| §2.4 / §8: Tailwind CDN, Google Fonts and inline JS flagged as hard-constraint violations | Stitch HTML is a visual reference; links and incidental scripts are not defects. The defect is a *behaviour* Streamlit cannot build (clamping steppers, bench, sliders). The no-CDN rule applies to the shipped app | Stitch always emits CDN markup; judging it would fail every output |
| §3.4 / §5: four evaluation tiles without progress bars | A quiet four-row key–value list — no tiles | The brief of this stage removes KPI-card styling |
| §7.2 chips use ■ Placed / ● Not Placed; band neutral | Adopted exactly; diamonds are reserved for threshold markers | Avoids two meanings for one glyph |
| §3.6 stale: "dim by outline/hatch" | Adopted; opacity-based dimming is a checklist failure (5.10) | Text must stay ≥ 4.5:1 |
| Badge "No data stored" (brief v1, mobile screen) | "Nothing is saved" (the app itself does not save or log values; they exist for the browser session) | Precision: Streamlit holds session state in memory |
| Tags "Example values" | "Example profile" (hypothetical profile, real model output) | Accuracy |
| §10 decision I left dark open | Light only; dark system not used | Authoritative direction |

### 12.3 What the next design iteration must correct (old defect → where it is now controlled)
| Old-screen defect (§2–§4, §9) | Specification | Checklist |
|---|---|---|
| Dark obsidian/cyan, glass, glow, gradient CTA, mono labels | brief §2, §6 | 2.9–2.10, 4.3–4.9, 4.14 |
| Radial gauge, dotted ring, second meter, overlapping tick labels | brief §5.4 | 2.9, 2.13, 3.1–3.14 |
| Sliders; clamping steppers with wrong ranges; pre-filled fields | brief §5.2, §12.3 | 1.2–1.4, 5.1–5.3 |
| "Compute …" button label; missing exact strings | brief §11.2 | 5.4, 10.6 |
| Fabricated metrics (N = 1,685, ROC-AUC 0.961, sensitivity 89.2 %), 95 % CI per prediction, "Platt", "Optimal F1", invented posteriors / benchmark / infrastructure / fields (Backlogs, Hackathons) | brief §11.1, §11.3, §15 | 1.8–1.13, 2.1–2.8, 9.1–9.4 |
| Nav, avatar, bottom tab bar, status strips, footer telemetry | brief §3, §5.1, §15 | 2.3, 2.5 |
| "Live Parameter Test Bench" / what-if simulator; "Likely Placed" | brief §15, §11.3 | 1.12, 2.8 |
| Tiny labels (9–12 px) | brief §6.2 | 4.8 |
| Band coloured like a classification; unequal chips | brief §5.3 | 3.11–3.12 |
| Disabled Predict on validation screen; range text three times | brief §9 | 5.5, 5.8 |
| Opacity-based stale dimming below 4.5:1 | brief §9 | 5.10 |
| Contribution list incomplete (SSC, CGPA, baseline group missing) | brief §5.5 | 1.10 |
| Mislabelled "Mobile" About frame (1280 px), single mobile frame | stitch_prompt S5, S7, S8 | 6.3–6.9 |
| Undocumented baselines 0.693 / 0.250 | brief §5.6 | 9.4 |
| Non-text contrast 1.3–1.7:1 | brief §6.1 | 7.2 |

### 12.4 Next stage (not performed here)
1. Create a new Stitch project; apply `stitch_prompt.md` Part 1 as its only design system.
2. Send the Main Prompt (desktop result screen only); review against `design_review_checklist.md` §1–§3 and §5.
3. After it passes, send S2 → S9 one at a time, reviewing each.
4. Keep the old project as an archive; do not edit, vary or delete its screens.
5. Implementation in Streamlit is a later, separately approved stage with its own regression run of `verify_app.py`.

