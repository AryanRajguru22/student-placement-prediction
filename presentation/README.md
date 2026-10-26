# Final presentation (Phase 8C)

**Predicting Student Placement Probability Using Academic, Technical and Employability Features**

| File | Content |
|---|---|
| `final_presentation.pptx` | 15 slides, 16:9, speaker notes embedded in every slide |
| `slide_notes.md` | The same speaker notes as plain text (about 1,800 words; roughly 11–12 minutes at a normal speaking pace) |
| `../report/viva_questions.md` | 59 viva questions with concise answers, grouped A–O, plus a one-page numbers cheat sheet |

## Open it

Open `final_presentation.pptx` in PowerPoint (or LibreOffice Impress). Speaker notes appear in Notes view / Presenter view.

## Before presenting — fill in the placeholders

Slide 1 contains bracketed placeholders that were deliberately left blank (nothing was invented):
`[Student name(s)]`, `[Registration number(s)]`, `[Course / programme]`, `[Department]`, `[University]`, `[Supervisor / faculty]`, `[Date]`.
The document author property is also `[Student name(s)]`.

## Slide outline

1. Title
2. Problem and motivation — probability is the primary output
3. Dataset — 10,000 rows, 11 columns, three feature groups (original dataset, before cleaning)
4. Data quality and leakage — 10,000 → 9,912 rows; 33 profiles crossed the supplied splits; new 6,938 / 1,487 / 1,487 split
5. What the data showed — strong associations; non-monotonic count variables
6. Statistical analysis — methods, VIF 1.96, aptitude AUC ≈ 0.81
7. Modeling pipeline — test evaluation comes last
8. Model comparison — LR_B, GB_B, RF_B practically tied
9. Cross-validation and calibration
10. Threshold selection — 0.35 selected operating threshold, 0.50 reference
11. Explainability — additive logit, permutation importance
12. Ablation and error analysis
13. Final held-out test results
14. Interactive application (workflow schematic)
15. Conclusions and limitations

## Sources and constraints

* Every number on the slides appears in `report/final_report.md` (checked automatically). Charts are native PowerPoint charts built from those numbers; no analysis was run for the presentation.
* No model was trained or retrained, no threshold or preprocessing was changed, and `test.csv` and the test prediction files were not opened.
* **No application screenshot exists in the project**, and none was manufactured: slide 14 is a schematic of the workflow and says so.
* Phrases to keep when speaking: the 0.886 value is a **ROC-AUC, not an accuracy**; 0.35 is the **selected operating threshold under an F1-oriented criterion**, not a universally optimal one; the three leading models are **practically tied**; all statements are **associations, not causal effects**.
* Test-access caveat (appears on slides 13 and 15 and in the notes): test rows were not used to fit, select, tune, calibrate or set thresholds, but early descriptive analyses (Phases 1–3) included them, and two orientation shell commands looked at the file before the single gated evaluation. See report Section 8.
