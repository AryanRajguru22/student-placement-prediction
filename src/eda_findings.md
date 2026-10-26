> All statements describe **association in this dataset** (n = 9,912 distinct feature profiles). None is a causal claim.

### A. Strong observed patterns
- **Every predictor is associated with placement, and most strongly the scores.** Standardised mean differences (Placed minus NotPlaced): AptitudeTestScore d = 1.23, HSC_Marks 1.18, Projects 1.08, SSC_Marks 1.07, SoftSkillsRating 0.95, CGPA 0.94. Correlations with the 0/1 target range from r = 0.52 (aptitude) and 0.50 (HSC) down to 0.42 (CGPA, soft skills).
- **Students who report extracurricular activities show a much higher observed placement rate** (61.6% vs 13.7%), as do students with placement training (51.2% vs 15.6%).
- **The two binary predictors co-occur and their effects combine:** observed placement rates run from 6.1% (neither, n = 1,826) through 19.7% (training only) and 35.9% (extracurricular only) to 66.1% (both, n = 4,912).
- **The joint pattern across academic scores is steep:** in the exploratory CGPA x aptitude grid the observed rate rises from 7.8% (lowest bins) to 88.7% (highest bins).

### B. Weaker / moderate patterns
- Internships (d = 0.53, r = 0.25) and Workshops/Certifications (d = 0.78, r = 0.36) have the smallest correlations with placement; the Internships association is carried almost entirely by the jump at 2 (see C).
- CGPA and SoftSkillsRating have large mean differences (d about 0.94) but only moderate linear correlation (r about 0.42), consistent with overlapping distributions.
- Pearson and Spearman agree closely for all eight numeric predictors (differences <= 0.03), so rank-based and linear summaries tell the same story for these variables.

### C. Unexpected / non-monotonic patterns
- **Internships:** 33.4% (0) vs 32.6% (1) vs 69.0% (2). No visible gain from 0 to 1, then a large jump at 2.
- **Workshops/Certifications:** 28.4% (0), 21.1% (1), 72.6% (2), 56.3% (3). Level 1 sits *below* level 0 and level 3 *below* level 2.
- **Projects:** 60.0% (0, n = 30 only), 16.9% (1), 31.4% (2), 71.4% (3). The Projects = 0 rate rests on 30 students (Wilson 95% CI 42-75%) and should not be over-read.
- **Composition:** students with 2 workshops have higher average CGPA (8.05 vs 7.5), aptitude (84.8 vs 76-77), more projects (2.56 vs 1.7) and far more extracurriculars (87% vs 41-43%). Their raw placement rate partly reflects who sits at that level.
- **Within aptitude terciles, the dip at Workshops = 1 persists** (low 15.4% -> 9.9%, mid 29.3% -> 25.3%, high 58.4% -> 53.3%), as does the 2 -> 3 drop in the mid and high bands (low band n = 55 at level 3). So the non-monotonicity is not explained by aptitude alone; other covariates (e.g. extracurriculars) may contribute. This remains descriptive.

### D. Potential data-quality / data-character concerns
- Hard floors and ceilings with pile-ups at the limits (CGPA at 6.5, aptitude at 90, soft skills at 4.8, SSC at 55) in the violins; ranges have no tails.
- Very large, systematic effects across *all* predictors plus step-like non-monotonic count patterns are consistent with a rule-based or synthetically generated dataset. This is an observation, not a conclusion; the dataset card does not state provenance. It also means model performance here may not transfer to real placement data.
- Predictors are moderately inter-correlated (max pairwise r = 0.56, aptitude / HSC), and ExtracurricularActivities and PlacementTraining co-occur (85% of extracurricular students also had training).
- 8 conflicting-label profiles (16 rows) were excluded in notebook 02. Within each pair every predictor is identical yet one student is Placed and one NotPlaced, and all 8 sit at the top of the range (aptitude 90, 2 workshops, 3 projects, extracurricular and training both Yes). The label therefore depends on something not captured by the features, or contains noise, at least in this high-profile region; this limits the best achievable accuracy there.

### E. Questions to test statistically in Phase 3
1. Does each numeric predictor differ between Placed and NotPlaced (Mann-Whitney U, effect sizes with CIs)?
2. Are ExtracurricularActivities and PlacementTraining associated with placement (chi-square, odds ratios), and are they associated with each other?
3. Is the Internships 0 vs 1 difference distinguishable from zero, and is the jump at 2 real?
4. Is the Workshops 1 < 0 and 3 < 2 pattern statistically distinguishable from noise, with multiple-comparison correction?
5. Does the Workshops/Projects association persist after stratifying on aptitude (Cochran-Mantel-Haenszel)?
6. How much redundancy exists among the score variables (VIF)?

### F. Features that may be useful for prediction
- All 10 predictors show a clear association with the outcome, so none is a candidate for removal at this stage; AptitudeTestScore, HSC_Marks, Projects, SSC_Marks and the two binary variables show the largest raw separations.
- Because Internships, Projects and Workshops/Certifications behave non-monotonically, a linear-in-count treatment may be inadequate; Phase 4 should compare treating them as numeric vs categorical, and tree-based models can capture steps natively.
- Possible interactions to examine later (not engineered here): Extracurricular x PlacementTraining, Projects x Workshops.
- No transformation, binning or feature creation was applied; exploratory bins were visualisation-only.

