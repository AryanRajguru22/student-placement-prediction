> All results are associations in observational (possibly synthetic or rule-based) data. None is a causal claim, and p-values below about 1e-300 are shown as "<1e-300" because they underflow double precision.

### A. Statistically supported associations (Holm-adjusted p < 0.05 within family)
- **All five scores differ between Placed and NotPlaced** (Mann-Whitney, confirmed by Welch): Cohen's d from 0.94 (CGPA) to 1.23 (AptitudeTestScore); AUC from 0.76 to 0.81. Welch and Mann-Whitney agree on every feature.
- **ExtracurricularActivities:** OR = 10.1 (95% CI 9.1-11.2), Cramer's V = 0.48. **PlacementTraining:** OR = 5.7 (5.1-6.4), V = 0.32. Chi-square assumptions were met (smallest expected count > 1,100).
- **Internships, Workshops/Certifications and Projects** are each associated with placement overall (Cramer's V = 0.31, 0.45, 0.49).
- **Pairwise:** Internships 1 vs 2 (+36.4 pp), 0 vs 2 (+35.5 pp); Workshops 0 vs 1 (-7.3 pp), 1 vs 2 (+51.5 pp), 2 vs 3 (-16.3 pp), 0 vs 2, 0 vs 3; Projects 1 vs 2 (+14.5 pp), 2 vs 3 (+40.0 pp), and 0 vs 1 and 0 vs 2 (based on only 30 students at level 0).
- **All of these were replicated on the training split alone** with the same direction and significance (effect sizes within about 0.03 in d, 0.3 in OR for training).

### B. Associations that were not statistically supported
- **Internships 0 vs 1:** -0.8 pp (95% CI -3.3 to +1.6), Holm p = 0.50. This is absence of evidence of a difference, not proof of equality; differences larger than roughly 3 pp in either direction are unlikely given the data.
- **Projects 0 vs 3:** +11.4 pp (CI -4.1 to +29.1), Holm p = 0.22. The comparison is underpowered because Projects = 0 has only 30 students.
- Aptitude-stratified Projects 0 vs 3: MH OR = 0.87 (0.41-1.85), p = 0.71.

### C. Non-monotonic relationships
- **Internships:** flat from 0 to 1, then a large jump at 2 (a step, not a trend).
- **Workshops/Certifications:** statistically supported dips at 0 -> 1 (OR 0.67, 0.60-0.76) and 2 -> 3 (OR 0.49, 0.38-0.62). Trend tests would be inappropriate.
- **Projects:** the 60% rate at level 0 rests on 30 students (Wilson CI 42-75%) and is weakly determined.
- **After stratifying by aptitude tercile:** the large upward steps shrink substantially (Workshops 1 -> 2: crude OR 9.9 -> MH OR 4.7; 0 -> 2: 6.7 -> 3.6; Projects 2 -> 3: 5.5 -> 3.0; 1 -> 2: 2.25 -> 1.86), so aptitude accounts for part of those differences. The non-monotonic dips **persist** (Workshops 0 -> 1: MH OR 0.74, 0.65-0.83; 2 -> 3: 0.63, 0.48-0.83), so they are not explained by aptitude alone. Breslow-Day tests suggest the ORs are not homogeneous across bands for Workshops 0 vs 2 (p = 0.007) and 2 vs 3 (p = 0.036). Only one coarse variable was stratified, so other covariates (e.g. extracurriculars, which co-vary strongly with Workshops) may contribute.

### D. Important effect sizes (practical, not just statistical, importance)
- Scores: d about 0.9-1.2 (large by convention). Binary predictors: OR 10.1 (extracurricular) and 5.7 (training). Count variables: Cramer's V 0.31-0.49 (medium).
- Biggest single contrasts: Workshops 1 vs 2 (OR 9.9, +51.5 pp crude) and Projects 2 vs 3 (OR 5.5, +40.0 pp crude); these are the contrasts most attenuated by aptitude stratification.
- Because n is about 10,000, even small differences yield tiny p-values; effect sizes and CIs, not p-values, should guide interpretation.

### E. Potential multicollinearity
- The strongest predictor-predictor association is AptitudeTestScore / HSC_Marks: Pearson r = 0.559, Spearman rho = 0.554 (the EDA value of about 0.56 is verified). Aptitude is also correlated with Projects (0.54), SoftSkills (0.51) and SSC (0.51).
- **VIF (numeric predictors) all <= 1.91; with the two binary predictors <= 1.96**, well below the usual 5 / 10 thresholds. Multicollinearity is low to moderate: coefficients for aptitude, HSC and Projects will share explanatory credit but are not expected to be unstable. No variable should be removed on correlation grounds.

### F. Findings that should inform model selection
- Treat **Internships, Projects and Workshops/Certifications as categorical (one-hot) or use models that capture steps natively**; a single linear slope would misrepresent the non-monotonic patterns. Compare numeric vs categorical coding on the validation set.
- Keep all ten predictors; all show a clear association, and VIF does not argue for dropping any.
- A regularised logistic regression is a sensible interpretable baseline; tree-based models should be compared for step patterns and interactions.
- Anticipate a high attainable performance because separation is large (AUC for single features 0.76-0.81), but also a ceiling in the top profile region where 8 conflicting-label profiles were found and excluded.
- Follow the policy: fit on **train**, select on **validation**, evaluate on **test once**. Preprocessing is fitted on train only.

### G. Questions the ML phase should test
1. Does categorical coding of the three count variables improve validation performance over numeric coding?
2. Do non-linear models (trees, boosting) beat regularised logistic regression, and is the gain attributable to steps and interactions (Extracurricular x Training, Projects x Workshops)?
3. Do the non-monotonic Workshops dips survive in the presence of all other predictors (e.g. extracurriculars), or are they proxies for them?
4. How well calibrated are the predicted probabilities, given the project goal of estimating placement probability?
5. How stable are logistic coefficients and feature importances for aptitude / HSC / Projects given their moderate overlap?
6. How does performance vary within subgroups (top-profile region, small count levels such as Projects = 0 or Workshops = 3)?
