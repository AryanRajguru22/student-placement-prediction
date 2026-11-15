<div class="cover">
<div class="cover-kicker">Predictive Analytics (CSE 3141) · learning &amp; viva masterbook</div>

<h1 class="cover-title">Student Placement Probability Predictor</h1>

<div class="cover-sub">From predictive-analytics basics to the mathematics of Logistic Regression and the working of our own app, explained so that you can defend every decision</div>

<div class="cover-meta">
Prepared for: [Your name] · B.Tech CSE, Semester V · Manipal University Jaipur<br>
Written: 1 November 2026, from the repository <code>student-placement-prediction/</code> and the course material in <code>theory ppts/</code><br>
Final model: Logistic Regression B (<code>models/phase7/LR_B_final.joblib</code>)<br>
Companions: <code>report/final_report.md</code> · <code>report/Student_Placement_Predictor_Complete_Project_Guide.pdf</code> · <code>report/viva_questions.md</code>
</div>

<div class="cover-box">
<b>How to use this book.</b> Chapters 1–2 set the scene. Chapters 3–6 follow the data from raw file to model input. Chapters 7–9 are the core: Logistic Regression, the other models, and what training really did. Chapters 10–12 explain the results, the two thresholds and the explanations. Chapters 13–15 explain the software. Chapter 16 is limitations, Chapter 17 is your presentation, Chapter 18 is the viva bank (110 questions). Glossary, formula sheet and a one-page revision sheet close the book. Chapters 1–14 and 16 end with a short wrap-up box (takeaways, formulas, five revision questions, likely follow-ups).
</div>

<div class="cover-box legend">
<b>Source labels used throughout.</b>
<span class="tag v">VERIFIED</span> read from code or from the saved model file &nbsp;
<span class="tag m">REPORTED</span> a result saved by the project &nbsp;
<span class="tag c">COURSE</span> taught in your course slides &nbsp;
<span class="tag s">SUPPLEMENT</span> standard ML theory that the slides do not cover &nbsp;
<span class="tag e">ESTIMATE</span> counted or inferred &nbsp;
<span class="tag u">UNKNOWN</span> not recorded anywhere
</div>
</div>

<div class="toc-placeholder"></div>

# 1. Understanding Predictive Analytics

## 1.1 The ladder of analytics <span class="tag c">COURSE</span>

Your course (Unit 1) defines **Predictive Analytics** as "using historical data, statistical methods and Machine Learning algorithms to predict future events or outcomes" and, in the simplest words, "learning from the past to predict the future". It separates three kinds of analytics by the question they answer:

| Type | Question | Uses | Course baby example | Our project |
|---|---|---|---|---|
| Descriptive | What happened? | past data, charts, averages | "The baby cried at 2 PM every day" | The EDA notebooks: placement rate by CGPA band, class balance |
| **Predictive** | What will / could happen? | past data + statistics + machine learning | "The baby is likely to cry at 2 PM today" | **The model: P(Placed) for a new student** |
| Prescriptive | What should we do? | prediction + optimisation | "Feed the baby at 1:50 PM" | **Not done.** The app never recommends an action |

The course also lists **Education – student performance prediction** among the application areas. Our project is exactly that kind of application. Remember that the *prescriptive* step is deliberately absent: telling a student "do this to get placed" would need causal evidence that our data cannot give (Chapter 16).

## 1.2 Core vocabulary, tied to our problem

| Term | Meaning | In our project |
|---|---|---|
| **Machine learning** | The computer finds the rule from examples instead of being given the rule | The weights of the model are found from 8,425 labelled students |
| **Supervised learning** | Every training example comes with the correct answer (a *label*) | Each student row carries "Placed" or "NotPlaced" |
| **Classification vs regression** | Classification predicts a category; regression predicts a number (course: "a continuous numerical value") | We predict a category, so it is classification. Logistic *Regression* is classification despite its name (Chapter 7) |
| **Binary classification** | Exactly two classes | Placed (1) and NotPlaced (0) |
| **Features / predictors / independent variables** | The inputs used to make the prediction (the course says "independent variable (input)") | Ten columns: CGPA, Internships, … HSC_Marks |
| **Target / dependent variable / label** | The thing we predict (course: "dependent variable (target)") | `PlacementStatus` |
| **Training data / test data** | Data the model learns from / data held back to judge it fairly | 6,938 rows to learn from; 1,487 held back for the final test |
| **Prediction model** | The saved function that maps the ten inputs to a prediction | `LR_B_final.joblib`, a scikit-learn *Pipeline* |

<div class="note" markdown="1">
**Probability first, label second.** Our model does not just say "Placed / Not placed". It first outputs a probability such as 0.373, and a *threshold* then converts it into a label. This one idea is the key to Chapters 7, 10 and 11.
</div>

<div class="wrap" markdown="1">
**Key takeaways.** Predictive analytics = learn from labelled history, estimate an unknown outcome. Ours is supervised, binary classification, with ten features and one label. It stops at prediction (no prescription).
**Formulas.** None yet.
**Revision questions.** (1) Descriptive vs predictive vs prescriptive: which is our app? (2) Feature vs target? (3) Why is our task classification, not regression? (4) What does "supervised" mean? (5) Why hold back test data?
**Professor may ask.** "Is this prescriptive?" No: it estimates a probability, it does not recommend actions. "Why not just regression?" Our target is a category.
</div>

# 2. Understanding Our Dataset

## 2.1 Where the data came from <span class="tag v">VERIFIED</span>

| Question | Answer | Evidence |
|---|---|---|
| Source | Dataset `Krooz/Campus_Recruitment_CSV` on **Hugging Face** (a public website where people share datasets and models), <https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV> | `README.md`, `app/model_metadata.json` |
| Licence | **CC0-1.0**: "Creative Commons Zero", a public-domain dedication. The creator gives up copyright claims, so anyone may copy, change and use it, even commercially, without asking. *Recorded in project documents; the licence page was not re-checked online for this book.* | `README.md` |
| Real or synthetic? | <span class="tag u">UNKNOWN</span> The dataset card does not say. The data show hard floors/ceilings and very strong effects for every predictor, which is *consistent with* a rule-generated dataset, but that is an observation, not a proof | `src/eda_findings.md` §D |
| Supplied as | Three CSV files of 7,225 + 1,275 + 1,500 = **10,000 rows**, kept unmodified in `data/raw/` | `results/metrics/02_data_quality_audit.csv` |
| Shape | 10,000 rows × 11 columns (10 predictors + 1 target); no missing values | `results/metrics/01_data_quality_summary.csv` |

**Rows vs columns.** A *row* (record, observation) is one student. A *column* (variable, attribute) is one measured property. A table of 10,000 rows × 11 columns holds 110,000 cells.

## 2.2 The ten predictors and the target

The repository has no data dictionary, so the "meaning" column is the conventional reading of the column name. Ranges are observed in the 8,425 model-fitting rows (`app/model_metadata.json`).

| # | Column | Meaning (conventional) | Python type | Kind | Supported values | Treatment in preprocessing |
|---|---|---|---|---|---|---|
| 1 | CGPA | Cumulative grade point average | float | numerical, continuous | 6.5 – 9.1 | StandardScaler |
| 2 | Internships | Number of internships | int | numerical *count*, treated as category | 0, 1, 2 | one-hot, reference level 1 dropped |
| 3 | Projects | Number of projects | int | count, as category | 0, 1, 2, 3 | one-hot, reference 3 dropped |
| 4 | Workshops/Certifications | Number of workshops/certificates | int | count, as category | 0, 1, 2, 3 | one-hot, reference 0 dropped |
| 5 | AptitudeTestScore | Score in an aptitude test | int | numerical | 60 – 90 | StandardScaler |
| 6 | SoftSkillsRating | Rating of soft skills | float | numerical | 3.0 – 4.8 | StandardScaler |
| 7 | ExtracurricularActivities | Takes part in extracurriculars | text | **binary** categorical | No, Yes | one-hot `drop='first'` → "Yes" flag |
| 8 | PlacementTraining | Attended placement training | text | binary categorical | No, Yes | one-hot `drop='first'` → "Yes" flag |
| 9 | SSC_Marks | Class 10 board marks (SSC) | int | numerical | 55 – 90 | StandardScaler |
| 10 | HSC_Marks | Class 12 board marks (HSC) | int | numerical | 57 – 88 | StandardScaler |
| – | **PlacementStatus** | **Target**: "Placed" or "NotPlaced" | text | binary label | Placed → 1, NotPlaced → 0 | not an input; used only as the answer |

**Terms.** *Numerical* variables are measured quantities. *Categorical* variables are labels. *Binary* has two values (Yes/No); *multicategory* has more than two. The three count columns are numbers, but Chapter 3 shows that placement does not rise steadily with them (for workshops: 28.4%, 21.1%, 72.6%, 56.3% placed at 0, 1, 2, 3), so the model treats each count as a **category** (this is "Variant B", Chapter 8).

**Input features versus the target.** Features go *into* the model; the target is what the model must *reproduce*. During training the model sees both; at prediction time (in the app) it sees only the ten features and outputs the estimate of the target. The target must never be used as an input, otherwise the model would just copy the answer (data leakage, Chapter 4).

**Why each feature is plausible.** Marks (CGPA, SSC, HSC) and the aptitude score measure academic and reasoning ability; internships, projects and workshops measure practical exposure; soft skills, extracurriculars and placement training measure employability. Plausible reasons are not proof of cause (Chapter 12).

## 2.3 Limitations of the dataset

* **Unknown provenance** (above). We cannot claim it is verified institutional data.
* **Label quality unknown.** Eight feature profiles carry contradictory labels (Chapter 4).
* **One dataset, one distribution.** Ranges are narrow (CGPA 6.5–9.1), so the app refuses inputs outside them.
* **No sensitive attributes** (gender, caste, income), so fairness cannot be audited (Chapter 16).
* **Rare levels.** Projects = 0 occurs for only 22 of 8,425 rows; its coefficient is unreliable and the app warns about it.

<div class="wrap" markdown="1">
**Key takeaways.** Public CC0 dataset from Hugging Face; 10,000 raw rows × 11 columns; ten predictors (five continuous, three counts, two Yes/No) and one binary target; origin undocumented.
**Formulas.** None.
**Revision questions.** (1) Dataset ID and licence? (2) Which columns are numerical, which categorical? (3) Why are the three count columns treated as categories? (4) Why must the target not be an input? (5) Name two limitations of the data.
**Professor may ask.** "Is it real student data?" Unknown; say so. "What does CC0 mean?" Public-domain dedication: free to reuse.
</div>

# 3. EDA: Exploratory Data Analysis

## 3.1 What EDA is and why we do it first <span class="tag c">COURSE</span>

The course note (EDA.pdf) says: "EDA means understanding the data before building a Machine Learning model." It asks: what is the average value? are there missing values? are there outliers? which features are related? The five steps it teaches are: understand the dataset (rows, columns, types) → check missing values → summary statistics → detect outliers → visualise. We followed that order, and added statistical tests (notebook 04).

Without EDA we would not have discovered the duplicates and contradictions, the non-monotonic count variables (which decided Variant A vs B), or the fact that the data look rule-generated.

## 3.2 The EDA operations that were actually performed <span class="tag v">VERIFIED</span>

| Operation | Pandas / library call in the notebooks | Notebook |
|---|---|---|
| Shape, types, unique values | `.shape`, `.dtypes`, `.nunique()`, `.head(10)` (`.info()` was **not** used) | 01 |
| Missing values | `.isna().sum()` (result: none) | 01, 02 |
| Summary statistics | `.describe(percentiles=[.01,.25,.5,.75,.99])`, `.skew()`, `.kurt()` | 01 |
| Class distribution | `.value_counts()`, `pd.crosstab(..., normalize='index')`, `sns.countplot` | 01 |
| Outlier check | IQR rule with `.quantile([.25,.75])`, **for information only; no row removed** | 01 |
| Histograms (with KDE), box plot | `sns.histplot(..., kde=True)`, `sns.boxplot` | 01 |
| Correlation heat-map | `.corr()` + `sns.heatmap` | 01, 03 |
| Relationship to target | `groupby` placement rates, violin plots, bar charts with Wilson-interval error bars, scatter/joint grids, ECDF step plots | 03 |
| Statistical tests | Mann-Whitney U, Welch t, Cohen's *d*, chi-square, odds ratios, VIF (all in `scipy`/`statsmodels`) | 04 |

**Not done:** pie charts (the course teaches them, we did not use one), imputation (nothing missing), log transforms, binning or any feature creation (`src/eda_findings.md`: "No transformation, binning or feature creation was applied").

## 3.3 The statistics, with one tiny example <span class="tag c">COURSE</span>

For the five marks **40, 50, 60, 70, 80**:

| Statistic | Formula | Value |
|---|---|---|
| Mean μ | Σx / n | (40+50+60+70+80)/5 = **60** |
| Median | middle value when sorted | **60** |
| Mode | most frequent value | none (all appear once) |
| Variance σ² | Σ(x − μ)² / n | (400+100+0+100+400)/5 = **200** |
| Standard deviation σ | √variance | √200 = **14.14** |

The course divides by *n* here (σ = 14.14), which is the **population** formula; pandas `.std()` divides by *n − 1* and would give 15.81. This matters in Chapter 6, because scikit-learn's StandardScaler also divides by *n*.

**Skewness** (course): when the tail is on the right, mean > median (right-skewed); symmetric data have mean = median = mode.

**Our data.** The summary table (`results/metrics/01_descriptive_statistics.csv`) shows, for example, CGPA mean 7.70, aptitude mean 79.3, soft skills mean 4.32; all five score columns have hard ceilings and floors (CGPA starts at exactly 6.5, aptitude ends at 90).

## 3.4 Reading the charts: what they tell us and what they cannot

| Chart | What it shows | Our finding | Cannot conclude |
|---|---|---|---|
| **Bar chart / countplot** | height = count of each category | 5,803 NotPlaced vs 4,197 Placed in the raw file (41.97% minority) | nothing about *why* |
| **Histogram** | how a number is spread over bins | scores have no long tails; piles at the limits | the data are normal; or real |
| **Box plot** | median, quartiles (box) and fences (whiskers) | IQR rule flags **0** outliers in all five score columns | counts (3–4 values) cannot be judged by IQR: it flags 4,458 Internships rows, which is meaningless because IQR = 0 |
| **Violin plot** | the whole distribution for each class | all five scores are higher for Placed students | overlapping shapes mean no clean cut |
| **Heat-map** | colour = correlation, −1 to +1 | largest predictor–predictor correlation r = 0.56 (aptitude–HSC); the eight numeric predictors correlate 0.25–0.52 with the 0/1 target | correlation is not causation |
| **Placement rate by level** | bars with error bars | counts are **non-monotonic** (see §2.2) | the Projects = 0 value rests on 30 students only |

![Target distribution (report figure 1)](masterbook_assets/img/r_target.png)

**Class imbalance.** After cleaning, 41.56% of rows are Placed (4,119 of 9,912). The ratio is about 1.4 : 1, a *moderate* imbalance: accuracy alone would be misleading (always saying "NotPlaced" is 58.4% accurate), so Chapter 10 uses probability-based metrics as well. **Outliers.** Course method: a value is an outlier if it is below Q1 − 1.5·IQR or above Q3 + 1.5·IQR. We applied it for information; no row was deleted, because the course's own rule is "investigate before acting".

**Correlation** is a number from −1 to +1 saying how strongly two variables move together (Pearson *r*). Strong correlation with the target means "useful for prediction", not "cause".

<div class="wrap" markdown="1">
**Key takeaways.** EDA = look before you model. We used shape/dtypes/describe, value counts, IQR, histograms, box/violin/bar plots, heat-maps and tests. EDA led to: dedupe, treat counts as categories, judge by probability quality. No pie chart, no imputation, no feature creation.
**Formulas.** mean = Σx/n; σ² = Σ(x−μ)²/n; outlier fences = Q1 − 1.5·IQR, Q3 + 1.5·IQR.
**Revision questions.** (1) Name three EDA outputs we produced. (2) Why not `.info()`? (it was simply not used; `.dtypes` and `.shape` gave the same facts). (3) What is class imbalance and what is ours? (4) What does r = 0.56 between aptitude and HSC tell us? (5) Why did EDA stop us from treating counts as numbers?
**Professor may ask.** "Did you remove outliers?" No, the IQR check was informational. "Which EDA did you not do?" Pie chart, imputation, transformations.
</div>
