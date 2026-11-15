# 7. Logistic Regression: the Complete Masterclass

## 7.1 What it is, and why "regression" for classification <span class="tag c">COURSE</span>

The course (Module 1, Dr. Divya Thakur) says: *"Logistic regression … is used for predicting the categorical dependent variable using a given set of independent variables … instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1 … Instead of a straight line, Logistic Regression uses the Sigmoid Function to 'S-shape' the output."*

**Why the odd name?** Historically it belongs to the family of regression methods: it *regresses* (fits a linear equation to) a transformed quantity, the **log-odds** of the class, not the class itself. The output is a number between 0 and 1 (a probability), and a threshold then turns it into a class. So the *model* is a regression of log-odds; its *use* here is classification.

| | Linear regression (course) | Logistic regression (course) |
|---|---|---|
| Predicts | a continuous number (house price) | the probability of a class (Placed) |
| Equation | Y = a + bX + ε, or Y = a + b₁X₁ + … + bₙXₙ + ε | p(y=1\|X) = 1 / (1 + e^−(β₀+β₁x₁+…+βₚxₚ)) |
| Output range | any real number | strictly between 0 and 1 |
| Fitted by | least squares | maximum likelihood (minimise log loss) |
| For our task | would give "placement = 1.3" or "−0.2": meaningless | gives 0.373 |

**Binary logistic regression** (two classes) is what we use. The course also covers multinomial (softmax, three or more unordered classes) and ordinal versions; **we did not use them**: our target has only two classes.

## 7.2 The linear predictor (the "score")

<div class="eq">z = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + b<sub>2</sub>x<sub>2</sub> + … + b<sub>n</sub>x<sub>n</sub></div>

* **x₁ … xₙ** (here n = 15): the processed input values of one student (Chapter 6).
* **b₁ … bₙ** (also called *weights* or *coefficients*; the course writes β or w): numbers the training learns. A bigger positive b means that feature pushes the score up more.
* **b₀** (the *intercept* or bias): the score when all x are 0, i.e. for the **reference student** (every scaled score at its average, Internships = 1, Projects = 3, Workshops = 0, Extracurricular = No, Training = No). Ours: b₀ = **−1.6607**.
* **z**: the **weighted sum**, a single number from −∞ to +∞; it is the log-odds of being Placed (§7.5).

*Relatable picture:* a scoring sheet in which each question gives plus or minus points; z is the total.

## 7.3 Concept card: the sigmoid (logistic) function <span class="tag c">COURSE</span>

* **What.** The S-shaped function that converts any score into a number between 0 and 1.
* **Why.** A raw score z can be −7 or +40; a probability must lie in [0, 1]. The sigmoid is a smooth, monotonic bridge.
* **Formula (course notation).**

<div class="eq">p = σ(z) = 1 / (1 + e<sup>−z</sup>)</div>

*z* = the score; *e* ≈ 2.718 (Euler's number, as in the course slide); *p* = probability of class 1 (Placed). So P(Placed) = p and P(Not Placed) = 1 − p; the two always add to 1.

| z | −4 | −2 | −1 | −0.52 | 0 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|---|
| p = σ(z) | 0.018 | 0.119 | 0.269 | 0.373 | 0.500 | 0.731 | 0.881 | 0.982 |

![Sigmoid, with our typical student marked](masterbook_assets/img/f_sigmoid.png)

*Reading it:* z = 0 gives exactly 0.5 (the 0.50 threshold is the same as "z ≥ 0"); the 0.35 threshold is the same as "z ≥ ln(0.35/0.65) = −0.619". Large positive z saturates near 1; large negative near 0.

<div class="note" markdown="1">
**Course slide 23 says** the sigmoid is used "in logistic regression and neural network activation layers for binary classification". In **our** model there is **no neural network**: no layers, neurons, hidden units or epochs. The sigmoid is simply the **logistic link function** that turns the single linear score into a probability. It is applied once, at the end, to one number.
</div>

## 7.4 Decision rule <span class="tag c">COURSE</span>

The course states the decision boundary "is typically at 0.5: if p(y=1\|X) ≥ 0.5, predict class 1; if < 0.5, predict class 0". Our app uses the same rule with **two** cut-offs (Chapter 11): predict Placed when p ≥ threshold.

## 7.5 Log-odds, odds and the odds ratio

* **Odds** = p / (1 − p): how many times more likely Placed than Not Placed. p = 0.373 → odds = 0.373/0.627 = **0.595** (about 3 : 5).
* **Log-odds (logit)** = ln(odds). Start from p = 1/(1 + e⁻ᶻ): then 1 − p = e⁻ᶻ/(1 + e⁻ᶻ), so p/(1 − p) = eᶻ, hence

<div class="eq">ln( p / (1 − p) ) = z = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + … + b<sub>n</sub>x<sub>n</sub></div>

So the model is *linear in the log-odds*: that is what the "regression" regresses. Check: ln(0.595) = −0.519 ≈ our z.

* **Odds ratio**: if a 0/1 feature changes from 0 to 1, z rises by its coefficient b, and the odds are multiplied by **eᵇ**. For PlacementTraining_Yes, b = +0.9745, eᵇ = **2.65**: other things equal, the odds of Placed are 2.65 times higher for students who had training. For a standardised score, eᵇ is the odds multiplier per +1 SD. These are *associations in this dataset*, not causal effects (Chapter 12).

## 7.6 Maximum likelihood, likelihood, log-likelihood and cross-entropy <span class="tag c">COURSE</span>

**Question:** how do we choose the b's? **Course answer:** by **Maximum Likelihood Estimation (MLE)**: pick the weights that make the *observed* training labels most probable.

1. **Bernoulli model.** Each student's label y ∈ {0, 1} is a coin toss with success probability pᵢ: P(Y = y) = pᵢʸ (1 − pᵢ)¹⁻ʸ. If y = 1 this equals pᵢ; if y = 0 it equals 1 − pᵢ. (Course slide: "Bernoulli probability mass function".)
2. **Likelihood** of the whole training set (students assumed independent), as in the course:

<div class="eq">L(w, b) = Π<sub>i=1</sub><sup>n</sup> p<sub>i</sub><sup>y<sub>i</sub></sup> (1 − p<sub>i</sub>)<sup>1 − y<sub>i</sub></sup>,&nbsp;&nbsp; where p<sub>i</sub> = σ(w·x<sub>i</sub> + b)</div>

3. **Log-likelihood.** A product of thousands of numbers below 1 is astronomically small, so we take logarithms (log of a product = sum of logs): ℓ(w, b) = Σ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ].
4. **Negative log-likelihood = binary cross-entropy = log loss** (the course uses these names for the same quantity). Maximising ℓ is the same as minimising −ℓ; dividing by *n* makes it an average:

<div class="eq">L = − (1/n) Σ<sub>i=1</sub><sup>n</sup> [ y<sub>i</sub> log(p<sub>i</sub>) + (1 − y<sub>i</sub>) log(1 − p<sub>i</sub>) ]</div>

*n* = number of students; *yᵢ* = true label (0/1); *pᵢ* = predicted P(Placed); log = natural log. This is a **cost (loss) function**: one number that says how bad the current weights are.

![Loss for one student against the predicted probability](masterbook_assets/img/f_logloss.png)

**Intuition.** A student who really was Placed (y = 1) contributes −log(p): tiny if p is near 1, huge if the model said p near 0. Confident wrong answers are punished most heavily.

**Worked example 1 (course "Numerical Example 1", likelihood).** Observations y = [1, 1, 0, 1, 0] (three successes, two failures) with a constant p. L(p) = p³(1 − p)². L(0.4) = 0.064 × 0.36 = 0.02304; L(0.5) = 0.125 × 0.25 = 0.03125; **L(0.6) = 0.216 × 0.16 = 0.03456**; L(0.7) = 0.343 × 0.09 = 0.03087. The best p is **0.6 = 3/5**, the observed proportion: MLE picks the value that makes the data most probable.

**Worked example 2 (log loss, four students).** True labels y = [1, 0, 1, 0], predicted p = [0.9, 0.2, 0.6, 0.7]:

| Student | y | p | loss = −[y ln p + (1−y) ln(1−p)] |
|---|---|---|---|
| 1 | 1 | 0.9 | −ln 0.9 = 0.105 |
| 2 | 0 | 0.2 | −ln 0.8 = 0.223 |
| 3 | 1 | 0.6 | −ln 0.6 = 0.511 |
| 4 | 0 | 0.7 | −ln 0.3 = 1.204 |

Mean = (0.105 + 0.223 + 0.511 + 1.204)/4 = **0.511**. Student 4 (the model was fairly confident, and wrong) costs 1.204, more than the other three students put together (0.839). A model that always answered 0.5 would score ln 2 = 0.693. Our final model on the test set: log loss **0.420** (for comparison, on validation the always-the-base-rate dummy scored 0.679).

## 7.7 Optimisation: gradient descent, then L-BFGS

**Gradient descent** <span class="tag c">COURSE</span>: "an iterative optimization algorithm used to minimize a function by adjusting its parameters in the direction of the steepest descent": initialise parameters, compute the gradient, update, repeat until the loss stops decreasing. For logistic regression the course gives the gradients (for the sum loss):

<div class="eq">∂L/∂w = Σ (σ(z<sub>i</sub>) − y<sub>i</sub>) x<sub>i</sub>,&nbsp;&nbsp; ∂L/∂b = Σ (σ(z<sub>i</sub>) − y<sub>i</sub>),&nbsp;&nbsp; w ← w − η ∂L/∂w,&nbsp;&nbsp; b ← b − η ∂L/∂b</div>

η = learning rate (step size). **Worked example 3 (the course's own, reproduced and checked in Python).** Data (x, y) = {(0, 0), (1, 1)}, η = 0.5, start w = 0, b = 0.

| Iteration | σ(z) for x=0, x=1 | ∂L/∂w | ∂L/∂b | new w | new b |
|---|---|---|---|---|---|
| 1 | 0.5, 0.5 | (0.5−0)·0 + (0.5−1)·1 = −0.5 | (0.5−0) + (0.5−1) = 0 | 0 − 0.5(−0.5) = **0.25** | 0 |
| 2 | 0.5, 0.5622 | (0.5)(0)+(0.5622−1)(1) = −0.4378 | 0.5 + (0.5622−1) = 0.0622 | 0.25 + 0.2189 = **0.4689** | 0 − 0.0311 = **−0.0311** |

After two iterations w ≈ 0.4689, b ≈ −0.0311, the course's answer. The loss falls each step.

**What our model actually uses: L-BFGS** <span class="tag s">SUPPLEMENT</span> <span class="tag v">VERIFIED</span>. Plain gradient descent uses only the slope. **L-BFGS** (limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a *quasi-Newton* method: it remembers the last few steps and gradients to estimate the curvature of the loss and take much better-aimed steps, so it needs far fewer iterations. In scikit-learn 1.9.1, `solver='lbfgs'` calls SciPy's `L-BFGS-B` routine with `gtol = tol = 1e-4` and `maxiter = max_iter` (read from `sklearn/linear_model/_logistic.py`).

<div class="note" markdown="1">
**L-BFGS is an optimisation solver, not an activation function.** An activation function (sigmoid, ReLU) is part of the *model's formula* and transforms a score into an output. A solver is the *search procedure* that finds the weights. They answer different questions: "what is the formula?" (linear score + sigmoid) versus "how do we find the best b's?" (L-BFGS).
</div>

* **max_iter = 5000** is a safety cap on iterations. **Convergence** means the gradient became tiny (below tol) so the search stopped by itself. Our fitted model has `n_iter_ = [30]`: it converged after **30** iterations, far below 5,000. I checked independently that the gradient of the objective at the saved coefficients has maximum size 6.2 × 10⁻⁵ < 10⁻⁴.
* **Iteration ≠ epoch ≠ fit.** One *iteration* = one L-BFGS update step inside one fit. An *epoch* = a full pass over the data in mini-batch training of neural networks: **no epochs exist here**. A *fit* = one complete `.fit()` call (which itself contains many iterations). See Chapter 9 for counts.

## 7.8 Regularisation: L1, L2 and C <span class="tag s">SUPPLEMENT</span>

**Overfitting and the penalty.** With many features, the unrestricted MLE weights can become very large to chase noise. **Regularisation** adds a penalty for large weights to the loss, so the optimiser trades fit against size.

| | L2 (ridge) | L1 (lasso) |
|---|---|---|
| Penalty on weights | ½ Σ w_j² (sum of squares) | Σ \|w_j\| (sum of absolute values) |
| Effect | shrinks all weights smoothly, rarely to exactly 0 | can set weights to exactly 0 (feature selection) |
| scikit-learn 1.9.1 | `l1_ratio=0` (the old `penalty='l2'`) | `l1_ratio=1` (needs `liblinear` or `saga` solver) |
| **Ours** | **yes, L2** (L-BFGS supports only L2 or none) | no |

**The exact objective scikit-learn minimises** (I read the source: `l2_reg_strength = 1.0 / (C * sw_sum)`, intercept not penalised; and verified numerically that the saved coefficients satisfy the gradient condition with the intercept unpenalised, whereas they would fail it if it were penalised):

<div class="eq">minimise over (w, b):&nbsp; (1/n) Σ<sub>i</sub> loss(y<sub>i</sub>, p<sub>i</sub>) + (1 / (2 C n)) ‖w‖²&nbsp;&nbsp;&nbsp; ⇔ &nbsp;&nbsp; C Σ<sub>i</sub> loss(y<sub>i</sub>, p<sub>i</sub>) + ½ ‖w‖²</div>

‖w‖² = w₁² + … + w₁₅² (the squared lengths of the 15 coefficients; **b₀ is excluded**). The two forms have the same minimiser (multiply by n·C).

**C is the *inverse* of the regularisation strength:** *small C → strong penalty → smaller weights (simpler model); large C → weak penalty → weights near the unpenalised values.* (C = ∞ means no penalty.) In our case C = 1.

**Numbers for our model** (n = 8,425): Σ loss = 3,623.8 and ½‖w‖² = 1.52, so the penalty is only 0.04% of the objective: with 8,425 rows and 15 columns, C = 1 is a **gentle** regularisation. That matches the sensitivity check on validation data (`05_lr_C_sensitivity.csv`, Variant B, AUC): C = 0.01 → 0.8692; 0.1 → 0.8714; 1 → 0.8717; 10 → 0.8718; 100 → 0.8718. Results are almost identical for C ≥ 0.1: the choice is not delicate.

**Why standardisation matters here:** the penalty adds up w_j² over all columns, so it is only even-handed if all inputs share a scale.

## 7.9 Our coefficients, and how to read them <span class="tag v">VERIFIED</span>

From `LR_B_final.joblib` (intercept b₀ = −1.6607). "OR" = eᵇ.

| Column | b | OR | Read as |
|---|---|---|---|
| CGPA | +0.233 | 1.26 | per +1 SD (0.639 CGPA points) |
| AptitudeTestScore | +0.501 | 1.65 | per +1 SD (8.16 points) |
| SoftSkillsRating | +0.271 | 1.31 | per +1 SD (0.413) |
| SSC_Marks | +0.257 | 1.29 | per +1 SD (10.4 marks) |
| HSC_Marks | +0.212 | 1.24 | per +1 SD (8.9 marks) |
| Internships_0 / _2 | +0.201 / +0.193 | 1.22 / 1.21 | vs 1 internship |
| Projects_0 / _1 / _2 | +0.615 / −0.473 / −0.315 | 1.85 / 0.62 / 0.73 | vs 3 projects (Projects_0 rests on only 22 students) |
| Workshops_1 / _2 / _3 | −0.322 / +0.449 / +0.030 | 0.73 / 1.57 / 1.03 | vs 0 workshops |
| ExtracurricularActivities_Yes | +0.721 | 2.06 | vs No |
| PlacementTraining_Yes | **+0.974** | **2.65** | vs No |

**Cautions.** Each coefficient assumes the *other columns are held fixed*. Count-level coefficients are relative to their reference level, so a negative sign (Projects_1) means "worse than 3 projects", not "projects hurt". The dataset's non-monotonic patterns mean these coefficients are descriptive. Associations only.

## 7.10 Worked examples with our actual model <span class="tag v">VERIFIED</span>

*Numbers computed from the saved artifact with `decision_function` and `predict_proba` and hand-checked.*

**Example A: the course's customer (Logistic Regression – Example 1).** β₀ = −4, income coefficient β₁ = 0.02, age coefficient β₂ = 0.3; annual income $50,000 (the slide plugs in 50, i.e. thousands) and age 30. z = −4 + 0.02 × 50 + 0.3 × 30 = −4 + 1 + 9 = **6**, and P = 1/(1 + e⁻⁶) = 1/(1 + 0.00248) = **0.9975**, which the slide reports as "approximately 99.75%". The same recipe, with 15 inputs instead of 2, is what our app does.

**Example B: the typical profile in our app** (CGPA 7.7, 1 internship, 2 projects, 1 workshop, aptitude 80, soft skills 4.4, extracurricular Yes, training Yes, SSC 70, HSC 73). Non-zero terms (x × coefficient):

| Term | x | b | x·b |
|---|---|---|---|
| intercept | – | – | −1.6607 |
| CGPA, Aptitude, SoftSkills, SSC | +0.004, +0.083, +0.194, +0.083 | 0.233, 0.501, 0.271, 0.257 | +0.0009, +0.0417, +0.0524, +0.0212 |
| HSC | −0.162 | 0.212 | −0.0342 |
| Projects_2 (=1) | 1 | −0.315 | −0.3148 |
| Workshops_1 (=1) | 1 | −0.322 | −0.3216 |
| Extracurricular_Yes | 1 | 0.721 | +0.7210 |
| PlacementTraining_Yes | 1 | 0.974 | +0.9745 |
| **z (sum)** | | | **−0.5195** |

P = 1/(1 + e^0.5195) = 1/(1 + 1.681) = **0.373 → 37.3%**, the number the app shows. At threshold 0.35 → Placed; at 0.50 → Not Placed. Raising CGPA to 8.45 (z-score +1.178) adds 1.178 × 0.233 = +0.275, so z = −0.2456 and P = **0.439** (43.9%, as the app shows).

**Example C: a weaker and a stronger student.**

| Profile | z | P(Placed) |
|---|---|---|
| CGPA 6.8, 0 internships, 1 project, 0 workshops, aptitude 68, soft 3.8, Extra No, Training No, SSC 58, HSC 64 | −3.82 | **0.021** (2.1%) |
| CGPA 8.6, 2 internships, 3 projects, 2 workshops, aptitude 88, soft 4.6, Yes, Yes, SSC 80, HSC 82 | +2.17 | **0.898** (89.8%) |

Rebuilding the first in code:

```python
import math
z = -1.6607 + sum(x*b for x, b in zip(x_row, coefs))   # x_row: the 15 processed values
p = 1 / (1 + math.exp(-z))                              # identical to pipe.predict_proba(df)[0, 1]
```

## 7.11 Assumptions, limits, and our verified configuration

* **Linear in the log-odds.** Each feature adds a fixed amount to z; interactions (for example Extracurricular × Training) are *not* modelled unless created. The count columns are dummy-coded precisely to relax the straight-line assumption.
* **Independent rows**, no extreme multicollinearity (all VIF < 2 in notebook 04), enough rows per class. **Not guaranteed to be calibrated** unless checked (Chapter 10).
* **Output is an estimate for a group of similar students, not a certainty.**

| Setting | Required value | Found in artifact | Source |
|---|---|---|---|
| Model | Logistic Regression B | `Pipeline(prep, model)` with `LogisticRegression` | `LR_B_final.joblib` |
| Regularisation | L2 | `l1_ratio=0` (`penalty='deprecated'` in 1.9.1) | artifact |
| C | 1 | `C=1.0` | artifact, `final_model_configs.json` |
| Solver | lbfgs | `solver='lbfgs'` | artifact |
| max_iter | 5000 | `max_iter=5000` | artifact |
| Class weighting | none | `class_weight=None` | artifact |
| Iterations used | 30 | `n_iter_=[30]` | artifact |
| Processed features | 15 | `n_features_in_=15`, 15 coefficients | artifact |
| Other | | `tol=1e-4`, `fit_intercept=True`, `random_state=42`, classes `[0, 1]` | artifact |
| Rows fitted | 8,425 | `n = 8425` | `final_model_configs.json`, metadata |

<div class="wrap" markdown="1">
**Key takeaways.** Logistic regression = linear score z, sigmoid to probability, MLE/log-loss to fit, L2 penalty with C = 1 to stay stable, L-BFGS to optimise (30 iterations). It is not a neural network. Odds multiply by eᵇ.
**Formulas.** z = b₀ + Σbⱼxⱼ; p = 1/(1+e⁻ᶻ); ln(p/(1−p)) = z; L = −(1/n)Σ[y log p + (1−y) log(1−p)]; objective = (1/n)Σ loss + ‖w‖²/(2Cn).
**Revision questions.** (1) Why "regression"? (2) What does the sigmoid do and does our model have an "activation layer"? (3) Smaller C means…? (4) L-BFGS vs sigmoid? (5) Compute P for z = 1 (0.731).
**Professor may ask.** "Why not the closed-form solution?" There is none for logistic regression; it needs iteration. "What is n_iter_?" Optimiser iterations (30), not epochs. "Does L2 penalise the intercept?" No.
</div>
