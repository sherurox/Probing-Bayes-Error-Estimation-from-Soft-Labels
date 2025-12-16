# Bayes Error Estimation Under Soft-Label Noise
**Reproducing and extending Ishida et al. (ICLR 2023): when soft-label assumptions break**

This repository reproduces and extends the Bayes error estimation framework proposed by **Ishida et al. (ICLR 2023)**. Our work focuses on how **violations of soft-label assumptions**—specifically **systematic annotation bias** and **model miscalibration**—impact the **consistency** and **reliability** of Bayes error estimators.

We present two controlled experiments:

- **Hypothesis A (Biased Annotators, Synthetic):** Even small systematic bias in soft labels can cause large, persistent distortion in the estimated Bayes error.
- **Hypothesis B (Miscalibrated Models, Fashion-MNIST):** Overconfident neural networks can severely underestimate Bayes error; **temperature scaling** improves alignment between estimated Bayes error and observed test error.

---

## Why this matters

The **Bayes error** (Bayes error rate) is the *irreducible* classification error:

\[
R^* \;=\; \mathbb{E}_x\Big[\min\big(P(y=1\mid x),\, 1 - P(y=1\mid x)\big)\Big]
\]

It is a principled lower bound on achievable error, and it helps answer:

> *“Is my model close to optimal, or am I just overconfident / overfitting?”*

Ishida et al. show that Bayes error can be estimated from **confidence information / soft labels** under assumptions that essentially require soft labels to approximate the true posterior \(c(x)=P(y=1\mid x)\) without systematic bias. This project stress-tests those assumptions in realistic failure scenarios.

---

## Repository contents

```text
.
├── Hypothesis_A.ipynb          # Hypothesis A: synthetic data + biased annotators (Samruddhi)
├── Hypothesis_B.ipynb          # Hypothesis B: Fashion-MNIST + calibration study (Shreyas)
├── report.pdf                  # Project write-up: hypotheses, setup, and results
└── README.md
```

---

## Team contributions

Group members' GitHub account names:
- @sherurox
- @snavale2-001

### Contribution by Shreyas — Hypothesis B & Fashion-MNIST reproduction
Shreyas completed all experiments and code required for **Hypothesis B**, which investigates how neural network **miscalibration** affects soft label-based Bayes error estimation.

Key contributions:
- Reproduced the Fashion-MNIST dataset preprocessing and model training pipeline.
- Trained **four CNN variants** representing different calibration states:
  - Overfitted CNN (**overconfident**)
  - Underfitted CNN (**underconfident**)
  - Baseline CNN
  - Temperature-scaled CNN (**calibrated**)
- Generated soft-label outputs for each model and computed the Bayes error estimate:
  \[
  \widehat{R} = \frac{1}{n}\sum_{i=1}^{n}\min(p_i,\,1-p_i)
  \]
- Compared Bayes error with actual test error to diagnose calibration influence:
  - Overfitted model: Bayes error **underestimated**
  - Underfitted model: Bayes error **closer** to actual
  - Temperature-scaled model: Bayes estimate **aligned** with true error
- Produced plots showing the relationship between calibration and deviation in Bayes error estimation.

### Contribution by Samruddhi — Hypothesis A (Biased annotators)
Samruddhi implemented all experiments related to **Hypothesis A**, which analyzes how systematic annotation bias affects Bayes error estimation.

Key contributions:
- Implemented the simulation pipeline for generating true posteriors using **Gaussian-based synthetic data**.
- Introduced positive soft-label bias to emulate consistently overestimating annotators, e.g.:
  \[
  u_i = \text{clip}(c_i + b,\,0,\,1)
  \]
  (and/or equivalent parameterizations that enforce a systematic shift).
- Computed and compared:
  - True Bayes error
  - Bayes error using **unbiased** soft labels
  - Bayes error using **biased** soft labels
- Demonstrated that even small annotation bias leads to large deviations (e.g., **0.157** true vs. **>0.222** biased estimate), confirming loss of estimator consistency.
- Produced figures and analysis for biased vs. unbiased Bayes error behavior.

---

## Methods (what we implemented)

### Common estimator (used in both experiments)

Given confidence/soft-label values \(u_i \in [0,1]\) intended to represent \(P(y=1\mid x_i)\), we estimate Bayes error via:

\[
\widehat{R} \;=\; \frac{1}{n}\sum_{i=1}^{n}\min(u_i,\,1-u_i)
\]

Interpretation:
- If \(u_i \approx c(x_i)\), then \(\widehat{R}\) approximates the irreducible error \(R^*\).
- If \(u_i\) is **systematically shifted** (annotator bias) or **overconfident** (model miscalibration), then \(\widehat{R}\) can be **systematically distorted**.

---

## Experiment A — Hypothesis A: biased annotators (synthetic, ground-truth Bayes error known)

### Goal
Test whether Bayes error estimation remains reliable when **soft labels are systematically biased**, violating the “unbiased soft label” assumption.

### Data generation
We simulate a binary classification problem where the true posterior is known:
- Class priors: \(\pi_0=\pi_1=0.5\)
- Class-conditionals (Gaussian mixture):
  - \(x\mid y=0 \sim \mathcal{N}(\mu_0, \sigma^2)\)
  - \(x\mid y=1 \sim \mathcal{N}(\mu_1, \sigma^2)\)

True posterior:
\[
c(x)=\frac{\pi_1 p(x\mid y=1)}{\pi_0 p(x\mid y=0)+\pi_1 p(x\mid y=1)}
\]

True Bayes error:
\[
R^*=\mathbb{E}_x[\min(c(x),1-c(x))]
\]

### Soft-label construction
We generate soft labels under two regimes:

**Unbiased soft labels (multi-annotator vote fraction)**
\[
z_{i,a}\sim \text{Bernoulli}(c_i),\qquad
u_i=\frac{1}{A}\sum_{a=1}^{A}z_{i,a}
\]
So \( \mathbb{E}[u_i\mid c_i]=c_i \).

**Biased soft labels (systematic positive bias)**
We emulate annotators who systematically overestimate the positive class, e.g.:
\[
u_i=\text{clip}(c_i+b,\,0,\,1)
\]
(or equivalently by sampling from a biased effective posterior \(c'_i\) that is shifted from \(c_i\)).

### What we measure
- **True Bayes error** from \(c_i\): \(R^*\)
- **Estimated Bayes error** from soft labels:
  - \(\widehat{R}_{\text{unbiased}}\)
  - \(\widehat{R}_{\text{biased}}\)

### Results (obtained)
- Under **unbiased** soft labels, \(\widehat{R}_{\text{unbiased}}\) is close to \(R^*\) (small deviations due to finite sampling).
- Under **biased** soft labels, \(\widehat{R}_{\text{biased}}\) deviates substantially and **does not “wash out”** by simply collecting more samples.
- In our runs, bias produced large distortions (e.g., **0.157** true Bayes error vs. **>0.222** biased estimate).

> **Conclusion (Hypothesis A supported):** Systematic annotation bias breaks the reliability/consistency of soft-label-based Bayes error estimation.

📓 Notebook: `Hypothesis_A.ipynb`

---

## Experiment B — Hypothesis B: model miscalibration (Fashion-MNIST)

### Goal
Test whether neural network probabilities can be used as soft labels for Bayes error estimation, and how **calibration** changes the estimate.

### Dataset
Fashion-MNIST is converted to a **binary** task (two super-classes, e.g., “tops vs. non-tops”). CNNs output logits \(s(x)\), and probabilities are:

\[
p(x)=\sigma(s(x))
\]

### Model variants
We train four CNN variants with intentionally different confidence behavior:

1. **Overfitted / overconfident CNN**
   - Higher capacity and/or more training epochs
   - Minimal regularization  
   Expected: probabilities concentrate near 0/1.

2. **Underfitted / underconfident CNN**
   - Lower capacity and/or fewer epochs  
   Expected: probabilities closer to 0.5.

3. **Baseline CNN**
   - Standard training settings.

4. **Temperature-scaled CNN (calibrated)**
   Post-hoc calibration with temperature \(T\), fit on a validation set:
   \[
   p_T(x)=\sigma\left(\frac{s(x)}{T}\right)
   \]
   \(T\) is selected to minimize validation negative log-likelihood / BCE.

### What we measure
For each model:
- **Test error** (0/1 classification error)
- **Bayes error estimate** from model probabilities (used as soft labels):
  \[
  \widehat{R} = \frac{1}{n}\sum_{i=1}^{n}\min(p_i,\,1-p_i)
  \]
- **Calibration effect** (raw vs temperature-scaled).

### Results (obtained)
- The **overfitted/overconfident** model severely **underestimates** Bayes error, producing unrealistically small \(\widehat{R}\).
- The **underfitted** model produces a Bayes estimate closer to actual error.
- The **temperature-scaled (calibrated)** model yields a Bayes estimate that aligns well with observed test error.
- Example observed behavior from our runs:
  - Overfitted model: Bayes estimate **too low** relative to test error (underestimation)
  - Temperature-scaled model: Bayes estimate **aligned** with test error (improved reliability)

> **Conclusion (Hypothesis B supported):** Raw NN probabilities are not safe soft labels for Bayes error estimation unless the model is properly calibrated.

📓 Notebook: `Hypothesis_B.ipynb`

---

## Summary of findings

### What works
If soft labels are unbiased approximations of the true posterior \(P(y\mid x)\), then
\[
\widehat{R}=\frac{1}{n}\sum_i\min(u_i,1-u_i)
\]
behaves as a meaningful estimate of irreducible error.

### What breaks it
1. **Systematic annotation bias** shifts \(u_i\) away from \(c(x)\) → \(\widehat{R}\) becomes persistently wrong.
2. **Model miscalibration (overconfidence)** collapses probabilities to 0/1 → \(\min(p,1-p)\to 0\) → Bayes error is underestimated.
3. **Temperature scaling** improves probability calibration and yields more credible Bayes error estimates.

---

## How to run

### Requirements
- Python 3.8+
- Jupyter

Install dependencies:
```bash
pip install numpy scipy matplotlib torch torchvision
```

### Run notebooks
```bash
jupyter notebook Hypothesis_A.ipynb
jupyter notebook Hypothesis_B.ipynb
```

---

## Reproducibility notes
- Hypothesis A uses fixed random seeds for consistent synthetic sampling.
- Hypothesis B uses train/val/test splits and fits temperature scaling on the validation set.

---

## References
- Ishida et al., *Estimating the Bayes Error Rate with Confidence Information*, ICLR 2023
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017

---

## Citation
If you build on this repository:
```bibtex
@misc{bayes_error_soft_label_noise_2025,
  title        = {Bayes Error Estimation Under Soft-Label Noise},
  author       = {Khandale, Shreyas and Navale, Samruddhi},
  year         = {2025},
  note         = {Reproduction and extension of Ishida et al. (ICLR 2023)}
}
```

---

## License
MIT License
