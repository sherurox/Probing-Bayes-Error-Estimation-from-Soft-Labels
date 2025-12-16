# Bayes Error Estimation Under Biased Soft Labels and Miscalibrated Models
**A reproducible study of when Bayes-error estimators succeed vs. fail in practice**

This repository contains two controlled experiments showing that Bayes error estimation methods that rely on **soft labels / confidence values** can **break** when the soft labels are **systematically biased** (human annotators) or **miscalibrated** (overconfident neural networks). When the assumptions hold, the estimator behaves as expected; when they don’t, the estimate can become **persistently wrong** (bias case) or **artificially near-zero** (overconfidence case).

---

## Why this matters

The **Bayes error** (a.k.a. Bayes error rate) is the *irreducible* classification error:

\[
R^* \;=\; \mathbb{E}_x\Big[\min\big(P(y=1|x),\, 1 - P(y=1|x)\big)\Big]
\]

It is a principled floor on achievable error. Recent work (e.g., Ishida et al., ICLR 2023) provides **simple estimators** of Bayes error using **confidence information / soft labels**, with theoretical guarantees under a key assumption:

> Observed soft labels are (approximately) **unbiased** estimates of the true posterior \(c(x)=P(y=1|x)\).

This project tests what happens when reality violates that assumption.

---

## Repository contents

```text
.
├── Hypothesis_A.ipynb          # Experiment A: synthetic data + biased annotators
├── Hypothesis_B.ipynb          # Experiment B: Fashion-MNIST + model calibration
├── report.pdf                  # Hypotheses, setup, and results (write-up)
└── README.md
```

---

## Methods (what we implemented)

### Common estimator (used in both experiments)

Given confidence/soft-label values \(u_i \in [0,1]\) intended to represent \(P(y=1|x_i)\), we estimate Bayes error via:

\[
\widehat{R} \;=\; \frac{1}{n}\sum_{i=1}^{n}\min(u_i,\,1-u_i)
\]

Interpretation:
- If \(u_i \approx c(x_i)\), then \(\widehat{R}\) approximates the irreducible error \(R^*\).
- If \(u_i\) is **systematically biased** or **overconfident**, \(\widehat{R}\) can be **systematically distorted**.

This repo is specifically about **stress-testing the assumptions** behind using \(u_i\) as a proxy for \(c(x_i)\).

---

## Experiment A — Biased annotators (Synthetic, ground-truth Bayes error known)

### Goal
Test whether Bayes error estimation remains reliable when **annotator soft labels are systematically biased**, violating the zero-mean/no-bias assumption.

### Data generation
We simulate a binary classification problem where the true posterior is known:
- Class priors: \(\pi_0=\pi_1=0.5\)
- Class-conditionals (1D Gaussian mixture):
  - \(x|y=0 \sim \mathcal{N}(\mu_0, \sigma^2)\)
  - \(x|y=1 \sim \mathcal{N}(\mu_1, \sigma^2)\)

True posterior:
\[
c(x)=\frac{\pi_1 p(x|y=1)}{\pi_0 p(x|y=0)+\pi_1 p(x|y=1)}
\]

True Bayes error (computable exactly from simulated \(c(x)\)):
\[
R^*=\mathbb{E}_x[\min(c(x),1-c(x))]
\]

### Soft-label construction (multi-annotator setting)
We simulate **A annotators** per example.

**Unbiased annotators**
\[
z_{i,a}\sim \text{Bernoulli}(c_i),\qquad
u_i=\frac{1}{A}\sum_{a=1}^{A}z_{i,a}
\]
This satisfies \( \mathbb{E}[u_i|c_i]=c_i \).

**Biased annotators (systematic bias)**
We introduce a bias toward the positive class using:
\[
c'_i = (1-\alpha)c_i + \alpha b
\]
where:
- \(b \in (0,1)\) is the “bias target” probability (e.g., a preference for the positive class)
- \(\alpha\in[0,1]\) controls bias strength

Annotators now sample:
\[
z_{i,a}\sim \text{Bernoulli}(c'_i),\qquad
u_i=\frac{1}{A}\sum_{a=1}^{A}z_{i,a}
\]

### What we measure
- **True Bayes error** from \(c_i\): \(R^*\)
- **Estimated Bayes error** from soft labels:
  - Unbiased: \(\widehat{R}_{\text{unbiased}}\)
  - Biased: \(\widehat{R}_{\text{biased}}\)
- **Noiseless biased baseline** (bias-only; infinite-annotator limit):
  \[
  \widehat{R}_{\text{biased,noiseless}}=\frac{1}{n}\sum_i \min(c'_i, 1-c'_i)
  \]

### Results (observed)
- **Unbiased case:** \(\widehat{R}_{\text{unbiased}}\) closely matches the true Bayes error.
- **Biased case:** \(\widehat{R}_{\text{biased}}\) deviates substantially.
- **Bias-only baseline:** \(\widehat{R}_{\text{biased,noiseless}}\) remains shifted, showing the error is **persistent** (bias, not variance).

Example numbers from our run (see `report.pdf` / notebook output):
- True Bayes error ≈ **0.157**
- Unbiased estimate ≈ **0.154**
- Biased estimate ≳ **0.222**

> **Key takeaway:** Bayes error estimators are consistent under unbiased noise, but are **not robust to systematic bias** in soft labels.

📓 Notebook: `Hypothesis_A.ipynb`

---

## Experiment B — Miscalibrated model probabilities (Fashion-MNIST, real data)

### Goal
Test whether using **model probabilities** as soft labels is valid—and how **calibration** changes Bayes error estimates.

### Dataset
Fashion-MNIST is converted to a **binary** task (two super-classes, e.g., “tops vs. non-tops”).
We train CNNs to output logits \(s(x)\), with probability:
\[
p(x)=\sigma(s(x))
\]

### Models trained (intentionally different calibration behavior)
We train multiple variants to create distinct probability/confidence profiles:

1. **Overfitted / Overconfident model**
   - Strong capacity
   - Many epochs
   - Little/no regularization  
   Expected: probabilities collapse toward 0/1.

2. **Base model**
   - Standard training recipe  
   Expected: moderate confidence.

3. **Underfitted model**
   - Reduced capacity and/or fewer epochs  
   Expected: more uncertainty (probabilities closer to 0.5).

4. **Temperature-scaled model (calibrated)**
   Post-hoc calibration: scale logits by \(T\) (fit on validation set):
   \[
   p_T(x) = \sigma\left(\frac{s(x)}{T}\right)
   \]
   Choose \(T\) to minimize validation negative log-likelihood / BCE.

### What we measure
For each model:
- **Test error** (0/1 classification error)
- **Bayes error estimate from model probabilities**
  \[
  \widehat{R}_{\text{model}}=\frac{1}{n}\sum_{i=1}^{n}\min(p_i,1-p_i)
  \]
- **Effect of calibration** (compare raw vs temperature-scaled)

### Results (observed)
- The **overfitted / overconfident** model produces a very small \(\widehat{R}\), implying “almost no irreducible error,” which is not credible.
- **Temperature scaling** increases the realism of probabilities and yields a Bayes estimate that is far more sensible.
- Underfitting increases uncertainty and increases \(\widehat{R}\), as expected.

Example numbers from our run (see `report.pdf` / notebook output):
- Overfitted: test_err ≈ **0.0053**, Bayes_est ≈ **0.0009** (severely underestimated)
- Temperature-scaled: test_err ≈ **0.0072**, Bayes_est ≈ **0.0074** (aligned)

> **Key takeaway:** Treating raw NN probabilities as soft labels can make Bayes error estimates **misleadingly optimistic** unless the model is **well-calibrated**.

📓 Notebook: `Hypothesis_B.ipynb`

---

## Summary of findings

### What works
If soft labels are unbiased approximations of \(P(y|x)\), then
\[
\widehat{R}=\frac{1}{n}\sum_i\min(u_i,1-u_i)
\]
behaves as a meaningful estimate of irreducible error.

### What breaks it
1. **Systematic annotator bias** shifts soft labels \(u_i\) away from \(c_i\) → estimate becomes persistently wrong.
2. **Model miscalibration (overconfidence)** collapses probabilities to 0/1 → \(\min(p,1-p)\to 0\) → Bayes error is underestimated.
3. **Calibration (temperature scaling)** counteracts overconfidence and improves estimate credibility.

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
- Experiment A uses fixed random seeds for consistent synthetic sampling.
- Experiment B uses train/val/test splits and fits temperature scaling on validation.

---

## Suggested extensions (easy wins)
- Multiclass extension of the estimator and experiments
- Class-dependent label noise / annotator-specific bias models
- Add calibration diagnostics (ECE, reliability diagrams) to connect calibration ↔ Bayes estimate distortion
- Compare temperature scaling with other calibration methods (Platt scaling, isotonic regression)

---

## References
- Ishida et al., *Estimating the Bayes Error Rate with Confidence Information*, ICLR 2023
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017

---

## License
MIT License
