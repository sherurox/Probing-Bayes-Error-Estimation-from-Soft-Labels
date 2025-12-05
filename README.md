Project: Bayes Error Estimation Under Soft-Label Noise
------------------------------------------------------

This project reproduces and extends the Bayes error estimation framework proposed 
by Ishida et al. (ICLR 2023). Our work focuses on examining how violations of the 
soft-label assumptions, specifically annotation bias and model miscalibration,
impact the consistency and reliability of Bayes error estimators.

TEAM CONTRIBUTIONS

Group members' GitHub account names:
- @sherurox
- @snavale2-001

**1. Contribution by Shreyas – Hypothesis B & Fashion-MNIST Reproduction**
----------------------------------------------------------------------
Shreyas completed all experiments and code required for **Hypothesis B**, which 
investigates the role of neural network miscalibration in soft label-based Bayes 
error estimation.

Key contributions:
- Reproduced the Fashion-MNIST dataset preprocessing and model training pipeline.
- Trained four CNN variants representing different calibration states:
    • Overfitted CNN (overconfident)  
    • Underfitted CNN (underconfident)  
    • Baseline CNN  
    • Temperature-scaled CNN (calibrated)
- Generated soft-label outputs for each model and computed the Bayes error 
  estimate: BER = (1/n) Σ min(pi, 1 − pi).
- Compared Bayes error with actual test error to diagnose calibration influence:
    • Overfitted model: Bayes error underestimated  
    • Underfitted model: Bayes error close to actual  
    • Temperature-scaled model: Bayes estimate aligned with true error
- Produced all plots showing the relationship between calibration and deviation 
  in Bayes error estimation.


**2. Contribution by Samruddhi – Hypothesis A (Biased Annotators)**
---------------------------------------------------------------
Samruddhi implemented all experiments related to **Hypothesis A**, which analyses 
How systematic annotation bias affects Bayes error estimation.

Key contributions:
- Implemented the simulation pipeline for generating true posteriors using 
  Gaussian-based synthetic data.
- Introduced positive soft-label bias (ui = clip(ci + b, 0, 1)) to emulate 
  consistently overestimating annotators.
- Computed and compared:
    • True Bayes error  
    • Bayes error using unbiased soft labels  
    • Bayes error using biased soft labels  
- Demonstrated that even small annotation bias leads to large deviations 
  (e.g., 0.157 true vs. >0.222 biased estimate), confirming loss of estimator 
  consistency.
- Produced all figures and analysis for biased vs. unbiased Bayes error behaviour.


------------------------------------------------------
