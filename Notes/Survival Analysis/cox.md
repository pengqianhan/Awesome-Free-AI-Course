# Proportional Hazards Model
## A Survival Analysis Overview
**Author**: Pengqian Han  
**Date**: March 25, 2025
**Source**: [Cox Proportional Hazards Model](https://en.wikipedia.org/wiki/Proportional_hazards_model)

---

## Slide 1: Survival Models: Basics
- **Goal**: Relate time to an event (e.g., death, failure) to covariates.
- **Hazard Rate**: Probability of event in $[t, t+dt)$ given survival to $t$.
- **Components**:
  - Baseline hazard: $\lambda_0(t)$ (time-dependent risk).
  - Covariate effects: Multiplicative scaling of hazard.

**Hazard Formula**:  
$\lambda(t|X) = \lambda_0(t) \cdot \text{effect of covariates}$

---

## Slide 2: Proportional Hazards: Core Idea
- **Assumption**: Covariates scale hazard multiplicatively.
- To start, suppose we only have a single covariate, \( x \), and therefore a single coefficient, \( \beta_1 \). Our model looks like:

\[
\lambda(t|x) = \lambda_0(t) \exp(\beta_1 x)
\]
- Consider the effect of increasing \( x \) by 1:

\[
\begin{aligned}
\lambda(t|x+1) &= \lambda_0(t) \exp(\beta_1 (x+1)) \\
&= \lambda_0(t) \exp(\beta_1 x + \beta_1) \\
&= (\lambda_0(t) \exp(\beta_1 x)) \exp(\beta_1) \\
&= \lambda(t|x) \exp(\beta_1)
\end{aligned}
\]
- **Proportion**:  
  $\frac{\lambda(t|x+1)}{\lambda(t|x)} = \exp(\beta_1)$ (constant over time)


---

## Slide 4: Cox Proportional Hazards Model
- **Introduced by**: Sir David Cox (1972).
- **Form**: For covariates $X_i = (X_{i1}, \dots, X_{ip})$:  
  $\lambda(t|X_i) = \lambda_0(t) \exp(\beta_1 X_{i1} + \cdots + \beta_p X_{ip}) = \lambda_0(t) \exp(X_i \cdot \beta)$
- **Key Feature**: Estimates $\beta$ without specifying $\lambda_0(t)$.
- **Hazard Ratio**:  
  $\frac{\lambda(t|X_i)}{\lambda(t|X_j)} = \exp((X_i - X_j) \cdot \beta)$.

------
## Slide 3: Example: Binary Covariate (Hospital)
- **Data**: Survival post-surgery (A vs. B).
- Model: $\lambda(t|X_i) = \lambda_0(t) \exp(\beta_1 X_i)$, $X_i = 1$ (A), $0$ (B).
- Estimate: $\beta_1 = 2.12$.
- Hazard Ratio: $\exp(2.12) = 8.32$.
- **Interpretation**: Hospital A has 8.3x higher risk than B.

---

## Slide 4: Absence of Intercept Term
- Typical regression: Includes intercept $\beta_0$.
- Cox model: $\lambda_0(t)$ absorbs intercept.
- If $\beta_0$ added:  
  $\lambda(t|X_i) = \lambda_0(t) \exp(X_i \cdot \beta + \beta_0) = [\exp(\beta_0) \lambda_0(t)] \exp(X_i \cdot \beta)$
- Redefine $\lambda_0^*(t) = \exp(\beta_0) \lambda_0(t)$ $\Rightarrow$ Unidentifiable.

---

## Slide 5: Extensions and Variations
- **Time-Varying Covariates**: $\lambda(t|X_i(t))$.
- **Time-Varying Coefficients**: $\beta(t)$.
- **Parametric Models**: Specify $\lambda_0(t)$ (e.g., Weibull).
- **LASSO**: High-dimensional covariate selection.

---

## Slide 6: Software Implementations
- R: `coxph()` (survival package).
- Python: `CoxPHFitter` (lifelines).
- SAS: `phreg`.
- Stata: `stcox`.
- MATLAB: `fitcox`.

---

## Summary
- **Cox Model**: Flexible, semi-parametric survival analysis.
- **Strength**: Focus on covariate effects via hazard ratios.
- **Limitation**: Assumes proportional hazards.
- **Applications**: Medicine, economics, engineering.

---