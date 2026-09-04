# Market Risk Volatility, Regime Modeling, and Portfolio Engine

This project presents a simulation and optimization framework for market risk. We first look at how one can assess market risk. Then we explore whether we can accurately model and predict market risk using XGBoost and HMM. Finally, we focus on using Monte Carlo to estimate Value at Risk (VaR) and Expected Shortfall (ES/CVaR), as well as XGBoost-based forward volatility forecasting, to devise a CVaR-minimizing portfolio optimizer benchmarked against Net Present Value (NPV) over a multi-year investment horizon.

## Motivation

I wanted to see whether the same instincts used for energy systems modeling can translate to market risk. The parallels turned out to be closer than I expected. Quantifying tail risk in a stochastic PEM electrolyzer model and quantifying tail risk in a stochastic equity portfolio are both, at their core, questions about the shape of a distribution's worst outcomes rather than its mean. Building a predictive tool was not something I was heavily focused on, as markets aren't physical systems per se that I can base on first principles. But I was still interested to see if it was possible by taking a more black box approach. I think what was most intriguing was the idea that I can build a framework honest about uncertainties. This project, and the debugging process behind it, is where that translation started to feel real to me, and it's part of what pointed me toward quantitative risk and decision analytics as a field I want to work in.

## Mathematics behind the work:

### Monte Carlo Value at Risk & Expected Shortfall/CVaR

For a portfolio of $n$ assets with weight vector $w$ and simulated daily return scenarios $r_i \sim \mathcal{N}(\mu, \Sigma)$, portfolio loss for scenario $i$ is $L_i = -w^\top r_i$. At confidence level $\alpha$, the Value at Risk is the smallest loss threshold not exceeded with probability $1-\alpha$:

$$\text{VaR}_\alpha = \inf \{ l \in \mathbb{R} : P(L > l) \le 1 - \alpha \}$$

Expected Shortfall or Conditional Value at Risk (ES/CVaR) is the expected loss *conditional* on being in the tail beyond that threshold:

$$\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$

### Forward-Looking Volatility via XGBoost

Rather than feeding the simulation trailing (realized) volatility, we forecast each asset's forward volatility with a gradient-boosted regressor trained on lagged realized-volatility features. We combine the forecasted volatilities with the historical correlation matrix $\rho$ to construct a forward-looking covariance matrix.

### CVaR-Minimizing Portfolio Optimization (Rockafellar–Uryasev)

Rather than a mean-variance objective, portfolio weights are solved by directly minimizing CVaR over the simulated scenario set. The Rockafellar–Uryasev formulation replaces the non-smooth CVaR objective with an equivalent linear program by introducing an auxiliary threshold variable and per-scenario slack variables.

References here: 
- [Risk-averse optimization: Linear Programming implementation of CVaR](https://hal.science/hal-04966655v2/document)
- [Optimization of CVaR](https://www.pacca.info/public/files/docs/public/finance/Active%20Risk%20Management/Uryasev%20Rockafellar-%20Optimization%20CVaR.pdf)
- [Statistical Odds and Ends: CVaR and a lemma from Rockafellar & Uryasev](https://statisticaloddsandends.wordpress.com/2024/04/15/cvar-and-a-lemma-from-rockafellar-uryasev/)

### Net Present Value of Portfolio Strategies

Portfolio allocations are also evaluated the way a capital project would be: multi-year value paths are simulated via compounded Monte Carlo returns, and the expected incremental value each year is discounted back to the present at rate $r$:

$$\text{NPV} = \sum_{t=1}^{T} \frac{\mathbb{E}[\Delta V_t]}{(1+r)^t} - V_0 $$

### Explaining Optimizer Behavior: Risk Contribution & the Exact CVaR Gradient

Because the optimizer carries no explicit return floor, resulting weights reflect an implicit trade-off between marginal return and marginal tail risk that isn't visible from any single asset-level statistic. Portfolio variance decomposes exactly (an identity, not an estimate) into each asset's marginal risk contribution:

$$\sigma_p = \sum_{i=1}^{n} w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

But the true, exact marginal contribution to *CVaR* (objective function) is a different quantity entirely:

$$\frac{\partial\, \text{CVaR}_\alpha}{\partial w_i} = \mathbb{E}\left[-r_i \,\middle|\, L \ge \text{VaR}_\alpha \right]$$

## Key Findings:

- **CVaR minimization is not risk minimization in isolation:** With no return floor in the objective, individual asset weights can't be explained by correlation, volatility, or variance-based risk contribution alone. Each is only part of the picture the optimizer is actually solving.
- **Diversification value requires the right lens:** Correlation heatmaps and forecasted-vs-realized volatility comparisons each ruled out plausible-sounding explanations for specific allocations before the tail-conditional expected return (the exact CVaR gradient) gave a direct, assumption-free answer.
- **CVaR-optimal allocations outperformed equal-weighting on expected NPV:** across tested asset universes and confidence levels, though the magnitude was sensitive to the mean-return estimation window and to per-asset weight caps, underscoring how much of portfolio optimization is really estimation-uncertainty management.

## Future Work: 
The current model focuses on CVaR or ES minimization. However, given the Hidden Markov Model's capability to track unobservable market phases such as calm, choppy, or stressed regimes, it would allow investors to dynamically adjust asset weights based on shifting market conditions. CVaR acts as a tail-risk constraint framework. Honestly, from an engineering standpoint, it's not bad in terms of practicing the application of objective function minimization relative to our desired solution. However, HMM would allow for a dynamic optimization approach, which is more practical for this application. 

## Languages used for this project:
* **Languages:** Python
* **Libraries:** NumPy, pandas, SciPy, scikit-learn, XGBoost, yfinance, seaborn, matplotlib
* **Environment:** Jupyter Notebooks
