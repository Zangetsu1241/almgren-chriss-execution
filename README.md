# Almgren-Chriss Optimal Execution

Reproducible quantitative research on risk-aware liquidation. Implements the discrete Almgren-Chriss sell-order mean-variance optimum, TWAP and synthetic **forecast VWAP** baselines, Monte Carlo shortfall simulations, and research outputs.

## Mathematical model
Sell X shares over N intervals of width dt = T/N. Trade n_i = x_(i-1) - x_i, with x_0 = X and x_N = 0. Let sigma be unaffected arithmetic-Brownian price volatility, gamma permanent impact and eta temporary impact.

```text
E[C] = gamma*X^2/2 + eta/dt * sum_i n_i^2
Var[C] = sigma^2 * dt * sum_{i=1}^{N-1} x_i^2
J = E[C] + lambda*Var[C]
```

The optimal intermediate inventory satisfies a tridiagonal system:
```text
(2a+b)*x_i - a*x_(i-1) - a*x_(i+1) = 0
a=eta/dt, b=lambda*sigma^2*dt
```

A risk-neutral or zero-volatility optimum reduces to TWAP. Larger risk aversion front-loads trading. Prices in USD, time in trading days, quantities in shares.

## Run
```bash
python -m pip install -e '.[test]'
python -m pytest -q
optimal-execution --shares 100000 --intervals 20 --paths 5000 --output results
```

Results include `schedules.csv`, `metrics.json` and `execution_schedules.png`. Run `optimal-execution --help` to see other parameters.

## Methodology
- Exact tridiagonal solve, fully liquidates inventory
- TWAP and forecast-VWAP (synthetic ex-ante U-shaped volume profile; **never future observed volume**)
- Brownian inventory-risk Monte Carlo, seeded and tested against analytical moments
- Analytic expected impact costs, variance, mean-variance objective and 95th-percentile simulated cost
- 12 numerical, robustness and validation tests; Python 3.10/3.12 GitHub Actions

## Limitations
Impact and volatility parameters are **assumed**, not fitted to market transactions. All experiments are simulated. A live execution system requires lot sizes, bid/ask spread, participation constraints, realized market-volume forecasting, adverse selection, dynamic liquidity, fills, exchange rules, and measured market impact. The model comparison does not imply that Almgren-Chriss has the lowest realized cost on every sample path.

MIT license.
