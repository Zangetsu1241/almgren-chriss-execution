"""Discrete Almgren-Chriss sell-order execution with arithmetic Brownian midprices.

Costs in USD, inventory in shares, time in trading days. Permanent impact is
charged at half the currently executed order to recover gamma * X**2 / 2.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class ExecutionProblem:
    shares: float = 100_000.0
    intervals: int = 20
    horizon_days: float = 1.0
    volatility: float = 1.5
    temporary_impact: float = 1e-6
    permanent_impact: float = 2e-7
    risk_aversion: float = 3e-5

    def __post_init__(self):
        for name in ('shares','horizon_days','temporary_impact'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('volatility','permanent_impact','risk_aversion'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if isinstance(self.intervals,bool) or not isinstance(self.intervals,(int,np.integer)) or self.intervals < 2:
            raise ValueError('intervals must be an integer >= 2')

    @property
    def dt(self):
        return self.horizon_days / self.intervals


def _verify_trades(trades, problem):
    n = np.asarray(trades, dtype=float)
    if n.shape != (problem.intervals,) or not np.isfinite(n).all() or (n < -1e-8).any():
        raise ValueError('trades must be a finite, nonnegative vector matching intervals')
    if not np.isclose(n.sum(),problem.shares,rtol=1e-9,atol=1e-6):
        raise ValueError('schedule must liquidate all shares')
    return n


def optimal_trades(problem):
    """Closed-form solution of quadratic intermediate inventories."""
    N=problem.intervals
    a=problem.temporary_impact/problem.dt
    b=problem.risk_aversion*problem.volatility**2*problem.dt
    band=np.zeros((3,N-1),dtype=float)
    band[1]=2*a+b
    band[0,1:]=-a
    band[2,:-1]=-a
    rhs=np.zeros(N-1)
    rhs[0]=a*problem.shares
    inside=solve_banded((1,1),band,rhs,check_finite=True)
    inventories=np.r_[problem.shares,inside,0.0]
    trades=-np.diff(inventories)
    trades=np.maximum(trades,0.0)
    trades*=problem.shares/trades.sum()
    return _verify_trades(trades,problem)


def twap_trades(problem):
    return np.full(problem.intervals,problem.shares/problem.intervals)


def vwap_trades(problem,weights=None):
    """Exogenous forecast volume only; not volume observed in the future."""
    if weights is None:
        grid=np.linspace(-1,1,problem.intervals)
        weights=1.0+0.9*grid**2
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(problem.intervals,) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid forecast volume weights')
    return problem.shares*weights/weights.sum()


def stats(trades,problem):
    n=_verify_trades(trades,problem)
    remaining=problem.shares-n.cumsum()
    expected=problem.permanent_impact*problem.shares**2/2 + problem.temporary_impact/problem.dt*np.dot(n,n)
    variance=problem.volatility**2*problem.dt*np.dot(remaining[:-1],remaining[:-1])
    return {'expected_cost_usd':float(expected),'cost_std_usd':float(np.sqrt(variance)),
            'variance_usd2':float(variance),'objective_usd':float(expected+problem.risk_aversion*variance),
            'max_trade_shares':float(n.max()),'total_shares':float(n.sum())}


def monte_carlo_costs(trades,problem,paths=10_000,seed=42,chunk=1024):
    """Brownian midprice costs under permanent and temporary linear impact.

    Execution at S_(i-1) - gamma*(executed_before+n_i/2) - eta*n_i/dt.
    """
    n=_verify_trades(trades,problem)
    if not isinstance(paths,int) or paths<1 or not isinstance(chunk,int) or chunk<1:
        raise ValueError('paths/chunk must be positive integers')
    rng=np.random.default_rng(seed)
    carry=(problem.shares-n.cumsum())[:-1]
    deterministic=stats(n,problem)['expected_cost_usd']
    results=np.empty(paths)
    for start in range(0,paths,chunk):
        m=min(chunk,paths-start)
        z=rng.standard_normal((m,problem.intervals-1))
        results[start:start+m]=deterministic-problem.volatility*np.sqrt(problem.dt)*np.sum(z*carry[None,:],axis=1)
    return results


def schedules(problem):
    return {'Almgren-Chriss':optimal_trades(problem),'TWAP':twap_trades(problem),'Forecast VWAP':vwap_trades(problem)}
