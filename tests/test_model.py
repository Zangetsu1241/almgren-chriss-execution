from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import minimize
from optimal_execution.model import (ExecutionProblem, optimal_trades, twap_trades,
    vwap_trades, stats, monte_carlo_costs, schedules)


def test_zero_risk_optimal_is_twap():
    p=ExecutionProblem(risk_aversion=0)
    assert np.allclose(optimal_trades(p),twap_trades(p),rtol=1e-11,atol=1e-7)


def test_zero_volatility_optimal_is_twap():
    p=ExecutionProblem(volatility=0)
    assert np.allclose(optimal_trades(p),twap_trades(p),atol=1e-7)


def test_trade_schedules_liquidate_without_shorting():
    p=ExecutionProblem()
    for n in schedules(p).values():
        assert len(n)==p.intervals and np.min(n)>=0
        assert np.isclose(np.sum(n),p.shares,atol=1e-7)
        assert np.all(np.diff(p.shares-n.cumsum())<=1e-7)


def test_more_risk_aversion_frontloads():
    low=optimal_trades(ExecutionProblem(risk_aversion=1e-6))
    high=optimal_trades(ExecutionProblem(risk_aversion=1e-4))
    assert high[:5].sum()>low[:5].sum()


def test_optimal_objective_vs_twap_and_vwap():
    p=ExecutionProblem()
    o=stats(optimal_trades(p),p)['objective_usd']
    assert o<=stats(twap_trades(p),p)['objective_usd']+1e-6
    assert o<=stats(vwap_trades(p),p)['objective_usd']+1e-6


def test_numeric_optimization_agrees():
    p=ExecutionProblem(shares=5000,intervals=8,risk_aversion=3e-5)
    x=optimal_trades(p)
    res=minimize(lambda n:stats(n,p)['objective_usd'],x,method='SLSQP',
                 constraints={'type':'eq','fun':lambda n:float(n.sum()-p.shares)},
                 bounds=[(0,p.shares)]*p.intervals)
    assert res.success
    assert np.isclose(stats(x,p)['objective_usd'],res.fun,atol=1e-5)


def test_expected_temporary_and_permanent_impact():
    p=ExecutionProblem(shares=1000,intervals=4,horizon_days=1,temporary_impact=.001,
                       permanent_impact=.002,risk_aversion=0)
    n=twap_trades(p)
    expected=.002*1000**2/2+.001/p.dt*np.dot(n,n)
    assert np.isclose(stats(n,p)['expected_cost_usd'],expected)


def test_zero_risk_variance():
    p=ExecutionProblem(volatility=0)
    assert stats(twap_trades(p),p)['cost_std_usd']==0
    assert np.std(monte_carlo_costs(twap_trades(p),p,paths=200))==0


def test_monte_carlo_matches_analytic_moments():
    p=ExecutionProblem(shares=5000,intervals=12,volatility=1.2)
    n=optimal_trades(p)
    analytical=stats(n,p)
    sim=monte_carlo_costs(n,p,paths=20000,seed=7)
    assert abs(sim.mean()-analytical['expected_cost_usd'])<.045*analytical['cost_std_usd']
    assert abs(sim.std(ddof=1)/analytical['cost_std_usd']-1)<.035


def test_simulation_chunk_invariance():
    p=ExecutionProblem()
    n=twap_trades(p)
    assert np.array_equal(monte_carlo_costs(n,p,paths=121,chunk=8),
                          monte_carlo_costs(n,p,paths=121,chunk=23))


def test_bad_params_and_schedule():
    for kwargs in [{'shares':0},{'intervals':1},{'horizon_days':-1},
                   {'risk_aversion':-1},{'volatility':float('nan')},
                   {'temporary_impact':0}]:
        with pytest.raises(ValueError):ExecutionProblem(**kwargs)
    p=ExecutionProblem()
    with pytest.raises(ValueError):stats(np.ones(p.intervals),p)
    with pytest.raises(ValueError):vwap_trades(p,weights=np.full(p.intervals,-1))


def test_zero_permanent_impact_only_removes_constant():
    p=ExecutionProblem()
    q=replace(p,permanent_impact=0)
    assert np.allclose(optimal_trades(p),optimal_trades(q))
    n=optimal_trades(p)
    assert np.isclose(stats(n,p)['expected_cost_usd']-stats(n,q)['expected_cost_usd'],
                      p.permanent_impact*p.shares**2/2)
