"""Generate reproducible execution schedules, simulations and plots."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import ExecutionProblem,schedules,stats,monte_carlo_costs


def main(argv=None):
    p=argparse.ArgumentParser(description='Almgren-Chriss optimal execution experiment')
    p.add_argument('--shares',type=float,default=100_000)
    p.add_argument('--intervals',type=int,default=20)
    p.add_argument('--days',type=float,default=1.0)
    p.add_argument('--volatility',type=float,default=1.5)
    p.add_argument('--temporary-impact',type=float,default=1e-6)
    p.add_argument('--permanent-impact',type=float,default=2e-7)
    p.add_argument('--risk-aversion',type=float,default=3e-5)
    p.add_argument('--paths',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='results')
    a=p.parse_args(argv)
    problem=ExecutionProblem(shares=a.shares,intervals=a.intervals,horizon_days=a.days,
        volatility=a.volatility,temporary_impact=a.temporary_impact,
        permanent_impact=a.permanent_impact,risk_aversion=a.risk_aversion)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    trade_table={'interval':np.arange(1,problem.intervals+1),'time_days':np.arange(1,problem.intervals+1)*problem.dt}
    report={}
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for name, trades in schedules(problem).items():
        trade_table[name]=trades
        measure=stats(trades,problem)
        draws=monte_carlo_costs(trades,problem,paths=a.paths,seed=a.seed)
        measure.update({'mc_mean_usd':float(draws.mean()),'mc_std_usd':float(draws.std(ddof=1)) if len(draws)>1 else 0.0,
            'mc_p95_usd':float(np.quantile(draws,.95))})
        report[name]=measure
        t=np.arange(problem.intervals+1)*problem.dt
        inventory=np.r_[problem.shares,problem.shares-trades.cumsum()]
        axs[0].plot(t,inventory,label=name)
        axs[1].plot(np.arange(1,problem.intervals+1),trades,label=name)
    pd.DataFrame(trade_table).to_csv(out/'schedules.csv',index=False)
    axs[0].set(title='Inventory liquidation',xlabel='Trading days',ylabel='Shares remaining')
    axs[1].set(title='Executed shares by interval',xlabel='Interval',ylabel='Shares sold')
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'execution_schedules.png',dpi=160);plt.close(fig)
    summary={'parameters':vars(a),'units':'USD, shares, trading days','schedules':report,
        'disclaimer':'Synthetic arithmetic Brownian price model; impact coefficients and ex-ante VWAP curve are assumed, not fitted to real executions.'}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':main()
