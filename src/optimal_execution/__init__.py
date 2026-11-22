"""Quant research toolkit for deterministic optimal execution."""
from .model import ExecutionProblem, optimal_trades, twap_trades, vwap_trades, stats, monte_carlo_costs, schedules
__all__=['ExecutionProblem','optimal_trades','twap_trades','vwap_trades','stats','monte_carlo_costs','schedules']
