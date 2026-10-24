"""Backward-compatible alias; sim download is 1m now. """

from tradingai.sim.jobs.download_alpaca_1m import download_alpaca_1m as download_alpaca_5m
from tradingai.sim.jobs.download_alpaca_1m import main

__all__ = ["download_alpaca_5m", "main"]
