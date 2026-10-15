# crypto-backtester
Backtester!

Crypto Strategy Backtester

A Python tool that replays a trading strategy over historical crypto prices and reports how it would have performed. It currently tests a simple moving-average (SMA) crossover: buy when the fast average rises above the slow average, and go flat when it falls below.

Educational project. Not financial advice. Backtest results do not guarantee future performance.

What it does
Pulls historical price data from Kraken (via ccxt), from GeckoTerminal (for DEX-traded tokens, experimental), or generates synthetic data for offline testing
Generates SMA crossover signals, shifted one bar so the strategy never sees future data (no look-ahead bias)
Models trading fees and slippage on every position change
Reports total return, CAGR, Sharpe ratio, max drawdown, win rate, and trade count
Plots the strategy's equity curve against a buy-and-hold benchmark (equity_curve.png)
Setup
bash
pip install -r requirements.txt
Run
bash
python3 crypto_backtest.py

By default this downloads BTC/USD daily candles from Kraken (no API key needed), runs a 20/50 SMA crossover, prints the metrics, and saves equity_curve.png.

Configuration

Edit the if __name__ == "__main__": block at the bottom of crypto_backtest.py:

Setting	What it does
DATA_SOURCE	"ccxt" (Kraken), "geckoterminal" (DEX tokens, experimental), or "synthetic"
fast, slow	SMA window lengths in candles
fee_bps, slippage_bps	Trading costs in basis points (100 bps = 1%)
initial_capital	Starting portfolio value

For GeckoTerminal, set network (e.g. "solana") and pool_address (the pool address from the token's page on geckoterminal.com, not the token contract address). Free-tier history is limited and rate-limited.

Limitations
One strategy, one asset, and one time period per run, so a good result here is not evidence of a real edge
Long-only, with simple flat-rate cost assumptions; thinly traded tokens can have much higher real slippage
The GeckoTerminal loader has not been tested against the live API
It does not place trades; it only simulates them
Ideas for next steps
Walk-forward testing across multiple time periods
Parameter sweeps over SMA window pairs
Additional strategies (RSI mean reversion, etc.)
A machine-learning signal generator
Tech

Python, pandas, NumPy, matplotlib, ccxt, requests