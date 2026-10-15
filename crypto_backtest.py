"""
Crypto Moving-Average Crossover Backtester
-------------------------------------------
A minimal backtesting script for a single-pair crypto strategy.

Strategy: go long when the fast SMA crosses above the slow SMA,
go flat when it crosses below.

This script is a STARTING POINT, not a finished trading system.
It includes:
  - Data loading (Kraken via ccxt, GeckoTerminal for DEX tokens, or synthetic data)
  - Transaction cost + slippage modeling
  - Vectorized backtest (no look-ahead bias)
  - Performance metrics: total return, CAGR, Sharpe, max drawdown, win rate
  - Equity curve plot vs buy-and-hold benchmark

Educational project. Not financial advice.

Usage:
    python3 crypto_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data_ccxt(symbol="BTC/USD", timeframe="1d", exchange_id="kraken", limit=1000):
    """
    Fetch real historical OHLCV data using ccxt (public data, no API key needed).
    Requires: pip install ccxt
    Note: Binance blocks some regions (HTTP 451), so Kraken is the default.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def load_data_geckoterminal(network, pool_address, timeframe="day", aggregate=1, limit=1000):
    """
    EXPERIMENTAL / untested against the live API.

    Fetch OHLCV data for a DEX-traded token (e.g. a memecoin) from the free
    GeckoTerminal public API. No API key required.

    network:       chain id, e.g. "solana", "eth", "base", "bsc"
    pool_address:  the LIQUIDITY POOL address (not the token contract address!)
                   Find it by searching the token on geckoterminal.com,
                   opening its main pool, and copying the address from the URL:
                   https://www.geckoterminal.com/<network>/pools/<POOL_ADDRESS>
    timeframe:     "day", "hour", or "minute"
    aggregate:     candle size within the timeframe (e.g. hour + aggregate=4 = 4h candles)
                   Allowed: day->[1], hour->[1,4,12], minute->[1,5,15]
    limit:         max candles to return (max 1000 per call; free tier is
                   rate-limited and history is capped at roughly 6 months)
    """
    import requests

    url = (f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/"
           f"{pool_address}/ohlcv/{timeframe}")
    params = {"aggregate": aggregate, "limit": limit, "currency": "usd"}

    resp = requests.get(url, params=params, headers={"Accept": "application/json"})
    resp.raise_for_status()
    payload = resp.json()

    candles = payload["data"]["attributes"]["ohlcv_list"]
    # Each row: [timestamp(sec), open, high, low, close, volume]
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)  # GeckoTerminal returns newest-first
    return df


def load_data_synthetic(n_days=730, start_price=30000, seed=42):
    """
    Generate a synthetic random-walk price series so the script is runnable
    offline. Results on this data are NOT meaningful; it only exercises the pipeline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")

    # Random walk with slight drift + volatility clustering
    daily_returns = rng.normal(loc=0.0006, scale=0.025, size=n_days)
    vol_regime = np.abs(rng.normal(1.0, 0.4, size=n_days)).clip(0.3, 3.0)
    daily_returns *= vol_regime

    price = start_price * np.exp(np.cumsum(daily_returns))
    close = pd.Series(price, index=dates)

    df = pd.DataFrame({
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
    })
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.01, n_days))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.01, n_days))
    df["volume"] = rng.uniform(1000, 5000, n_days)
    return df


# ---------------------------------------------------------------------------
# 2. STRATEGY: SMA CROSSOVER
# ---------------------------------------------------------------------------

def sma_crossover_signal(df, fast=20, slow=50):
    """
    Returns a signal series: 1 = long, 0 = flat.
    Signal is shifted by 1 bar so today's decision only uses data known
    at the close of the previous bar (avoids look-ahead bias).
    """
    fast_sma = df["close"].rolling(fast).mean()
    slow_sma = df["close"].rolling(slow).mean()
    raw_signal = (fast_sma > slow_sma).astype(int)
    signal = raw_signal.shift(1).fillna(0)
    return signal


# ---------------------------------------------------------------------------
# 3. BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(df, signal, fee_bps=10, slippage_bps=5, initial_capital=10_000):
    """
    Vectorized long-only backtest.
    fee_bps: taker fee in basis points per trade (e.g. 10 = 0.10%)
    slippage_bps: assumed slippage in basis points per trade
    """
    df = df.copy()
    df["signal"] = signal
    df["returns"] = df["close"].pct_change().fillna(0)

    # Strategy return = position held during the bar * that bar's return
    df["strategy_returns"] = df["signal"] * df["returns"]

    # Apply costs only when the position actually changes (a trade occurs)
    trades = df["signal"].diff().abs().fillna(0)
    cost_per_trade = (fee_bps + slippage_bps) / 10_000
    df["costs"] = trades * cost_per_trade
    df["strategy_returns_net"] = df["strategy_returns"] - df["costs"]

    df["equity"] = initial_capital * (1 + df["strategy_returns_net"]).cumprod()
    df["buy_hold_equity"] = initial_capital * (1 + df["returns"]).cumprod()

    return df


# ---------------------------------------------------------------------------
# 4. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df, periods_per_year=365):
    rets = df["strategy_returns_net"]
    equity = df["equity"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(df)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n_periods) - 1

    sharpe = 0.0
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    winning = (rets > 0).sum()
    losing = (rets < 0).sum()
    win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0

    n_trades = int(df["signal"].diff().abs().sum())

    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Win Rate (per candle)": f"{win_rate:.2%}",
        "Number of Trades": n_trades,
    }


# ---------------------------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------------------------

def plot_equity_curve(df, out_path="equity_curve.png"):
    plt.figure(figsize=(11, 6))
    plt.plot(df.index, df["equity"], label="Strategy (SMA Crossover)", linewidth=1.8)
    plt.plot(df.index, df["buy_hold_equity"], label="Buy & Hold", linewidth=1.2, alpha=0.7)
    plt.title("Strategy vs Buy & Hold Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved equity curve chart to {out_path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Choose your data source: "ccxt", "geckoterminal", or "synthetic"
    DATA_SOURCE = "ccxt"

    if DATA_SOURCE == "ccxt":
        data = load_data_ccxt(symbol="BTC/USD", timeframe="1d", limit=1000, exchange_id="kraken")

    elif DATA_SOURCE == "geckoterminal":
        # EXPERIMENTAL: this data source is untested. Replace the placeholder
        # below with a real pool address from geckoterminal.com (the part of the
        # URL after /pools/).
        data = load_data_geckoterminal(
            network="solana",
            pool_address="YOUR_POOL_ADDRESS_HERE",
            timeframe="hour",
            aggregate=4,     # 4-hour candles; memecoins move fast, daily may be too slow
            limit=500,
        )
        # NOTE: with hourly/4h candles, the SMA(20/50) below means
        # "20 candles" and "50 candles", not 20/50 days. Adjust to taste.

    else:
        print("Using synthetic random-walk data (results are NOT meaningful).")
        print("Set DATA_SOURCE to 'ccxt' or 'geckoterminal' to pull real prices.\n")
        data = load_data_synthetic(n_days=730)

    signal = sma_crossover_signal(data, fast=20, slow=50)
    results = run_backtest(data, signal, fee_bps=10, slippage_bps=5, initial_capital=10_000)
    metrics = compute_metrics(results)

    print("=== Backtest Results: SMA(20/50) Crossover ===")
    for k, v in metrics.items():
        print(f"{k:>22}: {v}")

    plot_equity_curve(results, out_path="equity_curve.png")
