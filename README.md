# NIFTY 500 Momentum Strategy

A momentum-based monthly rebalancing strategy for the NIFTY 500 universe, using K-Means clustering, Fama-French factor betas, and Efficient Frontier portfolio optimization.

## Results

| Metric | Strategy | NIFTY 500 Benchmark |
|---|---|---|
| Cumulative Return (2022–2026) | ~165% | ~65% |

## How It Works

1. **Universe** — All NIFTY 500 stocks scraped from Wikipedia, downloaded via `yfinance`
2. **Features** — Garman-Klass volatility, RSI, Bollinger Bands, ATR, MACD, dollar volume
3. **Factor Betas** — Rolling OLS regression against Fama-French factors (SMB, HML, WML, MF)
4. **Clustering** — K-Means (k=4) on standardized features; selects the cluster with highest mean RSI (momentum proxy)
5. **Optimization** — Max Sharpe via `PyPortfolioOpt` with Ledoit-Wolf covariance shrinkage, L2 regularization, and 10% weight cap per stock
6. **Rebalancing** — Monthly

## Stack

- `yfinance`, `pandas`, `numpy`
- `pandas_ta` for technical indicators
- `statsmodels` for rolling OLS
- `scikit-learn` for K-Means clustering
- `PyPortfolioOpt` for portfolio optimization

## Setup

```bash
pip install yfinance pandas numpy pandas_ta statsmodels scikit-learn pypfopt pyarrow
```

Place `FFdata.csv` (Fama-French India factors) in the root directory, then:

```bash
python strategy.py
```

Parquet caches (`nifty500data.parquet`, `pricedatadaily.parquet`) are auto-generated on first run.

## Notes

- Past performance does not guarantee future results
- Strategy has not been live/paper traded yet