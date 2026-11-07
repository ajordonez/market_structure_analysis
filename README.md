# Market Structure Analysis

Two empirical studies of daily market data:

1. **Sector factor structure:** PCA on ten years of daily excess returns for ~100 large-cap US stocks, to find what drives co-movement and whether GICS sectors show up as real clusters.
2. **BTC vs ETH predictability:** tests whether a week of lagged returns can predict which of Bitcoin or Ethereum outperforms the next day, using four classifiers compared against a naive baseline with walk-forward validation.

The full analysis, code, and figures are in [`Stock_Sector_And_Crypto_Analysis.ipynb`](Stock_Sector_And_Crypto_Analysis.ipynb).

## Key findings

### Part 1: Sector factors (98 stocks, May 2016 – Apr 2026)

| Component | Variance explained | Interpretation |
|---|---|---|
| PC1 | 34.8% | **Market factor.** Every sector loads positively. |
| PC2 | 8.3% | **Defensive vs cyclical.** Utilities, staples and real estate on one side; energy, tech and financials on the other. |
| PC3 | 5.6% | **Growth vs commodities.** Information technology vs energy. |

- The component time series line up with known market events: the COVID crash (March 2020), the 2022 rate-hiking cycle, the November 2020 vaccine rotation (the largest PC3 move), and the April 2025 tariff shock.
- **Sectors are clearly visible in co-movement.** About two-thirds of each stock's 5 nearest neighbours (in 10-component PCA space) belong to its own sector, compared with about 8% by chance. t-SNE and UMAP keep this structure (purity about 0.67–0.70) when compressing it to 2-D.
- Sector labels are imperfect, though. Tech and communication services behave as one cluster, and consumer discretionary splits between tech-like names (AMZN, TSLA) and traditional retailers.

### Part 2: BTC vs ETH next-day outperformance (Nov 2017 – Apr 2026)

Results from a chronological 60/20/20 split. The baseline always predicts the majority class of the training set ("BTC outperforms").

| Model | Train error | Val error | Test error | p (beats baseline) |
|---|---|---|---|---|
| Naive baseline | 0.475 | 0.444 | **0.447** | – |
| Logistic regression (elastic net) | 0.468 | 0.439 | 0.463 | 0.80 |
| Random forest | 0.415 | 0.459 | 0.464 | 0.82 |
| XGBoost | 0.325 | 0.465 | 0.489 | 0.98 |
| Neural network (MLP) | 0.422 | 0.477 | 0.463 | 0.80 |

**No model beats the naive baseline.** A 5-fold walk-forward evaluation (2,570 out-of-sample days from 2019 to 2026) confirms this: the baseline has the lowest pooled error (0.464), and the more flexible models (XGBoost, the neural network) mostly add overfitting. Short-horizon relative returns between BTC and ETH are not predictable from their own recent history, which fits weak-form market efficiency.

## Methodology

**Part 1**
- Daily log returns minus the 13-week T-bill rate (`^IRX`) give excess returns.
- Each stock's return series is standardized before the SVD, so this is PCA on the correlation matrix and high-volatility stocks don't dominate.
- Missing prices are never back-filled. Tickers with no data or a late listing date are dropped and reported.
- t-SNE and UMAP run on the first 10 principal components. Sector clustering is measured with 5-NN sector purity instead of being judged by eye.

**Part 2**
- 14 features: BTC and ETH log returns for the previous 7 days. The target is whether BTC's return beats ETH's that day.
- Chronological split with no shuffling. Hyperparameters are tuned on validation only, and the test set is used once.
- Logistic regression and the neural network use a `StandardScaler` fitted on training data only.
- Significance: one-sided binomial tests against the baseline's accuracy, plus 95% confidence intervals.
- Robustness: `TimeSeriesSplit` walk-forward validation, with all tuning re-done inside each fold's training window.

## Limitations

- **Survivorship bias.** The stock universe is today's large caps, chosen in hindsight. Two tickers on the original list (EA, AVB) are no longer available from Yahoo Finance and are excluded.
- **Data drift.** Yahoo Finance revises adjusted prices and removes delisted tickers, so re-running the notebook later can change the results slightly.
- **Static factors.** The PCA covers the full decade at once. A rolling-window version would show how the factor structure changes over time.
- **Narrow features.** Part 2 only tests lagged returns. It does not use volume, volatility, funding rates or on-chain data.

