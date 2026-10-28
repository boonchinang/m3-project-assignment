# Results summary (TSLA)

- **Data:** Yahoo Finance (downloaded), 2013-01-03 to 2026-09-01, 3,436 modelling rows, 18 features
- **Target:** 20-day return > +6.0% = Buy, < -6.0% = Sell, else Hold
- **Selected model:** Logistic Regression (tuned), params {'clf__C': 0.1}, confidence threshold 0.46

| Test 2023-24 | ML strategy | Buy & Hold |
|---|---|---|
| CAGR | 33.2% | 38.6% |
| Sharpe | 0.99 | 0.82 |
| Max drawdown | -39.9% | -53.8% |
| Macro F1 | 30.1% | – |
| Trades | 68 | 1 |

- **Excess return 95% CI:** [-63.48%, +34.43%] per year
- **Recommendation:** DO NOT DEPLOY YET: promising, but the edge is not yet statistically reliable or cost-justified; run paper trading
