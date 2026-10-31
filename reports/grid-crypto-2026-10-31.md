# Grid trading results (5-minute bars, snapshot 2026-10-31)

Spot grids, 20 geometric levels, 0.10% fee per side, fills only on a 0.05% trade-through. Benchmark: 50/50 coin/cash rebalanced monthly. Vault (last 18 months) untouched.

## BTC/USD (2017-01-01 – 2025-04-29)

```
grid-btc-v1 [crypto] -> FAIL  (selected: grid_2.5pct_breakout, N=12)
  FAIL  wf_excess_ann                   -0.1174   > 0 vs benchmark, after costs
  FAIL  wf_excess_ann_2x_cost           -0.1355   > 0 with costs doubled
  FAIL  wf_max_dd                       -0.5062   >= benchmark -0.4855
  FAIL  cv_excess_ann                   -0.1635   > 0 vs benchmark
  FAIL  deflated_sharpe_active           0.0000   > 0.95 (SR* 1.78 ann.)
  PASS  pbo                              0.0000   < 0.2
```

| Version | CAGR | Sharpe | Max DD | vs 50/50 /yr | Fills /yr | Fee drag /yr | Time fully in coin | Time fully in cash | Re-centres |
|---|---|---|---|---|---|---|---|---|---|
| grid_1pct_monthly | -8.0% | 0.03 | -85.1% | -39.8% | 1,734 | 9.0% | 15.6% | 23.0% | 117 |
| grid_1pct_breakout | -27.6% | -0.61 | -96.7% | -65.1% | 4,370 | 23.7% | 0.5% | 1.0% | 425 |
| grid_2.5pct_monthly | 18.8% | 0.64 | -60.7% | -16.0% | 574 | 3.1% | 3.2% | 7.6% | 117 |
| grid_2.5pct_breakout | 25.7% | 0.83 | -62.6% | -11.9% | 841 | 4.4% | 0.2% | 1.2% | 70 |
| 50/50 rebalanced | 41.3% | 1.14 | -56.9% | | | | | | |
| buy and hold | 72.7% | 1.12 | -83.8% | | | | | | |

Calendar-year returns:

|   date | grid_1pct_monthly   | grid_1pct_breakout   | grid_2.5pct_monthly   | grid_2.5pct_breakout   | 50/50   | hold    |
|-------:|:--------------------|:---------------------|:----------------------|:-----------------------|:--------|:--------|
|   2017 | 80.2%               | -23.9%               | 168.0%                | 168.2%                 | 336.1%  | 1296.2% |
|   2018 | -66.5%              | -79.1%               | -51.5%                | -48.4%                 | -43.1%  | -73.4%  |
|   2019 | 8.1%                | -22.7%               | 32.2%                 | 42.1%                  | 47.2%   | 94.1%   |
|   2020 | 12.8%               | 14.9%                | 48.7%                 | 72.0%                  | 118.7%  | 304.6%  |
|   2021 | -8.8%               | -29.6%               | 31.1%                 | 27.3%                  | 36.0%   | 59.4%   |
|   2022 | -53.5%              | -54.3%               | -41.9%                | -38.6%                 | -36.9%  | -64.2%  |
|   2023 | 24.5%               | 36.5%                | 46.7%                 | 52.3%                  | 65.6%   | 155.8%  |
|   2024 | 26.1%               | 16.0%                | 41.0%                 | 53.2%                  | 55.6%   | 120.8%  |
|   2025 | 1.5%                | -5.8%                | 4.1%                  | 9.0%                   | 1.0%    | 1.0%    |

## ETH/USD (2017-01-01 – 2025-04-29)

```
grid-eth-v1 [crypto] -> FAIL  (selected: grid_2.5pct_monthly, N=12)
  FAIL  wf_excess_ann                   -0.1718   > 0 vs benchmark, after costs
  FAIL  wf_excess_ann_2x_cost           -0.2341   > 0 with costs doubled
  FAIL  wf_max_dd                       -0.6586   >= benchmark -0.5017
  FAIL  cv_excess_ann                   -0.4447   > 0 vs benchmark
  FAIL  deflated_sharpe_active           0.0000   > 0.95 (SR* 1.78 ann.)
  PASS  pbo                              0.0010   < 0.2
```

| Version | CAGR | Sharpe | Max DD | vs 50/50 /yr | Fills /yr | Fee drag /yr | Time fully in coin | Time fully in cash | Re-centres |
|---|---|---|---|---|---|---|---|---|---|
| grid_1pct_monthly | -31.0% | -0.37 | -97.7% | -78.2% | 2,116 | 10.9% | 22.6% | 28.0% | 117 |
| grid_1pct_breakout | -49.6% | -1.03 | -99.9% | -111.0% | 7,294 | 39.6% | 0.5% | 0.9% | 714 |
| grid_2.5pct_monthly | -0.3% | 0.26 | -85.6% | -44.0% | 794 | 4.3% | 8.2% | 13.3% | 117 |
| grid_2.5pct_breakout | 20.3% | 0.64 | -80.3% | -27.8% | 1,422 | 7.5% | 0.1% | 0.8% | 130 |
| 50/50 rebalanced | 57.3% | 1.17 | -70.7% | | | | | | |
| buy and hold | 91.0% | 1.15 | -94.0% | | | | | | |

Calendar-year returns:

|   date | grid_1pct_monthly   | grid_1pct_breakout   | grid_2.5pct_monthly   | grid_2.5pct_breakout   | 50/50   | hold    |
|-------:|:--------------------|:---------------------|:----------------------|:-----------------------|:--------|:--------|
|   2017 | 28.0%               | -26.2%               | 132.6%                | 363.8%                 | 1169.6% | 8962.0% |
|   2018 | -87.3%              | -93.2%               | -76.4%                | -66.1%                 | -48.1%  | -82.4%  |
|   2019 | -28.2%              | -43.1%               | -2.5%                 | -5.0%                  | 7.6%    | -1.5%   |
|   2020 | 1.3%                | 2.8%                 | 51.0%                 | 72.0%                  | 178.0%  | 473.9%  |
|   2021 | 38.9%               | -33.4%               | 132.8%                | 120.7%                 | 153.6%  | 398.3%  |
|   2022 | -67.3%              | -69.9%               | -56.6%                | -43.1%                 | -35.9%  | -67.5%  |
|   2023 | 37.5%               | 9.6%                 | 44.3%                 | 41.6%                  | 40.9%   | 90.8%   |
|   2024 | -10.0%              | -17.4%               | 9.4%                  | 39.7%                  | 28.8%   | 46.0%   |
|   2025 | -31.1%              | -37.2%               | -24.2%                | -26.8%                 | -25.2%  | -46.0%  |

## Resolution check: same grids on coarser bars (research period)

| Coin | Variant | 5m CAGR | 1h CAGR | 1d CAGR | 5m fills /yr | 1h fills /yr | 1d fills /yr |
|---|---|---|---|---|---|---|---|
| BTC/USD | grid_1pct_monthly | -8.0% | -11.6% | -25.4% | 1,734 | 1,557 | 770 |
| BTC/USD | grid_1pct_breakout | -27.6% | -24.0% | -41.5% | 4,370 | 3,457 | 1,253 |
| BTC/USD | grid_2.5pct_monthly | 18.8% | 17.0% | 6.2% | 574 | 551 | 396 |
| BTC/USD | grid_2.5pct_breakout | 25.7% | 19.4% | 3.3% | 841 | 761 | 455 |
| ETH/USD | grid_1pct_monthly | -31.0% | -35.1% | -46.9% | 2,116 | 1,831 | 809 |
| ETH/USD | grid_1pct_breakout | -49.6% | -51.6% | -63.9% | 7,294 | 5,296 | 1,640 |
| ETH/USD | grid_2.5pct_monthly | -0.3% | -3.3% | -17.2% | 794 | 744 | 482 |
| ETH/USD | grid_2.5pct_breakout | 20.3% | 23.6% | -15.2% | 1,422 | 1,236 | 650 |
