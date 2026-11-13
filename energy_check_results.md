# Energy check: frozen strategy on energy vs equities

Signal: Combined (MA 50/200 + 126d momentum), long/flat, 5 bps costs. Data from 2008-01-01 (or first available).

| Asset                         | From       | Vol (ann.)   |   B&H Sharpe | B&H Max DD   |   Strat Sharpe | Strat Max DD   | Time invested   |   Avg size w/ 10% vol target | B&H 2022   | Strat 2022   |
|:------------------------------|:-----------|:-------------|-------------:|:-------------|---------------:|:---------------|:----------------|-----------------------------:|:-----------|:-------------|
| US Equities (SPY)             | 2008-01-03 | 20%          |         0.63 | -52%         |           0.74 | -26%           | 81%             |                         0.88 | -18%       | -13%         |
| Gold (GLD)                    | 2008-01-03 | 18%          |         0.51 | -46%         |           0.57 | -29%           | 67%             |                         0.68 | -1%        | -8%          |
| Crude Oil ETF (USO)           | 2008-01-03 | 38%          |        -0.01 | -98%         |           0.14 | -67%           | 56%             |                         0.34 | 29%        | 24%          |
| Broad Commodities (DBC)       | 2008-01-03 | 19%          |         0.13 | -76%         |           0.29 | -41%           | 57%             |                         0.62 | 19%        | 17%          |
| Brent Crude futures (BZ=F)    | 2008-01-03 | 39%          |         0.19 | -87%         |           0.14 | -61%           | 52%             |                         0.35 | 10%        | 23%          |
| US Natural Gas futures (NG=F) | 2008-01-03 | 61%          |         0.23 | -89%         |           0.11 | -71%           | 52%             |                         0.18 | 20%        | 48%          |

Natural gas volatility: winter (Nov-Mar) 69% vs rest of year 54%.

Since 2008: Brent front-month series -3% vs USO ETF -79% (gap = mostly roll cost in contango, plus WTI vs Brent).
