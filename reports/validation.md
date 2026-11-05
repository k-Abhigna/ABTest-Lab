# Validation (Monte Carlo)

Planned: 22,855 users/group to detect +8% on a 10% baseline (alpha 0.05, power 0.80) -> 14 days at 3,300 users/day.

## Whole-experiment simulations (400 runs each)

| Check | Expected | Observed |
|---|---|---|
| A/A false-positive rate | 5% | 4.0% |
| A/A SRM false alarms | 0.1% | 0.0% |
| Power, conversion (true +8%) | 80% | 80.0% |
| Mean estimated lift | +8.0% | +7.93% |
| Power, revenue/user: raw -> CUPED | CUPED higher | 78.2% -> 85.0% |
| Revenue diff, mean raw vs CUPED (no bias) | equal | 0.550 vs 0.549 |
| Revenue diff, SD raw vs CUPED | CUPED lower | 0.192 vs 0.174 |
| CUPED variance reduction | rho^2 | 18.6% |
| SRM bug detected | high | 98.8% |
| Lift reported when SRM bug is ignored | +8.0% (truth) | +10.58% |

## Peeking (20,000 A/A tests, 14 daily looks)

| Strategy | False-positive rate |
|---|---|
| Look once at the end | 5.1% |
| Look daily, stop at first p < 0.05 | 21.8% |
| Look daily with Pocock boundary \|z\| > 2.62 | 5.3% |