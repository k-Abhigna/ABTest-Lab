# ABTest-Lab: A/B Test Readouts You Can Trust

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

ABTest-Lab plans, analyses and validates an e-commerce checkout A/B test. The analysis runs in a fixed
order: sample-ratio check → primary metric → CUPED-adjusted secondary metric → guardrail → SHIP / DON'T SHIP /
NO DECISION / INVALID. The data is simulated with a **known true effect (+8% conversion)**, so every number the
toolkit produces can be checked against the truth with Monte Carlo runs.

| Result (400 simulated experiments each) | Expected | Observed |
|---|---|---|
| False-positive rate, A/A tests | 5% | **4.0%** |
| Power to detect the true +8% lift | 80% (planned) | **80.0%** |
| Mean estimated lift | +8.0% | **+7.93%** (unbiased) |
| Revenue/user power: raw → **CUPED** | CUPED higher | **78.2% → 85.0%** |
| SRM bug detected | high | **98.8%** |
| Lift you would report if the SRM bug were ignored | +8.0% | **+10.6%** (overstated by a third) |
| False positives when peeking daily → **with Pocock boundary** | 5% | **21.8% → 5.3%** |

## What it does

1. **Plan:** sample size for a proportion or a mean (`sample_size_proportion` gives 22,855 users/group for
   +8% on a 10% baseline, so 14 days at 3,300 users/day).
2. **Trust check:** chi-square sample-ratio-mismatch (SRM) test. If it fails, the readout stops with
   `INVALID` and reports no metrics.
3. **Primary metric:** two-proportion z-test on conversion (pooled SE for the p-value, unpooled for the CI).
4. **Secondary metric:** Welch t-test on revenue per user, raw and **CUPED**-adjusted on 12-week
   pre-experiment spend. Percentile bootstrap CI for relative lift.
5. **Guardrail:** one-sided non-inferiority test: the refund rate must not rise by more than 0.5 pp.
6. **Peeking:** shows how checking p-values daily inflates false positives, and calibrates a Pocock
   boundary by Monte Carlo to fix it.

All formulas are plain NumPy/SciPy in [`abtestlab/stats.py`](abtestlab/stats.py) so they can be read and checked.

## Sample readout (`reports/readout_checkout_test.md`)

**Decision: SHIP**: Conversion +1.11 pp (+11.5%), 95% CI [+0.56, +1.66] pp; refund guardrail cleared.

| Metric | Control | Treatment | Diff | 95% CI | p-value |
|---|---|---|---|---|---|
| conversion | 9.70% | 10.81% | +1.11 pp | [+0.56, +1.66] | 0.0001 |
| revenue/user | 5.556 | 6.519 | +0.963 | [+0.550, +1.376] | <0.0001 |
| revenue/user (CUPED) | 5.678 | 6.397 | +0.719 | [+0.353, +1.085] | 0.0001 |

CUPED removed 21.5% of revenue variance (CI width 0.826 → 0.732). This one run's +11.5% is above the true
+8% but the CI contains it; the Monte Carlo table above shows the estimate is unbiased on average.

With the SRM bug (8% of new treatment users lost by a failing redirect) the readout is:
**INVALID**: treatment has 48.84% of users vs 50% expected (p = 8.7e-07).

## What the validation shows

- **An SRM is not a small problem.** Losing low-engagement users from one arm makes the treatment look
  10.6% better instead of 8%. The SRM check catches it in 98.8% of runs, which is why it runs first.
- **CUPED is free power.** Same unbiased estimate (0.550 vs 0.549 mean difference), lower spread
  (SD 0.192 → 0.174), and 7 more points of power without more traffic.
- **Peeking breaks the 5% promise.** Stopping at the first daily p < 0.05 gives 21.8% false positives over
  14 looks. A constant boundary of |z| > 2.62 restores 5.3%.

## Tests (19)

Formulas are checked against independent references: z-test vs SciPy chi-square, Welch vs
`scipy.stats.ttest_ind`, sample size vs the textbook value (3,841/group for 10% → 12%), Pocock boundaries vs the
published table (2.178 / 2.413 / 2.555 for 2 / 5 / 10 looks). Plus decision-logic tests (SRM blocks the decision,
guardrail failure blocks ship, no effect gives NO DECISION, missing columns raise) and a Streamlit smoke test.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate       # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python -m pytest                                     # 19 tests
python -m abtestlab.run                              # simulated test -> reports/readout_checkout_test.md
python -m abtestlab.run --srm-bug                    # broken split -> INVALID
python -m abtestlab.run --csv my_test.csv            # your own data
python -m abtestlab.validate --sims 400              # Monte Carlo validation (~a few minutes)
streamlit run app.py                                 # Plan / Analyse / Why-not-peek dashboard
```

CSV columns: `group` (control/treatment), `converted`, `revenue`, `pre_revenue`, `refunded`.

## Limitations

- The data is synthetic. That is what makes the validation possible, but real traffic has novelty effects,
  weekly seasonality and network effects that this simulator does not model.
- Users are assumed independent. Tests that randomise by session or by cluster need a delta-method or
  clustered SE.
- One primary metric, so no multiple-testing correction is applied. Testing many metrics would need
  Holm or Benjamini-Hochberg.

## Project layout

```
abtestlab/stats.py      z-test, Welch, CUPED, bootstrap, SRM, non-inferiority, sample size, Pocock
abtestlab/simulate.py   checkout experiment with known effect, optional SRM bug
abtestlab/analyze.py    readout pipeline and decision rules
abtestlab/validate.py   Monte Carlo checks -> reports/validation.{md,json}
abtestlab/run.py        CLI
app.py                  Streamlit dashboard
```
