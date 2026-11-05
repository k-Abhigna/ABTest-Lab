# Experiment readout

**Decision: SHIP**: Conversion +1.11 pp (+11.5%), 95% CI [+0.56, +1.66] pp; refund guardrail cleared.

## 1. Sample ratio check
Control 23,040 / treatment 23,160 (share 50.13%), chi-square p = 0.577 -> OK

## 2. Metrics

| Metric | Control | Treatment | Diff | 95% CI | Rel. lift | p-value |
|---|---|---|---|---|---|---|
| conversion | 9.701% | 10.812% | +1.111% | [+0.558, +1.664] | +11.46% | 0.0001 |
| revenue_per_user | 5.556 | 6.519 | +0.963 | [+0.550, +1.376] | +17.33% | 0.0000 |
| revenue_per_user_cuped | 5.678 | 6.397 | +0.719 | [+0.353, +1.085] | +12.67% | 0.0001 |

CUPED (theta = 0.120) removed 21.5% of revenue variance; CI width 0.826 -> 0.732.

## 3. Guardrail: refund rate
0.460% -> 0.531%; one-sided 95% upper bound on the increase +0.178 pp vs margin 0.5 pp -> PASSED