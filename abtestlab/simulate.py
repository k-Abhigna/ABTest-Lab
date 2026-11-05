"""Synthetic e-commerce checkout experiment with a known true effect.

Because we generate the data, we know the right answer, so the analysis can be
checked against it (something real experiment data can never give you).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_CONV = 0.10          # control checkout conversion
TRUE_LIFT_REL = 0.08      # treatment raises conversion by 8% relative (10.0% -> 10.8%)
REFUND_RATE = 0.05        # same in both groups: the redesign does no harm


def _calibrate(propensity: np.ndarray, target_conv: float) -> float:
    """Scale r so that mean(1 - exp(-r * propensity)) == target_conv (bisection)."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 - np.exp(-mid * propensity)) < target_conv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_experiment(n_days: int = 14, users_per_day: int = 3300,
                        lift_rel: float = TRUE_LIFT_REL, seed: int = 42,
                        srm_bug: bool = False) -> pd.DataFrame:
    """One row per user.

    pre_revenue : spend in the 12 weeks *before* the test (CUPED covariate)
    converted   : bought during the test
    revenue     : spend during the test
    refunded    : asked for a refund (guardrail)

    srm_bug=True drops 8% of treatment users with no purchase history, mimicking a redirect
    that fails on slow devices. The groups then differ in more than the treatment.
    """
    rng = np.random.default_rng(seed)
    n = n_days * users_per_day
    day = np.repeat(np.arange(1, n_days + 1), users_per_day)
    group = rng.choice(["control", "treatment"], size=n)

    # Each user has a latent purchase rate that drives both past and future orders,
    # so pre-period spend predicts test-period spend (what makes CUPED work).
    propensity = rng.lognormal(mean=0.0, sigma=1.2, size=n)
    basket = rng.lognormal(mean=np.log(45), sigma=0.35, size=n)      # typical order value
    rate = _calibrate(propensity, BASE_CONV)                         # P(>=1 order) = BASE_CONV
    rate_t = _calibrate(propensity, BASE_CONV * (1 + lift_rel))
    lam = np.where(group == "treatment", rate_t, rate) * propensity

    pre_orders = rng.poisson(6 * rate * propensity)                  # 12 weeks before = 6x the 2-week test
    pre_revenue = pre_orders * basket * rng.gamma(20, 1 / 20, n)
    orders = rng.poisson(lam)
    converted = orders > 0
    revenue = orders * basket * rng.gamma(20, 1 / 20, n)
    refunded = converted & (rng.random(n) < REFUND_RATE)

    df = pd.DataFrame({
        "user_id": np.arange(n), "day": day, "group": group,
        "pre_revenue": pre_revenue.round(2), "converted": converted.astype(int),
        "revenue": revenue.round(2), "refunded": refunded.astype(int),
    })
    if srm_bug:
        low_engaged = (df.group == "treatment") & (df.pre_revenue == 0)
        drop = df[low_engaged].sample(frac=0.08, random_state=seed).index
        df = df.drop(drop).reset_index(drop=True)
    return df
