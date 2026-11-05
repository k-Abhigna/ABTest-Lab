"""Statistical tests for A/B experiments.

Everything here is plain NumPy/SciPy so each formula can be read and checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    diff: float           # treatment - control (absolute)
    ci_low: float         # CI for the absolute difference
    ci_high: float
    rel_lift: float       # diff / control
    p_value: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


# ---------------------------------------------------------------- proportions
def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int,
                         alpha: float = 0.05, metric: str = "conversion") -> TestResult:
    """Two-sided z-test. Pooled SE for the p-value, unpooled SE for the CI."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    d = p_t - p_c
    return TestResult(metric, p_c, p_t, d, d - zc * se, d + zc * se, d / p_c, float(p_value))


# ---------------------------------------------------------------- means
def welch_ttest(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05,
                metric: str = "revenue_per_user") -> TestResult:
    """Welch's t-test (unequal variances) with a matching Welch CI."""
    m_c, m_t = y_c.mean(), y_t.mean()
    v_c, v_t = y_c.var(ddof=1) / len(y_c), y_t.var(ddof=1) / len(y_t)
    se = np.sqrt(v_c + v_t)
    df = (v_c + v_t) ** 2 / (v_c ** 2 / (len(y_c) - 1) + v_t ** 2 / (len(y_t) - 1))
    t = (m_t - m_c) / se
    p_value = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(1 - alpha / 2, df)
    d = m_t - m_c
    return TestResult(metric, m_c, m_t, d, d - tc * se, d + tc * se, d / m_c, float(p_value))


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: y_adj = y - theta * (x - mean(x)), theta = cov(y, x) / var(x).

    x must be a pre-experiment covariate (unaffected by treatment), and theta is
    estimated on both groups pooled, so the adjustment cannot create a bias.
    """
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def bootstrap_rel_lift_ci(y_c: np.ndarray, y_t: np.ndarray, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(y_t) / mean(y_c) - 1."""
    rng = np.random.default_rng(seed)
    mc = np.array([rng.choice(y_c, len(y_c)).mean() for _ in range(n_boot)])
    mt = np.array([rng.choice(y_t, len(y_t)).mean() for _ in range(n_boot)])
    lift = mt / mc - 1
    return float(np.quantile(lift, alpha / 2)), float(np.quantile(lift, 1 - alpha / 2))


# ---------------------------------------------------------------- guardrails
def srm_check(n_c: int, n_t: int, expected_share_t: float = 0.5,
              threshold: float = 0.001) -> dict:
    """Sample-ratio mismatch: chi-square goodness of fit on the group sizes."""
    n = n_c + n_t
    expected = [n * (1 - expected_share_t), n * expected_share_t]
    chi2, p = stats.chisquare([n_c, n_t], expected)
    return {"n_control": n_c, "n_treatment": n_t, "share_treatment": n_t / n,
            "chi2": float(chi2), "p_value": float(p), "srm": bool(p < threshold)}


def noninferiority_test(x_c: int, n_c: int, x_t: int, n_t: int, margin: float,
                        alpha: float = 0.05) -> dict:
    """One-sided test that a 'bad' rate (e.g. refunds) did not rise by more than `margin`.

    H0: p_t - p_c >= margin   vs   H1: p_t - p_c < margin.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = (p_t - p_c - margin) / se
    p = float(stats.norm.cdf(z))
    upper = p_t - p_c + stats.norm.ppf(1 - alpha) * se
    return {"control": p_c, "treatment": p_t, "diff": p_t - p_c, "upper_bound": float(upper),
            "margin": margin, "p_value": p, "passed": bool(p < alpha)}


# ---------------------------------------------------------------- planning
def sample_size_proportion(p_base: float, mde_rel: float, alpha: float = 0.05,
                           power: float = 0.8) -> int:
    """Users per group to detect a relative lift `mde_rel` on baseline rate `p_base`."""
    p1, p2 = p_base, p_base * (1 + mde_rel)
    pbar = (p1 + p2) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def sample_size_mean(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb) ** 2 * sd ** 2 / mde_abs ** 2))


def pocock_boundary(n_looks: int, alpha: float = 0.05, n_sim: int = 200_000,
                    seed: int = 0) -> float:
    """Constant |z| boundary for `n_looks` equally spaced interim looks (Pocock).

    Calibrated by Monte Carlo: under H0 the z-statistics at the looks are a scaled
    random walk, so we find the (1 - alpha) quantile of max_k |Z_k|.
    """
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_sim, n_looks))
    z = steps.cumsum(axis=1) / np.sqrt(np.arange(1, n_looks + 1))
    return float(np.quantile(np.abs(z).max(axis=1), 1 - alpha))
