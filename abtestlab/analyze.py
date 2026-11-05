"""Full experiment readout: SRM -> primary -> secondary (CUPED) -> guardrail -> decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as st

REQUIRED = {"group", "converted", "revenue", "pre_revenue", "refunded"}


def analyze(df: pd.DataFrame, alpha: float = 0.05, refund_margin: float = 0.005,
            expected_share_t: float = 0.5, stop_on_srm: bool = True) -> dict:
    """stop_on_srm=False still computes metrics after an SRM (used only by simulations)."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    c, t = df[df.group == "control"], df[df.group == "treatment"]
    out: dict = {"n_users": len(df)}

    # 1. Trust check first: if the split is broken, no metric can be trusted.
    out["srm"] = st.srm_check(len(c), len(t), expected_share_t)
    if out["srm"]["srm"] and stop_on_srm:
        out["decision"] = "INVALID"
        out["reason"] = (f"Sample ratio mismatch (p = {out['srm']['p_value']:.1e}): treatment has "
                         f"{out['srm']['share_treatment']:.2%} of users vs {expected_share_t:.0%} "
                         "expected. Find the assignment/logging bug before reading any metric.")
        return out

    # 2. Primary metric: conversion.
    conv = st.two_proportion_ztest(c.converted.sum(), len(c), t.converted.sum(), len(t), alpha)
    out["conversion"] = conv

    # 3. Secondary: revenue per user, raw and CUPED-adjusted on pre-period spend.
    raw = st.welch_ttest(c.revenue.to_numpy(), t.revenue.to_numpy(), alpha)
    y_adj, theta = st.cuped_adjust(df.revenue.to_numpy(), df.pre_revenue.to_numpy())
    is_t = (df.group == "treatment").to_numpy()
    cuped = st.welch_ttest(y_adj[~is_t], y_adj[is_t], alpha, metric="revenue_per_user_cuped")
    var_red = 1 - y_adj.var() / df.revenue.to_numpy().var()
    out["revenue_raw"], out["revenue_cuped"] = raw, cuped
    out["cuped"] = {"theta": theta, "variance_reduction": float(var_red),
                    "ci_width_raw": raw.ci_high - raw.ci_low,
                    "ci_width_cuped": cuped.ci_high - cuped.ci_low}

    # 4. Guardrail: refund rate must not rise by more than the margin.
    out["refund_guardrail"] = st.noninferiority_test(
        c.refunded.sum(), len(c), t.refunded.sum(), len(t), refund_margin, alpha)

    # 5. Decision.
    if not out["refund_guardrail"]["passed"]:
        out["decision"], out["reason"] = "DON'T SHIP", "Refund guardrail not cleared."
    elif conv.significant and conv.diff > 0:
        out["decision"] = "SHIP"
        out["reason"] = (f"Conversion +{conv.diff * 100:.2f} pp ({conv.rel_lift:+.1%}), "
                         f"95% CI [{conv.ci_low * 100:+.2f}, {conv.ci_high * 100:+.2f}] pp; "
                         "refund guardrail cleared.")
    elif conv.significant:
        out["decision"], out["reason"] = "DON'T SHIP", "Conversion dropped significantly."
    else:
        out["decision"] = "NO DECISION"
        out["reason"] = "Conversion effect not significant; the test may be underpowered."
    return out


def format_report(r: dict) -> str:
    lines = ["# Experiment readout", "", f"**Decision: {r['decision']}**: {r['reason']}", ""]
    s = r["srm"]
    lines += ["## 1. Sample ratio check",
              f"Control {s['n_control']:,} / treatment {s['n_treatment']:,} "
              f"(share {s['share_treatment']:.2%}), chi-square p = {s['p_value']:.3g} "
              f"-> {'SRM DETECTED' if s['srm'] else 'OK'}", ""]
    if r["decision"] == "INVALID":
        return "\n".join(lines)

    def row(m: st.TestResult, scale: float, unit: str) -> str:
        return (f"| {m.metric} | {m.control * scale:.3f}{unit} | {m.treatment * scale:.3f}{unit} | "
                f"{m.diff * scale:+.3f}{unit} | [{m.ci_low * scale:+.3f}, {m.ci_high * scale:+.3f}] | "
                f"{m.rel_lift:+.2%} | {m.p_value:.4f} |")

    cu = r["cuped"]
    g = r["refund_guardrail"]
    lines += ["## 2. Metrics", "",
              "| Metric | Control | Treatment | Diff | 95% CI | Rel. lift | p-value |",
              "|---|---|---|---|---|---|---|",
              row(r["conversion"], 100, "%"), row(r["revenue_raw"], 1, ""),
              row(r["revenue_cuped"], 1, ""), "",
              f"CUPED (theta = {cu['theta']:.3f}) removed {cu['variance_reduction']:.1%} of revenue "
              f"variance; CI width {cu['ci_width_raw']:.3f} -> {cu['ci_width_cuped']:.3f}.", "",
              "## 3. Guardrail: refund rate",
              f"{g['control']:.3%} -> {g['treatment']:.3%}; one-sided 95% upper bound on the increase "
              f"{g['upper_bound'] * 100:+.3f} pp vs margin {g['margin'] * 100:.1f} pp -> "
              f"{'PASSED' if g['passed'] else 'FAILED'}"]
    return "\n".join(lines)
