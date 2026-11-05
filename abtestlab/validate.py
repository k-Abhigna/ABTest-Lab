"""Monte Carlo checks: does the toolkit behave the way the theory says it should?

Run:  python -m abtestlab.validate [--sims 400]
Writes reports/validation.json and reports/validation.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from . import stats as st
from .analyze import analyze
from .simulate import BASE_CONV, TRUE_LIFT_REL, simulate_experiment

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def experiment_sims(n_sims: int, seed0: int = 1000) -> dict:
    """Repeat the whole experiment: A/A (no effect), A/B (true +8%), A/B with SRM bug."""
    rec = {k: [] for k in ("aa_sig", "aa_srm", "ab_sig", "ab_rev_raw_sig", "ab_rev_cuped_sig",
                           "ab_lift", "ab_cuped_diff", "ab_raw_diff", "var_red",
                           "bug_srm", "bug_lift")}
    for i in range(n_sims):
        aa = analyze(simulate_experiment(lift_rel=0.0, seed=seed0 + i), stop_on_srm=False)
        rec["aa_srm"].append(aa["srm"]["srm"])
        rec["aa_sig"].append(aa["conversion"].significant)

        ab = analyze(simulate_experiment(seed=seed0 + i), stop_on_srm=False)
        rec["ab_sig"].append(ab["conversion"].significant)
        rec["ab_lift"].append(ab["conversion"].rel_lift)
        rec["ab_rev_raw_sig"].append(ab["revenue_raw"].significant)
        rec["ab_rev_cuped_sig"].append(ab["revenue_cuped"].significant)
        rec["ab_raw_diff"].append(ab["revenue_raw"].diff)
        rec["ab_cuped_diff"].append(ab["revenue_cuped"].diff)
        rec["var_red"].append(ab["cuped"]["variance_reduction"])

        bug_df = simulate_experiment(seed=seed0 + i, srm_bug=True)
        bug = analyze(bug_df)
        rec["bug_srm"].append(bug["srm"]["srm"])
        c, t = bug_df[bug_df.group == "control"], bug_df[bug_df.group == "treatment"]
        rec["bug_lift"].append(t.converted.mean() / c.converted.mean() - 1)
    r = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    return {
        "n_sims": n_sims,
        "aa_false_positive_rate": r["aa_sig"].mean(),
        "aa_srm_false_alarm_rate": r["aa_srm"].mean(),
        "ab_power_conversion": r["ab_sig"].mean(),
        "ab_mean_rel_lift": r["ab_lift"].mean(),
        "ab_power_revenue_raw": r["ab_rev_raw_sig"].mean(),
        "ab_power_revenue_cuped": r["ab_rev_cuped_sig"].mean(),
        "cuped_mean_variance_reduction": r["var_red"].mean(),
        "revenue_diff_mean_raw": r["ab_raw_diff"].mean(),
        "revenue_diff_mean_cuped": r["ab_cuped_diff"].mean(),
        "revenue_diff_sd_raw": r["ab_raw_diff"].std(ddof=1),
        "revenue_diff_sd_cuped": r["ab_cuped_diff"].std(ddof=1),
        "srm_bug_detection_rate": r["bug_srm"].mean(),
        "srm_bug_mean_rel_lift": r["bug_lift"].mean(),
    }


def peeking_sims(n_sims: int = 20_000, n_looks: int = 14, users_per_look: int = 3300,
                 p: float = BASE_CONV, alpha: float = 0.05, seed: int = 7) -> dict:
    """A/A tests checked once a day. Stopping at the first p < 0.05 inflates false positives."""
    rng = np.random.default_rng(seed)
    half = users_per_look // 2
    xc = rng.binomial(half, p, (n_sims, n_looks)).cumsum(axis=1)
    xt = rng.binomial(half, p, (n_sims, n_looks)).cumsum(axis=1)
    n = half * np.arange(1, n_looks + 1)
    pool = (xc + xt) / (2 * n)
    z = (xt - xc) / n / np.sqrt(pool * (1 - pool) * 2 / n)
    z_fixed = stats.norm.ppf(1 - alpha / 2)
    bound = st.pocock_boundary(n_looks, alpha)             # calibrated on separate draws
    fpr_by_look = [(np.abs(z[:, :k]) > z_fixed).any(axis=1).mean() for k in range(1, n_looks + 1)]
    return {
        "n_sims": n_sims, "n_looks": n_looks,
        "fpr_single_final_look": float((np.abs(z[:, -1]) > z_fixed).mean()),
        "fpr_peek_every_day": float(fpr_by_look[-1]),
        "fpr_by_number_of_looks": [round(float(f), 4) for f in fpr_by_look],
        "pocock_z_boundary": bound,
        "fpr_peek_with_pocock": float((np.abs(z) > bound).any(axis=1).mean()),
    }


def planning() -> dict:
    n = st.sample_size_proportion(BASE_CONV, TRUE_LIFT_REL)
    return {"baseline": BASE_CONV, "mde_rel": TRUE_LIFT_REL, "alpha": 0.05, "power": 0.8,
            "users_per_group": n, "total_users": 2 * n, "days_at_3300_per_day": int(np.ceil(2 * n / 3300))}


def to_markdown(v: dict) -> str:
    e, pk, pl = v["experiments"], v["peeking"], v["planning"]
    return "\n".join([
        "# Validation (Monte Carlo)", "",
        f"Planned: {pl['users_per_group']:,} users/group to detect +{pl['mde_rel']:.0%} on a "
        f"{pl['baseline']:.0%} baseline (alpha 0.05, power 0.80) -> 14 days at 3,300 users/day.", "",
        f"## Whole-experiment simulations ({e['n_sims']} runs each)", "",
        "| Check | Expected | Observed |", "|---|---|---|",
        f"| A/A false-positive rate | 5% | {e['aa_false_positive_rate']:.1%} |",
        f"| A/A SRM false alarms | 0.1% | {e['aa_srm_false_alarm_rate']:.1%} |",
        f"| Power, conversion (true +8%) | 80% | {e['ab_power_conversion']:.1%} |",
        f"| Mean estimated lift | +8.0% | {e['ab_mean_rel_lift']:+.2%} |",
        f"| Power, revenue/user: raw -> CUPED | CUPED higher | {e['ab_power_revenue_raw']:.1%} -> "
        f"{e['ab_power_revenue_cuped']:.1%} |",
        f"| Revenue diff, mean raw vs CUPED (no bias) | equal | {e['revenue_diff_mean_raw']:.3f} vs "
        f"{e['revenue_diff_mean_cuped']:.3f} |",
        f"| Revenue diff, SD raw vs CUPED | CUPED lower | {e['revenue_diff_sd_raw']:.3f} vs "
        f"{e['revenue_diff_sd_cuped']:.3f} |",
        f"| CUPED variance reduction | rho^2 | {e['cuped_mean_variance_reduction']:.1%} |",
        f"| SRM bug detected | high | {e['srm_bug_detection_rate']:.1%} |",
        f"| Lift reported when SRM bug is ignored | +8.0% (truth) | {e['srm_bug_mean_rel_lift']:+.2%} |",
        "", f"## Peeking ({pk['n_sims']:,} A/A tests, {pk['n_looks']} daily looks)", "",
        "| Strategy | False-positive rate |", "|---|---|",
        f"| Look once at the end | {pk['fpr_single_final_look']:.1%} |",
        f"| Look daily, stop at first p < 0.05 | {pk['fpr_peek_every_day']:.1%} |",
        f"| Look daily with Pocock boundary \\|z\\| > {pk['pocock_z_boundary']:.2f} | "
        f"{pk['fpr_peek_with_pocock']:.1%} |",
    ])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    args = ap.parse_args(argv)
    v = {"planning": planning(), "experiments": experiment_sims(args.sims), "peeking": peeking_sims()}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation.json").write_text(json.dumps(v, indent=2, default=float))
    md = to_markdown(v)
    (REPORTS / "validation.md").write_text(md, encoding="utf-8")
    print(md)
    return v


if __name__ == "__main__":
    main()
