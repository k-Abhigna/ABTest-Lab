import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtestlab import stats as st
from abtestlab.analyze import analyze, format_report
from abtestlab.simulate import simulate_experiment
from abtestlab.validate import peeking_sims


# ---------------------------------------------------------------- formulas vs SciPy / textbooks
def test_ztest_matches_chi_square():
    r = st.two_proportion_ztest(1000, 10000, 1100, 10000)
    _, p, _, _ = stats.chi2_contingency([[1000, 9000], [1100, 8900]], correction=False)
    assert r.p_value == pytest.approx(p, rel=1e-9)
    assert r.diff == pytest.approx(0.01)


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 500), rng.normal(10.4, 3, 700)
    r = st.welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    ci = ref.confidence_interval()
    assert (r.ci_low, r.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)


def test_sample_size_textbook_value():
    # 10% -> 12% (20% relative), alpha 0.05, power 0.8: 3,841 per group (standard formula)
    assert st.sample_size_proportion(0.10, 0.20) == 3841
    assert st.sample_size_mean(sd=1.0, mde_abs=0.2) == 393


@pytest.mark.parametrize("k, table", [(2, 2.178), (5, 2.413), (10, 2.555)])
def test_pocock_boundary_matches_published_table(k, table):
    assert st.pocock_boundary(k, n_sim=400_000) == pytest.approx(table, abs=0.015)


def test_srm_check():
    assert not st.srm_check(5000, 5050)["srm"]
    assert st.srm_check(5000, 5400)["srm"]


def test_noninferiority():
    assert st.noninferiority_test(2000, 40000, 2020, 40000, margin=0.005)["passed"]
    assert not st.noninferiority_test(500, 10000, 505, 10000, margin=0.005)["passed"]  # underpowered
    assert not st.noninferiority_test(500, 10000, 600, 10000, margin=0.005)["passed"]


def test_cuped_is_unbiased_and_reduces_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 20000)
    y = 0.7 * x + rng.normal(0, 1, 20000)
    y_adj, theta = st.cuped_adjust(y, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert theta == pytest.approx(0.7, abs=0.03)
    assert y_adj.mean() == pytest.approx(y.mean())
    assert 1 - y_adj.var() / y.var() == pytest.approx(rho ** 2, abs=1e-6)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    a, b = rng.exponential(10, 3000), rng.exponential(11, 3000)
    lo, hi = st.bootstrap_rel_lift_ci(a, b, n_boot=500)
    assert lo < b.mean() / a.mean() - 1 < hi


# ---------------------------------------------------------------- simulator
def test_simulator_hits_its_targets():
    df = simulate_experiment(users_per_day=20000, lift_rel=0.08, seed=3)
    conv = df.groupby("group").converted.mean()
    assert conv["control"] == pytest.approx(0.10, abs=0.004)
    assert conv["treatment"] / conv["control"] - 1 == pytest.approx(0.08, abs=0.03)
    assert df.revenue.corr(df.pre_revenue) > 0.3


def test_srm_bug_drops_treatment_users():
    clean, bug = simulate_experiment(seed=5), simulate_experiment(seed=5, srm_bug=True)
    assert (bug.group == "control").sum() == (clean.group == "control").sum()
    assert (bug.group == "treatment").sum() < (clean.group == "treatment").sum() * 0.97


# ---------------------------------------------------------------- end-to-end decisions
def test_default_experiment_ships():
    r = analyze(simulate_experiment())
    assert r["decision"] == "SHIP"
    assert r["cuped"]["ci_width_cuped"] < r["cuped"]["ci_width_raw"]
    assert "SHIP" in format_report(r)


def test_srm_blocks_decision():
    r = analyze(simulate_experiment(srm_bug=True))
    assert r["decision"] == "INVALID" and "conversion" not in r
    assert "SRM DETECTED" in format_report(r)


def test_no_effect_gives_no_decision():
    assert analyze(simulate_experiment(lift_rel=0.0, seed=11))["decision"] == "NO DECISION"


def test_guardrail_failure_blocks_ship():
    df = simulate_experiment()
    buyers_t = df.index[(df.group == "treatment") & (df.converted == 1)]
    df.loc[buyers_t[: len(buyers_t) // 4], "refunded"] = 1        # treatment refunds jump
    r = analyze(df)
    assert r["decision"] == "DON'T SHIP" and not r["refund_guardrail"]["passed"]


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        analyze(pd.DataFrame({"group": ["control"], "converted": [1]}))


def test_peeking_inflates_and_pocock_fixes():
    pk = peeking_sims(n_sims=5000)
    assert pk["fpr_single_final_look"] == pytest.approx(0.05, abs=0.012)
    assert pk["fpr_peek_every_day"] > 0.15
    assert pk["fpr_peek_with_pocock"] == pytest.approx(0.05, abs=0.012)


def test_dashboard_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].value == "22,855"
    at.radio[0].set_value("Simulated test with SRM bug").run()
    assert not at.exception and "INVALID" in at.markdown[0].value
