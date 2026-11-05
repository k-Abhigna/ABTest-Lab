"""ABTest-Lab dashboard:  streamlit run app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from abtestlab import stats as abst
from abtestlab.analyze import analyze, format_report
from abtestlab.simulate import simulate_experiment

st.set_page_config(page_title="ABTest-Lab", layout="wide")
st.title("ABTest-Lab: A/B test readouts you can trust")
tab_plan, tab_read, tab_peek = st.tabs(["1. Plan", "2. Analyse", "3. Why not peek?"])

with tab_plan:
    st.subheader("How many users do I need?")
    c1, c2, c3, c4 = st.columns(4)
    base = c1.number_input("Baseline conversion", 0.001, 0.99, 0.10, 0.01, format="%.3f")
    mde = c2.number_input("Smallest lift worth detecting (relative)", 0.005, 2.0, 0.08, 0.01)
    power = c3.slider("Power", 0.5, 0.99, 0.8)
    daily = c4.number_input("Eligible users per day", 100, 10_000_000, 3300, 100)
    n = abst.sample_size_proportion(base, mde, power=power)
    st.metric("Users per group", f"{n:,}", f"{-(-2 * n // daily)} days of traffic", delta_color="off")
    st.caption("Two-sided alpha = 0.05. Pick the MDE from business value, not from what traffic allows.")

with tab_read:
    src = st.radio("Data", ["Simulated checkout test", "Simulated test with SRM bug", "Upload CSV"],
                   horizontal=True)
    if src == "Upload CSV":
        up = st.file_uploader("Columns: group, converted, revenue, pre_revenue, refunded", type="csv")
        df = pd.read_csv(up) if up else None
    else:
        seed = st.number_input("Seed", 0, 10_000, 42)
        df = simulate_experiment(seed=int(seed), srm_bug=src.endswith("bug"))
    if df is not None:
        try:
            r = analyze(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        colour = {"SHIP": "green", "INVALID": "red", "DON'T SHIP": "red"}.get(r["decision"], "orange")
        st.markdown(f"### :{colour}[{r['decision']}]")
        if r["decision"] != "INVALID":
            fig = go.Figure()
            for m, scale, label in [(r["conversion"], 100, "Conversion (pp)"),
                                    (r["revenue_raw"], 1, "Revenue/user raw ($)"),
                                    (r["revenue_cuped"], 1, "Revenue/user CUPED ($)")]:
                fig.add_trace(go.Scatter(x=[m.diff * scale], y=[label], mode="markers", showlegend=False,
                                         error_x=dict(type="data", symmetric=False,
                                                      array=[(m.ci_high - m.diff) * scale],
                                                      arrayminus=[(m.diff - m.ci_low) * scale]),
                                         marker=dict(size=11)))
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(title="Treatment minus control, 95% CI", height=280)
            st.plotly_chart(fig, width="stretch")
        st.markdown(format_report(r).split("\n", 2)[2])

with tab_peek:
    st.subheader("Checking results every day turns 5% false positives into ~22%")
    vpath = Path(__file__).parent / "reports" / "validation.json"
    if vpath.exists():
        pk = json.loads(vpath.read_text())["peeking"]
    else:
        from abtestlab.validate import peeking_sims
        pk = peeking_sims(n_sims=5000)
    looks = list(range(1, pk["n_looks"] + 1))
    fig = go.Figure(go.Scatter(x=looks, y=[f * 100 for f in pk["fpr_by_number_of_looks"]],
                               mode="lines+markers", name="Stop at first p < 0.05"))
    fig.add_hline(y=5, line_dash="dash", annotation_text="Promised 5%")
    fig.update_layout(xaxis_title="Number of daily looks", yaxis_title="False-positive rate (%)",
                      height=380)
    st.plotly_chart(fig, width="stretch")
    st.write(f"Simulated {pk['n_sims']:,} A/A tests (no real effect). Using a Pocock boundary of "
             f"|z| > {pk['pocock_z_boundary']:.2f} at every look brings the rate back to "
             f"{pk['fpr_peek_with_pocock']:.1%}.")
