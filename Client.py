import os, hmac, hashlib, time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from storage import Session, Run

# ---------- helpers ----------
def _all_present(*vals):
    return all(v is not None for v in vals)

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _get_param(name: str):
    try:
        qp = st.query_params
        v = qp.get(name)
        # Streamlit often returns lists; normalize
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        return st.experimental_get_query_params().get(name, [None])[0]

def _get_secret(name: str, default=None):
    val = os.environ.get(name)
    if val not in (None, ""):
        return val
    try:
        return st.secrets[name]
    except Exception:
        return default

def _sign(run_id: str) -> str:
    secret = _get_secret("TUM_SIGNING_SECRET", "dev")
    return hmac.new(secret.encode(), msg=run_id.encode(), digestmod=hashlib.sha256).hexdigest()

def _verify(run_id: str, sig: str) -> bool:
    return bool(run_id and sig and hmac.compare_digest(sig, _sign(run_id)))

def _axis_from_data(arr, q_lo=0.5, q_hi=99.5, pad_frac=0.02, min_span=2.0):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo, hi = np.nanpercentile(a, [q_lo, q_hi])
    span = max(min_span, hi - lo)
    pad = max(0.5, pad_frac * span)
    return float(lo - pad), float(hi + pad)

# ---------- friendly fallback when opened from sidebar ----------
run_id = _get_param("run_id")
sig    = _get_param("sig")

if not run_id or not sig:
    st.info("This page is for clients opening a **signed link** generated on the admin page")
    st.markdown("[← Back to app](/)")
    st.stop()

# ---------- verify + load snapshot ----------
if not _verify(run_id, sig):
    st.error("Invalid or expired link.")
    st.stop()

with Session() as sess:
    row = sess.query(Run).filter(Run.id == run_id).first()

if not row:
    st.error("Run not found.")
    st.stop()

out = row.outputs.get("figs_data", {})
summary = row.outputs.get("summary", {})
age = int(row.inputs.get("age", 0)) if isinstance(row.inputs, dict) else 0

# Safe getters
def A(name, default=None):
    v = out.get(name, default)
    return np.asarray(v, dtype=float) if v is not None else None

# ---------- Overview ----------
st.title("Time Utility Model")

# 1. REFINED TOOLTIP AND LABELS
# We explain that one is the "math" answer, and one is the "reality" (luck) answer.
hist_mode = st.radio(
    "Simulation View",
    ["Average Scenario (Deterministic)", "Range of Possibilities (Monte Carlo)"],
    index=0, 
    horizontal=True,
    help=(
        "**Average Scenario:** Assumes you live exactly as long as your biological age predicts. "
        "Useful for planning.\n\n"
        "**Range of Possibilities:** Introduces statistical 'luck' (good and bad). "
        "Shows that even with good habits, random health events can still happen."
    )
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# 2. DYNAMIC METRIC CALCULATION
life_det = _first_not_none(A("projected_life_frac"), A("projected_life"))
life_mc  = _first_not_none(A("projected_life_mc"), life_det)

# Logic: If they want "Range of Possibilities", we show the median of the random runs.
# If they want "Average Scenario", we show the deterministic threshold crossing.
if "Range" in hist_mode:
    current_life_data = life_mc
    metric_label = "Simulated Lifespan (Median outcome)"
    sub_label = "Includes statistical randomness"
else:
    current_life_data = life_det
    metric_label = "Projected Lifespan (Biological)"
    sub_label = "Based purely on your hazard curve"

if current_life_data is not None:
    median_life_val = float(np.median(current_life_data))
else:
    median_life_val = float(summary.get("median_life", np.nan))

p5, p95 = (summary.get("range_90") or [np.nan, np.nan])
median_net = float(summary.get("median_net_mm", np.nan))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(metric_label, f"{median_life_val:.0f} years")
    if age:
        # Make the years remaining label dynamic too, to match the tone
        st.metric("Years remaining", f"{int(round(median_life_val - age))} years")

with c2:
    st.metric("Net worth (median, $MM)", f"{median_net:,.2f}")
    
    # Keep this STATIC, but label it as "Benefit" so they know it's an input
    yrs_int = A("yrs_added_interventions")
    if yrs_int is not None:
        val = float(np.nansum(yrs_int))
        st.metric(
            "Benefit from Habits", 
            f"+{val:.1f} years",
            help="The expected extra time your habits add to your life, regardless of luck."
        )

with c3:
    st.metric("90% Confidence Range", f"{float(p5):.0f}-{float(p95):.0f} years")
    
    # Keep this STATIC, label as "Benefit"
    le = _first_not_none(A("threshold_series"), A("le_threshold_series"))
    bio = A("bio_age")
    tech = A("tech_years_by_age")
    if le is not None and bio is not None and tech is not None:
        alive = (bio < le[None, :]).astype(float)
        exp_tech_by_age = (tech * alive).mean(axis=0)
        val_tech = float(np.nansum(exp_tech_by_age))
        st.metric(
            "Benefit from Treatments", 
            f"+{val_tech:.1f} years",
            help="The expected extra time future treatments add, accounting for their arrival probability."
        )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------- Charts row 1: lifespan histogram + wealth vs life ----------

life_for_hist = life_mc if hist_mode.startswith("Random") else life_det

if life_for_hist is not None:
    if hist_mode.startswith("Random"):
        nbins = int(np.ptp(life_for_hist)) + 1 if np.ptp(life_for_hist) >= 1 else 10
        fig_life = px.histogram(
            life_for_hist, 
            nbins=nbins, 
            title="Simulated Ages at Death",
            # 3. FIX CHART LABELS: Capitalized Y-axis, clean X-axis
            labels={"value": "Age at death (years)", "count": "Simulations"}
        )
    else:
        fig_life = px.histogram(
            life_for_hist, 
            title="Predicted Lifespan",
            labels={"value": "Age at death (years)", "count": "Simulations"}
        )
        fig_life.update_traces(xbins=dict(size=0.5))
    
    x_lo, x_hi = _axis_from_data(life_for_hist)
    fig_life.update_xaxes(range=[x_lo, x_hi])
    
    # 3. FIX LEGEND: Hide the "variable" legend
    fig_life.update_layout(showlegend=False)
    # Force Y-axis title capitalization
    fig_life.update_yaxes(title_text="Simulations")

else:
    fig_life = px.histogram([0], title="No lifespan data")

# ---------- Wealth vs lifespan (match admin logic) ----------
bal_with  = A("balance_path")
net_at_death = A("net_worth")  # $MM array saved in snapshot

# Use MC lifespans when "Random" is selected, otherwise deterministic
life_for_net = life_mc if hist_mode.startswith("Random") else life_det

if hist_mode.startswith("Random") and bal_with is not None and life_mc is not None:
    # Pick net worth at each draw’s MC death year; convert $ to $MM
    idx = np.clip((life_mc - age).astype(int), 0, bal_with.shape[1] - 1)
    nw = bal_with[np.arange(idx.size), idx] / 1e6
else:
    nw = net_at_death  # already $MM

# Bucket purchases exactly like admin
ty = A("tech_years_by_age")
bucket = None
if ty is not None:
    starts = (ty > 0) & np.concatenate(
        [np.ones((ty.shape[0], 1), dtype=bool), ty[:, :-1] == 0],
        axis=1
    )
    purchases = starts.sum(axis=1)
    bucket = np.where(purchases == 0, "0",
              np.where(purchases == 1, "1",
              np.where(purchases <= 3, "2-3", "4+")))

# Build dataframe and plot
if life_for_net is not None and nw is not None:
    df_nw = pd.DataFrame({"Life": life_for_net, "NetWorth": nw})
    if bucket is not None:
        df_nw["Purchases"] = bucket

    if "Purchases" in df_nw.columns:
        fig_nw = px.scatter(
            df_nw, x="Life", y="NetWorth", color="Purchases",
            title="Simulated Wealth Outcomes by Lifespan",
            labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"},
            opacity=0.55
        )
    else:
        fig_nw = px.scatter(
            df_nw, x="Life", y="NetWorth",
            title="Simulated Wealth Outcomes by Lifespan",
            labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"},
            opacity=0.55
        )

    x_lo2, x_hi2 = _axis_from_data(df_nw["Life"])
    y_lo2, y_hi2 = _axis_from_data(df_nw["NetWorth"], pad_frac=0.04, min_span=0.2)
    fig_nw.update_xaxes(range=[x_lo2, x_hi2])
    fig_nw.update_yaxes(range=[y_lo2, y_hi2])
else:
    fig_nw = px.scatter(title="Wealth vs lifespan (not available)")

# render the two charts side by side
col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(fig_life, use_container_width=True)
with col_r:
    st.plotly_chart(fig_nw, use_container_width=True)

# ---------- Impact Analysis (Forest Plot) ----------
impact_data = out.get("impact_analysis")

if impact_data and len(impact_data) > 0:
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    
    # We use an expander, but default it to OPEN so they see it
    with st.expander("Impact Analysis (Forest Plot)", expanded=True):
        st.caption("How your specific habits contribute to your projected lifespan. "
                   "Green (Left) = Adds years. Red (Right) = Removes years.")
        
        df_imp = pd.DataFrame(impact_data)
        
        # Sort and Color logic
        df_imp = df_imp.sort_values("HR", ascending=False)
        df_imp["Color"] = df_imp["HR"].apply(lambda x: "#ff4b4b" if x > 1 else "#09ab3b")
        
        fig_forest = go.Figure()

        # Center Line
        fig_forest.add_vline(x=1, line_width=2, line_dash="dash", line_color="#555")

        # Dots
        fig_forest.add_trace(go.Scatter(
            x=df_imp["HR"],
            y=df_imp["Factor"],
            mode='markers',
            marker=dict(
                color=df_imp["Color"],
                size=12,
                line=dict(width=2, color="#333")
            ),
            # Tooltip
            text=[f"HR: {r['HR']:.2f}<br>Impact: {r['Delta']:+.1f} years" for i, r in df_imp.iterrows()],
            hoverinfo="text+y",
        ))

        # Layout
        fig_forest.update_layout(
            title="",
            xaxis=dict(
                title="Hazard Ratio (Log Scale)",
                type="log",
                tickvals=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                ticktext=["0.5x", "0.75x", "1.0x", "1.5x", "2.0x", "3.0x"],
                range=[np.log10(0.4), np.log10(3.5)]
            ),
            yaxis=dict(title="", type="category", tickfont=dict(size=14)),
            height=max(250, 80 + (len(df_imp) * 40)),
            margin=dict(l=0, r=0, t=10, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_forest, use_container_width=True)

# ---------- Years-added area ----------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
mode = st.selectbox(
    "Added Years Breakdown",
    ["From Health Habits", "From Habits + Treatments (survivors only)", "From Future Treatments (survivors only)"],
    index=0
)

yrs_int = A("yrs_added_interventions")
le = _first_not_none(A("threshold_series"), A("le_threshold_series"))
bio = A("bio_age")
tech = A("tech_years_by_age")
age_grid = A("chrono_age")

y = None
if yrs_int is not None and mode.startswith("From Health Habits"):
    y = yrs_int
elif _all_present(yrs_int, tech, le, bio) and mode.startswith("From Habits + Treatments"):
    alive = (bio < le[None, :]).astype(float)
    y = yrs_int + (tech * alive).mean(axis=0)
elif _all_present(tech, le, bio):
    alive = (bio < le[None, :]).astype(float)
    y = (tech * alive).mean(axis=0)

if y is not None and age_grid is not None:
    fig_yrs = px.area(x=age_grid, y=y, labels={"x": "Age", "y": "Years Added by Age"}, title=mode)
    st.plotly_chart(fig_yrs, use_container_width=True)

# ---------- Spending outcomes KPIs ----------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.subheader("Spending Outcomes")
st.caption("What purchases cost, how many years they buy, and what's left for your estate")

age_grid = A("chrono_age")
bal_with = A("balance_path")
bal_no   = A("balance_no_tech_path")
bio      = A("bio_age")
tech     = A("tech_years_by_age")
le = _first_not_none(A("threshold_series"), A("le_threshold_series"))
tech_spend = A("tech_costs_by_age")  # may be None

if _all_present(age_grid, bal_with, bal_no, bio, tech, le):
    alive_with  = (np.maximum(0.0, bio - np.cumsum(tech, axis=1)) < le[None, :])
    alive_no    = (bio < le[None, :])

    # median death ages by scenario
    def _median_death_age(alive_mask):
        surv = alive_mask.mean(axis=0)
        idx  = np.where(surv <= 0.5)[0]
        return float(age_grid[idx[0]]) if idx.size else float(age_grid[-1])

    med_age_with    = _median_death_age(alive_with)
    med_age_without = _median_death_age(alive_no)

    # alive-weighted medians for portfolio center lines
    def _median_alive(path, alive_mask):
        D, T = path.shape
        med = np.full(T, np.nan)
        for t in range(T):
            vals = path[alive_mask[:, t], t]
            if vals.size:
                med[t] = np.median(vals)
        return med

    m_with    = _median_alive(bal_with/1e6, alive_with)
    m_without = _median_alive(bal_no/1e6,   alive_no)

    # costs + ROI to WITH median age
    m = (age_grid <= med_age_with)
    if tech_spend is None:
        spend_by_draw = np.zeros(bio.shape[0])
    else:
        spend_by_draw = (tech_spend[:, m] * alive_with[:, m]).sum(axis=1)

    yrs_by_draw = (tech[:, m] * alive_with[:, m]).sum(axis=1)

    typ_cost_total = float(np.median(spend_by_draw))
    roi_draw = np.full_like(yrs_by_draw, np.nan, dtype=float)
    nz = spend_by_draw > 0
    roi_draw[nz] = yrs_by_draw[nz] / (spend_by_draw[nz] / 100000.0)
    roi_median = float(np.nanmedian(roi_draw))

    # bequest delta
    def _terminal_wealth_at_death(bal, alive_mask):
        D, T = bal.shape
        tw = np.zeros(D, dtype=float)
        for d in range(D):
            idx = np.where(alive_mask[d])[0]
            t = idx[-1] if idx.size else 0
            tw[d] = bal[d, t]
        return tw

    tw_with    = _terminal_wealth_at_death(bal_with, alive_with)
    tw_without = _terminal_wealth_at_death(bal_no,   alive_no)
    bequest_delta_med = float(np.median(tw_with) - np.median(tw_without))

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected treatment costs", f"${typ_cost_total:,.0f}")
    c2.metric("Years gained for every $100k", f"{roi_median:.2f} yrs")
    sign = "+" if bequest_delta_med > 0 else ""
    c3.metric("Money you leave behind", f"{sign}${abs(bequest_delta_med):,.0f}")

    # portfolio figure
    fig_bal = go.Figure()
    mask_with    = (age_grid <= med_age_with)
    mask_without = (age_grid <= med_age_without)
    fig_bal.add_trace(go.Scatter(x=age_grid[mask_with],    y=m_with[mask_with],       mode="lines", name="With treatments"))
    fig_bal.add_trace(go.Scatter(x=age_grid[mask_without], y=m_without[mask_without], mode="lines", name="Without treatments"))
    fig_bal.update_layout(title="Portfolio value by age", xaxis_title="Age (years)", yaxis_title="Balance ($MM)")
    fig_bal.update_xaxes(range=[float(age_grid[0]), max(med_age_with, med_age_without)])
    st.plotly_chart(fig_bal, use_container_width=True)

# ---------- Diagnostics ----------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
with st.expander("Health vs Life Expectancy", expanded=False):
    le = _first_not_none(A("threshold_series"), A("le_threshold_series"))
    bio = A("bio_age")
    age_grid = A("chrono_age")
    if _all_present(le, bio, age_grid):
        df = pd.DataFrame({
            "Age": age_grid,
            "Biological Age (mean)": bio.mean(axis=0),
            "Life Expectancy": le,
        })
        fig_diag = px.line(df, x="Age", y=["Biological Age (mean)", "Life Expectancy"],
                           title="Biological Age vs Societal Life Expectancy",
                           labels={"Age": "Age (years)", "value": "Years", "variable": ""})
        fig_diag.update_layout(legend_title_text="")
        st.plotly_chart(fig_diag, use_container_width=True)

with st.expander("Personal Finances, First 20 Years"):
    age_grid = A("chrono_age")
    di  = A("discretionary_income_by_year")
    hsp = A("health_spend_by_year")
    con = A("contrib_by_year")
    if _all_present(age_grid, di, hsp, con):
        df_fin = pd.DataFrame({
            "Age": age_grid,
            "Discretionary Income": di,
            "Total Healthcare Spending": hsp,
            "Excess Cash for Investments": con,
        })
        st.dataframe(df_fin.head(20), use_container_width=True, height=480)
