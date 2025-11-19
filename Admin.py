from storage import Session, Client, Run
from models import RunInputs, RunOutputs
import os, hmac, hashlib
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from engine import Inputs, Tier, run_monte_carlo, IntervCost

from storage import engine, ensure_tables
print("DB URL seen by app:", engine.url)
ensure_tables()

# Safe secret/env access
def _get_secret(name: str, default=None):
    # Prefer env var
    val = os.environ.get(name)
    if val not in (None, ""):
        return val
    # Only touch st.secrets if it’s actually configured
    try:
        return st.secrets[name]  # will raise if secrets.toml missing
    except Exception:
        return default

def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

# --- link builder that survives file renames ---
from urllib.parse import quote
import hmac, hashlib

CLIENT_PAGE_LABEL = "Client"  # No longer needed for URL generation

def _client_link_for(run_id: str) -> str:
    secret = _get_secret("TUM_SIGNING_SECRET", "dev")
    sig = hmac.new(secret.encode(), msg=run_id.encode(), digestmod=hashlib.sha256).hexdigest()
    base = _get_secret("PUBLIC_BASE_URL", "http://localhost:8501")
    
    # UPDATED LINE: Clean URL routing
    return f"{base}/client?run_id={run_id}&sig={sig}"

def saved_runs_drawer(title: str = "Saved runs"):
    from storage import Session, Run, Client
    from sqlalchemy import or_
    from sqlalchemy.orm import defer  # <-- NEW IMPORT
    import datetime as dt

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.expander(title, expanded=False):
        sess = Session()

        # Scope to current org
        org_id = st.session_state.get("org_id")

        # OPTIMIZATION: We use .options(defer(...)) to tell Postgres 
        # NOT to send the massive JSON blobs for this list view.
        q = (sess.query(Run, Client)
                .options(defer(Run.inputs), defer(Run.outputs)) 
                .join(Client, Run.client_id == Client.id, isouter=True)
                .filter(Run.org_id == org_id))

        term = st.text_input("Search by client, email, or run id",
                             key=f"runs_search_{title.replace(' ', '_')}").strip()
        
        if term:
            like = f"%{term}%"
            q = q.filter(or_(Client.name.ilike(like), Client.email.ilike(like), Run.id.ilike(like)))

        rows = q.order_by(Run.created_at.desc()).limit(20).all()

        if not rows:
            st.caption("No saved runs yet.")
            return

        for (r, c) in rows:
            who = (c.name or "Unknown") + (f" · {c.email}" if c and c.email else "")
            when = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else ""
            header = f"{who} · {when}"
            with st.expander(header):
                link = _client_link_for(r.id)

                st.text_input("Client link", value=link, key=f"link_{r.id}")

                cols = st.columns([2, 1, 1])

                cols[0].markdown(f"[Open client view]({link})")
                if cols[1].button("Copy link", key=f"copy_{r.id}"):
                    st.toast("Link ready to copy above.")
                if cols[2].button("Delete run", key=f"del_{r.id}"):
                    sess.delete(r); sess.commit()
                    st.success("Deleted. Refreshing…")
                    st.rerun()

# ---------- Auth: hard gate ----------
def _sb_client():
    url = _get_secret("SUPABASE_URL", "")
    key = _get_secret("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    from supabase import create_client
    return create_client(url, key)

def require_auth():
    """
    Gate the app behind Supabase Auth.
    Provides a toggle to Sign in or Create account.
    On first successful auth, mirror the user into our Org/User tables.
    """
    # Already authed this session?
    if st.session_state.get("user") and st.session_state.get("org_id"):
        return st.session_state["user"]

    sb = _sb_client()
    if not sb:
        st.error("Auth is not configured on this deployment.")
        st.stop()

    st.title("Sign in")

    mode = st.radio("",
                    ["Sign in", "Create account"],
                    horizontal=True, label_visibility="collapsed")

    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if mode == "Create account":
            pw2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Continue")

    if submitted:
        try:
            if mode == "Create account":
                if not email or not pw:
                    st.error("Email and password are required")
                    st.stop()
                if pw != pw2:
                    st.error("Passwords don't match")
                    st.stop()

                # Create user
                res = sb.auth.sign_up({"email": email, "password": pw})

                # If email confirmations are ON, there may be no session yet.
                # Try immediate password sign-in to keep local dev smooth.
                try:
                    res = sb.auth.sign_in_with_password({"email": email, "password": pw})
                except Exception:
                    pass

            else:
                res = sb.auth.sign_in_with_password({"email": email, "password": pw})

            user_obj = getattr(res, "user", None)
            if not user_obj:
                # Supabase can return None when email confirmation is required.
                st.error("Auth error: Invalid login credentials or email not confirmed, check your inbox to verify")
                st.stop()

            # Mirror to our DB on first login
            from storage import Session, Org, User
            sess = Session()
            u = sess.get(User, user_obj.id)
            if not u:
                org = Org(name=f"{email.split('@')[0]}'s org")
                sess.add(org); sess.flush()
                u = User(id=user_obj.id, email=email, org_id=org.id)
                sess.add(u); sess.commit()

            st.session_state["user"] = {"id": u.id, "email": u.email}
            st.session_state["org_id"] = u.org_id
            st.rerun()

        except Exception as e:
            st.error(f"Auth error: {e}")
            st.stop()

    # Block the rest of the app until authenticated
    st.stop()

# ---------- Percent helpers (UI shows %, engine gets 0–1) ----------
def _infer_digits(step_pct: float) -> int:
    # 5.0 -> 0, 0.1 -> 1, 0.01 -> 2, etc.
    s = str(step_pct)
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0

def pct_slider(label, *, min_pct=0.0, max_pct=100.0, value_pct=0.0,
               step_pct=0.1, key=None, help=None, sidebar=True,
               period=None, digits=None, fmt=None):
    """
    digits: number of decimal places to display (overrides inference)
    fmt: full Streamlit format string, e.g. "%.0f%%" (overrides digits)
    """
    unit = f" (% {period})" if period else " (%)"
    if fmt is None:
        d = _infer_digits(step_pct) if digits is None else digits
        fmt = f"%.{d}f%%"

    args = dict(
        label=label + unit,
        min_value=float(min_pct),
        max_value=float(max_pct),
        value=float(value_pct),
        step=float(step_pct),
        format=fmt,
        key=key, help=help
    )
    v = (st.sidebar.slider(**args) if sidebar else st.slider(**args))
    return v / 100.0  # normalize to 0–1 for the engine

# ---------- Axis helpers ----------
def _axis_from_data(arr, q_lo=0.5, q_hi=99.5, pad_frac=0.02, min_span=2.0):
    """
    Build tight (lo, hi) axis limits from data quantiles, with a small pad
    and a minimum span so tiny samples still render sensibly.
    """
    import numpy as np
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo, hi = np.nanpercentile(a, [q_lo, q_hi])
    span = max(min_span, hi - lo)
    pad = max(0.5, pad_frac * span)
    return float(lo - pad), float(hi + pad)

# ---------- Hazard-dose helper ----------
def _apply_dose_log(hr_full: float, a: float) -> float:
    """Scale a hazard ratio by exposure fraction a in [0,1] on the log scale."""
    a = max(0.0, min(1.0, float(a)))
    return float(hr_full) ** a

def _scale_hr(base_hr: float, adherence: float, *, mode: str = "log", curvature: float = 1.0) -> float:
    """
    Scale a base HR by adherence in [0,1].

    mode="log"     -> base_hr ** adherence  (actuarial symmetry; time-mixing invariant)
    mode="linear"  -> 1 - adherence*(1 - base_hr)  (legacy partial credit)

    curvature only applies in log mode: adherence_effect = adherence**curvature
    curvature > 1 softens mid-adherence; curvature < 1 strengthens it.
    """
    a = max(0.0, min(1.0, float(adherence)))
    if mode == "linear":
        return 1.0 - a * (1.0 - float(base_hr))
    eff = a ** max(0.1, float(curvature))
    return float(base_hr) ** eff

def _parse_float(s):
    try:
        return float(str(s).strip().replace(",", ""))
    except Exception:
        return None

# --- BMI risk mapper with safety + certainty flags ---

# Evidence band from the meta-analysis; outside this we admit we're extrapolating
BMI_EVIDENCE_MIN = 15.0
BMI_EVIDENCE_MAX = 40.0

# Absolute sanity bounds so typos don't nuke results
BMI_VALID_MIN = 12.0
BMI_VALID_MAX = 60.0

# Hard cap so extreme extrapolation stays bounded
BMI_HR_CAP = 3.0

# Anchors (single minimum at 22.5), same values you already use
BMI_HR_ANCHORS = [
    (15.0, 2.76), (18.5, 1.13),
    (22.5, 1.00),
    (25.0, 1.07), (27.5, 1.20),
    (30.0, 1.45), (35.0, 1.94), (40.0, 2.76)
]

def hr_bmi_continuous(bmi: float):
    """
    Return (hr, flag) where flag ∈ {"valid","extrapolated","invalid"}.
      - invalid: outside [BMI_VALID_MIN, BMI_VALID_MAX] → ignore BMI in results
      - extrapolated: computed outside [BMI_EVIDENCE_MIN, BMI_EVIDENCE_MAX]
      - valid: within evidence band
    """
    import numpy as np
    b = float(bmi)

    # 1) Hard validity gate
    if not (BMI_VALID_MIN <= b <= BMI_VALID_MAX):
        return None, "invalid"

    # 2) Interpolate in log(HR); linear end-slope extrapolation
    xs = np.array([x for x,_ in BMI_HR_ANCHORS], dtype=float)
    ys = np.log(np.array([y for _,y in BMI_HR_ANCHORS], dtype=float))

    if b <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y = ys[0] + slope * (b - xs[0])
    elif b >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        y = ys[-1] + slope * (b - xs[-1])
    else:
        y = float(np.interp(b, xs, ys))

    hr = float(np.exp(y))
    hr = min(hr, BMI_HR_CAP)  # 3) Safety cap

    flag = "valid" if (BMI_EVIDENCE_MIN <= b <= BMI_EVIDENCE_MAX) else "extrapolated"
    return hr, flag

def _combine_multipliers(mult_dict: dict[str, float]) -> float:
    import math
    total_log = 0.0
    for v in mult_dict.values():
        v = max(1e-9, float(v))
        total_log += math.log(v)
    return math.exp(total_log)

def pct_number(label, *, value_pct=0.0, step_pct=0.1, key=None, help=None,
               sidebar=True, period=None, min_pct=0.0, max_pct=100.0,
               digits=None, fmt=None):
    """
    number_input can't show % in the format, so we keep % in the label.
    """
    unit = f" (% {period})" if period else " (%)"
    if fmt is None:
        d = _infer_digits(step_pct) if digits is None else digits
        fmt = f"%.{d}f"

    args = dict(
        label=label + unit,
        min_value=float(min_pct),
        max_value=float(max_pct),
        value=float(value_pct),
        step=float(step_pct),
        format=fmt,
        key=key, help=help
    )
    v = (st.sidebar.number_input(**args) if sidebar else st.number_input(**args))
    return v / 100.0

# Require sign-in before anything else renders
user = require_auth()           # sets st.session_state["user"] / ["org_id"]
org_id = st.session_state["org_id"]

# Persist results across reruns so charts don't error before you run.
if "results" not in st.session_state:
    st.session_state.results = None

st.title("Time Utility Model")

# ------------------ Sidebar: profile & lifestyle types ------------------
def sign_out():
    try:
        _sb_client().auth.sign_out()
    except Exception:
        pass
    for k in ("user","org_id"):
        st.session_state.pop(k, None)
    st.rerun()

st.caption(f"Signed in as: {st.session_state['user']['email']}")

st.sidebar.header("1) Your Demographics")

age = st.sidebar.number_input("Age", min_value=18, max_value=95, value=21, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

# --- Weight status (forced exact BMI via height & weight) ---
st.sidebar.subheader("Body Mass Index")

# Units radio; hide its label to avoid clutter
unit = st.sidebar.radio(
    "", ["US (ft/in, lb)", "Metric (cm, kg)"],
    index=0, horizontal=True, key="bmi_units",
    label_visibility="collapsed"
)

bmi = None
bmi_hr = None
h_m = None
kg = None

if "US" in unit:
    # Three boxes, placeholders only
    c1, c2, c3 = st.sidebar.columns([1, 1, 1])
    ft_s   = c1.text_input("", value="", placeholder="ft", key="ht_ft",   label_visibility="collapsed")
    in_s   = c2.text_input("", value="", placeholder="in", key="ht_in",   label_visibility="collapsed")
    lb_s   = c3.text_input("", value="", placeholder="lb", key="wt_lb",   label_visibility="collapsed")

    ft   = _parse_float(ft_s)
    inch = _parse_float(in_s)
    lb   = _parse_float(lb_s)

    if None not in (ft, inch, lb) and ft >= 0 and 0 <= inch < 12 and 60 <= lb <= 600:
        h_m = (ft*12.0 + inch) * 0.0254
        kg  = lb * 0.45359237
else:
    # Two boxes, placeholders only
    c1, c2 = st.sidebar.columns(2)
    h_cm_s = c1.text_input("", value="", placeholder="cm", key="ht_cm", label_visibility="collapsed")
    kg_s   = c2.text_input("", value="", placeholder="kg", key="wt_kg", label_visibility="collapsed")

    h_cm = _parse_float(h_cm_s)
    kg   = _parse_float(kg_s)

    if None not in (h_cm, kg) and 120.0 <= h_cm <= 230.0 and 40.0 <= kg <= 300.0:
        h_m = h_cm / 100.0

# Compute BMI only when fields are valid
bmi_flag = None
if h_m and kg:
    bmi = float(kg / max(h_m*h_m, 1e-6))
    bmi_hr, bmi_flag = hr_bmi_continuous(bmi)
else:
    bmi = None
    bmi_hr = None

# Persist for results/impact panels
st.session_state["bmi"] = bmi
st.session_state["bmi_hr"] = bmi_hr
st.session_state["bmi_flag"] = bmi_flag

# Gentle guidance in the sidebar (no scoreboard)
if bmi_flag == "invalid":
    st.sidebar.warning("BMI is outside supported range (12-60). Weight will be ignored in results.")
elif bmi_flag == "extrapolated":
    st.sidebar.caption("BMI outside evidence range (15-40). Risk extrapolated; less certain.")

# Keep a stable label for downstream text/debug
weight_label = "Exact BMI"

# Default risk multipliers (HR×adherence at 100%)
BASE_RISK_MULT = {
    "Consistent Sleep": 0.88,
    "Frequent Exercise": 0.68,
    "Mediterranean Diet": 0.77,
    "Meditation": 0.93,
    "Frequent Sauna": 0.90, # ≥3–4 sessions/week; consider 0.75–0.80 for 4–7/wk
    "Red-Light Therapy": 0.98,
    "Heavy Smoking": 2.5, # harmful, editable in UI later if we want
    "Heavy Drinking": 1.25, # 3-4 drinks/day; use 1.35 for "very heavy"
}

# Preset → default toggle set
PRESET_TOGGLES = {
    "None":               {"Consistent Sleep": False, "Frequent Exercise": False, "Mediterranean Diet": False, "Meditation": False, "Red-Light Therapy": False, "Frequent Sauna": False, "Heavy Smoking": False, "Heavy Drinking": False},
    "Core Routine":            {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": False,  "Meditation": False,  "Red-Light Therapy": False, "Frequent Sauna": False, "Heavy Smoking": False, "Heavy Drinking": False},
    "Active Routine":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True, "Meditation": True,  "Red-Light Therapy": False, "Frequent Sauna": False, "Heavy Smoking": False, "Heavy Drinking": False},
    "Longevity Protocol":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True,  "Meditation": True,  "Red-Light Therapy": True, "Frequent Sauna": True, "Heavy Smoking": False, "Heavy Drinking": False},
    "Bad Idea Mode":      {"Consistent Sleep": False,  "Frequent Exercise": False,  "Mediterranean Diet": False,  "Meditation": False,  "Red-Light Therapy": False, "Frequent Sauna": False, "Heavy Smoking": True, "Heavy Drinking": True}
}

# ---------------- 2) Your Health Habits ----------------
st.sidebar.header("2) Your Health Habits")

# Hazard scaling type (applies to all habits)
scale_choice = st.sidebar.selectbox(
    "Risk Scaling Method",
    ["Log (recommended)", "Linear"],
    index=0,
    key="hazard_scale_mode",
    help="Chooses how health effects scale. Logarithmic compounds over time; Linear applies flat, partial-credit changes"
)
scaling_mode = "log" if "log" in scale_choice.lower() else "linear"
benefit_curvature = 1.0  # fixed; not exposed

def _apply_preset():
    p = st.session_state["preset"]  # new preset value

    # Helpful: set each checkbox and seed its adherence slider (percent units)
    for name in BENEFICIAL:
        key = CANON[name]  # e.g., "sleep", "exercise"
        on = bool(PRESET_TOGGLES[p].get(name, False))
        st.session_state[f"help_{key}_on"] = on
        # seed per-habit slider to 75% (or keep last if you prefer)
        st.session_state[f"help_{key}_adh"] = 75.0 if on else 0.0

    # Harmful: set toggles and exposures (sliders expect % values)
    st.session_state["smoke_on"] = bool(PRESET_TOGGLES[p].get("Heavy Smoking", False))
    st.session_state["drink_on"] = bool(PRESET_TOGGLES[p].get("Heavy Drinking", False))
    st.session_state["smoke_exposure"] = 100.0 if st.session_state["smoke_on"] else 0.0
    st.session_state["drink_exposure"] = 100.0 if st.session_state["drink_on"] else 0.0

    # Clear previous results so “Run Simulation” isn’t using stale outputs
    st.session_state.results = None

preset = st.sidebar.selectbox(
    "Preset Habits",
    ["None", "Core Routine", "Active Routine", "Longevity Protocol", "Bad Idea Mode"],
    index=0,
    key="preset",
    help="Optional: choose a plan to auto-fill your health habits below. You can edit them after",
    on_change=_apply_preset
)

# Split habits by effect
BENEFICIAL = [n for n, hr in BASE_RISK_MULT.items() if hr < 1.0]
HARMFUL    = [n for n, hr in BASE_RISK_MULT.items() if hr > 1.0]

toggles = {}

# ---- Helpful habits ----
st.sidebar.subheader("Helpful habits")

def helpful_block(label: str, key_prefix: str, default_on: bool):
    """Checkbox on one line; if enabled, show adherence slider directly below (percent)."""
    on = st.sidebar.checkbox(label, value=default_on, key=f"{key_prefix}_on")
    a = 0.0
    if on:
        a = pct_slider(
            "Adherence",
            value_pct=75.0, step_pct=5.0, digits=0,
            key=f"{key_prefix}_adh",
            help="How consistently you do this habit"
        )
    return on, a

# Canonical keys so UI labels, costs, and HR multipliers stay in sync
CANON = {
    "Consistent Sleep":         "sleep",
    "Frequent Exercise":        "exercise",
    "Frequent Sauna":           "sauna",
    "Mediterranean Diet":       "mediterraneandiet",
    "Meditation":               "meditation",
    "Red-Light Therapy":        "redlight",
    "Heavy Smoking":            "smoker",
    "Heavy Drinking":           "heavyalcohol",
    "Weight status":            "weight",
}

helpful_adh = {}  # canonical_key -> adherence in [0,1]
for name in BENEFICIAL:
    key = CANON[name]  # keep keys stable across UI/engine/costs
    default_on = PRESET_TOGGLES[preset].get(name, False)
    on, a = helpful_block(name, f"help_{key}", default_on)
    toggles[name] = on
    if on:
        helpful_adh[key] = a

# ---- Harmful habits ----
st.sidebar.subheader("Harmful habits")

def harmful_block(label: str, key_prefix: str):
    """Full-width checkbox on one line; if enabled, show its slider directly below."""
    on = st.sidebar.checkbox(
        label,
        value=PRESET_TOGGLES[preset].get(label, False),
        key=f"{key_prefix}_on"
    )
    exposure = 0.0
    if on:
        exposure = pct_slider(
            "Exposure",
            value_pct=100.0, step_pct=5.0, digits=0,
            key=f"{key_prefix}_exposure",
            help="Percentage of the time or dose you're exposed to"
        )
    return on, exposure

# Each harmful toggle sits on one line; slider appears directly underneath
smoke_on, smoke_exposure = harmful_block("Heavy Smoking",  "smoke")
drink_on, drink_exposure = harmful_block("Heavy Drinking", "drink")

# Keep these in toggles for the rest of the app
toggles["Heavy Smoking"]  = smoke_on
toggles["Heavy Drinking"] = drink_on

# Final multipliers to feed engine (Excel rule)

intervention_on = {}
lifestyle_HRs = {}
for name, base in BASE_RISK_MULT.items():
    key = CANON[name]
    on = bool(toggles[name])
    intervention_on[key] = on

    if not on or base == 1.0:
        m = 1.0
    elif base < 1.0:
        a = helpful_adh.get(key, 0.0)  # if off or not set, 0.0 gives m=1.0 in practice due to earlier 'on' gate
        m = _scale_hr(base, a, mode=scaling_mode)
    else:
        # Harmful exposure → scale by exposure on the log-hazard scale
        if key == "smoker":
            a = smoke_exposure
        elif key == "heavyalcohol":
            a = drink_exposure
        else:
            a = 1.0
        m = _scale_hr(base, a, mode=scaling_mode)

    lifestyle_HRs[key] = float(m)

# Add Weight status as its own factor (exact BMI only; off until BMI is valid)
if bmi_hr is not None:
    w_on = True
    w_mult = float(bmi_hr)
else:
    w_on = False
    w_mult = 1.0  # neutral until user enters data

lifestyle_HRs["weight"] = w_mult
intervention_on["weight"] = w_on

# ------------------ Sidebar: finance ------------------
st.sidebar.header("3) Your Finances")

# ------------------ Sidebar: habit costs (annual only) ------------------
with st.sidebar.expander("Health Habit Costs", expanded=False):
    st.caption("One field per habit: dollars per year (negative allowed if it saves you money). We apply healthcare inflation automatically each year.")

    # canonical keys used everywhere (from CANON above)
    ORDER = ["sleep", "exercise", "sauna", "mediterraneandiet", "meditation", "redlight", "smoker", "heavyalcohol", "weight"]
    LABEL_FOR = {v: k for k, v in CANON.items()}  # reverse map: key -> UI label

    # Defaults: previous one-time + recurring, converted to annual (keeps totals comparable)
    ANNUAL_DEFAULT = {
        "sleep":              50.0,         # ≈ $50/yr
        "exercise":           240.0,      # gym, etc.
        "sauna":              300.0,      # electricity/entry
        "mediterraneandiet":  5400.0,     # extra groceries vs baseline
        "meditation":         70.0,       # app subscription
        "redlight":           150.0,        # ≈ $157/yr
        "smoker":             800.0,      # cigarettes; make negative if quitting = savings
        "heavyalcohol":       1000.0,     # user-editable
        "weight":             0.0,         # no expense by default
    }

    annual_inputs = {}
    for key in ORDER:
        label = LABEL_FOR[key]
        if key == "weight":
            enabled = (weight_label != "None / Healthy Range")
        else:
            enabled = toggles[label]

        annual_inputs[key] = st.number_input(
            f"{label} — $ per year",
            value=float(ANNUAL_DEFAULT[key]),
            step=50.0,
            min_value=-1_000_000.0,  # allow negative for savings
            key=f"{key}_annual",
            disabled=not enabled,
        )

    # Feed the engine: horizon=1, one_time=0, recurring=annual
    intervention_costs = {
        k: IntervCost(horizon=1, one_time=0.0, recurring=float(annual_inputs[k]))
        for k in ORDER
    }

start_capital = st.sidebar.number_input("Starting Capital ($)", min_value=0, value=10_000, step=1_000)
di0 = st.sidebar.number_input("Yearly Spending Budget ($)", min_value=0, value=10_000, step=1_000)
ret = pct_slider("Portfolio Return", min_pct=0.0, max_pct=15.0,
                 value_pct=5.0, step_pct=0.1, digits=1, period="per year")
income_growth = pct_slider("Spending Growth", min_pct=0.0, max_pct=10.0,
                           value_pct=3.0, step_pct=0.1, digits=1, period="per year")
hc_infl = pct_slider("Healthcare Inflation", min_pct=0.0, max_pct=10.0,
                     value_pct=3.0, step_pct=0.1, digits=1, period="per year")

# ------------------ Sidebar: breakthroughs ------------------
st.sidebar.header("4) Future Treatments, Ranked from Tiers 1-3")
def tier_inputs(label, p0, g_pp, cap, cost, years_gained):
    st.sidebar.subheader(label)
    kid = label.lower().replace(" ", "_")
    c = st.sidebar.number_input("Estimated Cost Today ($)", min_value=0, value=cost, step=1_000, key=f"{kid}_cost")
    y = st.sidebar.number_input("Years Added (after purchase)", min_value=0.0, value=years_gained, step=0.1, key=f"{kid}_years")
    p = pct_slider("Annual Chance of Occurring", min_pct=0.0, max_pct=50.0,
               value_pct=p0*100.0, step_pct=0.5, digits=1, key=f"{kid}_p0", period="per year")
    g = pct_slider("Increase in Chance Per Missed Year", min_pct=0.0, max_pct=5.0,
               value_pct=g_pp*100.0, step_pct=0.05, digits=2, key=f"{kid}_gpp", period="per missed year")
    cap_ = pct_slider("Max Annual Chance", min_pct=0.0, max_pct=100.0,
                  value_pct=cap*100.0, step_pct=0.5, digits=1, key=f"{kid}_cap", period="per year")
    return Tier(
        cost_today=c,
        years_gain=y,
        base_prob=p,
        growth_per_year=g,
        cap_prob=cap_,
    )

tier1 = tier_inputs("Tier 1", p0=0.03,  g_pp=0.0015, cap=0.10, cost=12_000,     years_gained=0.7)
tier2 = tier_inputs("Tier 2", p0=0.006, g_pp=0.0015, cap=0.10, cost=550_000,   years_gained=2.5)
tier3 = tier_inputs("Tier 3", p0=0.0012,g_pp=0.0015, cap=0.10, cost=2_400_000, years_gained=7.0)

# ------------------ Sidebar: longevity params ------------------
st.sidebar.header("5) Longevity Parameters")
lambda_plateau = st.sidebar.number_input(
    "Late-age Risk Plateau λ",
    min_value=0.0, max_value=5.0, value=0.6, step=0.05,
    help="Only affects very late life. Sets a minimum death risk after the frontier age; larger λ increases it"
)
drift_days = st.sidebar.number_input("Frontier Age Drift (days per year)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
le_improve = pct_slider("Life Expectancy Growth", min_pct=0.0, max_pct=2.0,
                        value_pct=0.2, step_pct=0.1, digits=2, period="per year")
max_age_today = st.sidebar.number_input("Frontier Age Today", min_value=100.0, max_value=130.0, value=119.0, step=0.5)

# ------------------ Sidebar: simulation ------------------
st.sidebar.header("6) Simulation")
draws = st.sidebar.slider("Simulation Runs", 1000, 50000, 5000, step=1000)
seed = st.sidebar.number_input("Random Seed (reproducible)", min_value=0, max_value=1_000_000, value=49, step=1)

import hashlib, json

def _hash_inputs():
    # Pull BMI-related fields from session state (they exist even when empty)
    bmi_val = st.session_state.get("bmi")
    bmi_hr  = st.session_state.get("bmi_hr")
    units   = st.session_state.get("bmi_units")  # "US (ft/in, lb)" or "Metric (cm, kg)"
    # Raw text boxes, so a single keystroke clears stale results
    ft_s  = st.session_state.get("ht_ft")
    in_s  = st.session_state.get("ht_in")
    lb_s  = st.session_state.get("wt_lb")
    hcm_s = st.session_state.get("ht_cm")
    kg_s  = st.session_state.get("wt_kg")

    cfg = dict(
        age=int(age), sex=str(sex),
        scaling_mode=scaling_mode,
        preset=str(preset),
        toggles=toggles,
        helpful_adh=helpful_adh,
        adherence=1.0,

        # Harmful exposures
        smoke_on=bool(smoke_on), smoke_exposure=float(smoke_exposure),
        drink_on=bool(drink_on), drink_exposure=float(drink_exposure),

        # BMI inputs (both derived and raw so any edit clears results)
        bmi=bmi_val, bmi_hr=bmi_hr, bmi_units=units,
        ht_ft=ft_s, ht_in=in_s, wt_lb=lb_s, ht_cm=hcm_s, wt_kg=kg_s,

        # Legacy label kept for stability (always "Exact BMI" now)
        weight_label=str(weight_label),

        # Longevity + finance
        lambdaP=float(lambda_plateau), drift=float(drift_days),
        le_trend=float(le_improve), frontier=float(max_age_today),
        ret=float(ret), di0=float(di0), hc=float(hc_infl),

        # Tech tiers
        tier1=dict(cost=tier1.cost_today, years=tier1.years_gain, p=tier1.base_prob,
                   g=tier1.growth_per_year, cap=tier1.cap_prob),
        tier2=dict(cost=tier2.cost_today, years=tier2.years_gain, p=tier2.base_prob,
                   g=tier2.growth_per_year, cap=tier2.cap_prob),
        tier3=dict(cost=tier3.cost_today, years=tier3.years_gain, p=tier3.base_prob,
                   g=tier3.growth_per_year, cap=tier3.cap_prob),
    )
    s = json.dumps(cfg, sort_keys=True, default=float)
    return hashlib.sha256(s.encode()).hexdigest()

new_sig = _hash_inputs()
if "input_sig" not in st.session_state:
    st.session_state.input_sig = new_sig
elif st.session_state.input_sig != new_sig:
    # inputs changed since last run → clear stale results
    st.session_state.input_sig = new_sig
    st.session_state.results = None

# Sidebar Run Simulation styling: full width, custom colors, bold label
st.markdown("""
<style>
/* Scope to sidebar so main-pane buttons stay default */
[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  background-color: #cddae9 !important;  /* fill */
  color: #09427d !important;             /* label text */
  border: 1px solid #cddae9 !important;
  border-radius: 12px !important;
}

/* hover + active states */
[data-testid="stSidebar"] .stButton > button:hover {
  background-color: #dbe6f2 !important;
  border-color: #dbe6f2 !important;
  color: #072f59 !important;
}
[data-testid="stSidebar"] .stButton > button:active {
  background-color: #bfd0e3 !important;
  border-color: #bfd0e3 !important;
  color: #072f59 !important;
}
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("Run Simulation", type="primary", use_container_width=True):
    inputs = Inputs(
        start_age=int(age),
        sex=sex,
        draws=int(draws),
        investment_return=float(ret),
        start_capital=float(start_capital),
        # Excel-parity finance
        discretionary_income=float(di0),
        income_growth=float(income_growth),
        annual_contrib=0.0,                  # ignored when discretionary_income is provided
        contrib_growth=0.0,
        lambdaP=float(lambda_plateau),
        frontier_drift_days=float(drift_days),
        le_trend=float(le_improve),
        max_age_today=int(max_age_today),
        hc_inflation=float(hc_infl),
        lifestyle_HRs=lifestyle_HRs,
        adherence=1.0,
        tiers=[tier1, tier2, tier3],
        tier_repeatable=True,                 # repeatable breakthroughs
        intervention_costs=intervention_costs,
        intervention_on=intervention_on,
        grid_max_age=max(int(age) + 126, 170),
        seed=int(seed),
    )
    st.session_state.results = run_monte_carlo(inputs)

# ------------------ Render results (if any) ------------------
out = st.session_state.results

if out is None:
    # show drawer even before a run so admins can open past snapshots
    saved_runs_drawer("Saved runs")
    st.info("Describe yourself on the left, then click **Run Simulation**")
    st.stop()
else:
    # show drawer near the top of the results view too
    saved_runs_drawer("Saved runs")
    # --- Helpers to package a snapshot for storage (define BEFORE using them) ---
    def _summarize_for_save(out, age) -> dict:
        import numpy as np
        med_life = float(np.median(out["projected_life"]))
        p5, p95 = np.percentile(out["projected_life"], [5, 95])
        med_net = float(np.median(out["net_worth"]))
        return {"median_life": med_life, "range_90": [float(p5), float(p95)], "median_net_mm": med_net}

    def _fig_payload(out) -> dict:
        """
        Trim the big simulation result to only what the client page needs.
        Keeps names stable and converts numpy to JSON-safe types.
        Also thins very large (D,T) arrays so snapshots don't balloon.
        """
        import numpy as np

        wanted = [
            # axes + life
            "chrono_age", "projected_life", "projected_life_frac", "projected_life_mc",
            # wealth + cashflows
            "net_worth", "balance_path", "balance_no_tech_path",
            "discretionary_income_by_year", "health_spend_by_year", "contrib_by_year",
            # survival + tech
            "yrs_added_interventions", "threshold_series", "le_threshold_series",
            "bio_age", "tech_years_by_age", "tech_costs_by_age",
        ]

        payload = {}
        for name in wanted:
            if name in out:
                payload[name] = out[name]

        # Thin very large 2D arrays so saving to JSON stays reasonable
        MAX_DRAWS = 2000

        def _thin_2d(a):
            a = np.asarray(a)
            if a.ndim == 2 and a.shape[0] > MAX_DRAWS:
                idx = np.linspace(0, a.shape[0] - 1, MAX_DRAWS).astype(int)
                return a[idx]
            # Also thin 1D arrays (like projected_life) to match!
            if a.ndim == 1 and a.size > MAX_DRAWS:
                idx = np.linspace(0, a.size - 1, MAX_DRAWS).astype(int)
                return a[idx]
            return a
        
        # UPDATED LIST: Added "net_worth" to ensure x and y axes match
        keys_to_thin = [
            "balance_path", 
            "balance_no_tech_path", 
            "bio_age", 
            "tech_years_by_age", 
            "tech_costs_by_age", 
            "projected_life_mc",
            "projected_life",
            "projected_life_frac",
            "net_worth"  # <--- NEW: MUST BE HERE
        ]
        
        for name in keys_to_thin:
            if name in payload:
                payload[name] = _thin_2d(payload[name])

        # Prefer the canonical 1-D threshold series key if both exist
        if "threshold_series" not in payload and "le_threshold_series" in payload:
            payload["threshold_series"] = payload.pop("le_threshold_series")

        return _to_jsonable(payload)

    # =================== Client (Operator snapshot controls) ===================
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.subheader("Client")

    sess = Session()
    org_id = st.session_state["org_id"]               # set by require_auth()
    user_id = st.session_state["user"]["id"]          # set by require_auth()

    # List clients in this org
    clients = (sess.query(Client)
            .filter(Client.org_id == org_id)
            .order_by(Client.created_at.desc())
            .all())

    choices = ["+ New client"] + [f"{c.name} · {c.email or ''}" for c in clients]
    pick = st.selectbox("Who is this run for?", choices, index=0)

    new_name = new_email = None
    if pick == "+ New client":
        c1, c2 = st.columns(2)
        new_name = c1.text_input("Client name")
        new_email = c2.text_input("Client email (optional)")

    if st.button("Save run for client", type="secondary"):
        # 1) resolve/create client inside the button block
        if pick == "+ New client":
            if not new_name:
                st.warning("Client name required")
                st.stop()
            client = Client(org_id=org_id, name=new_name.strip(), email=(new_email or "").strip())
            sess.add(client)
            sess.commit()
        else:
            client = clients[choices.index(pick) - 1]

        # 2) Inputs (JSON-safe via model_dump + _to_jsonable)
        inputs_json = RunInputs(
            age=int(age), sex=sex, draws=int(draws),
            ret=float(ret), start_capital=float(start_capital),
            di0=float(di0), income_growth=float(income_growth), hc_infl=float(hc_infl),
            lambdaP=float(lambda_plateau), drift_days=float(drift_days),
            le_trend=float(le_improve), max_age_today=int(max_age_today),
            lifestyle_HRs=lifestyle_HRs, intervention_on=intervention_on,
            tier1=dict(cost=tier1.cost_today, years=tier1.years_gain, p=tier1.base_prob, g=tier1.growth_per_year, cap=tier1.cap_prob),
            tier2=dict(cost=tier2.cost_today, years=tier2.years_gain, p=tier2.base_prob, g=tier2.growth_per_year, cap=tier2.cap_prob),
            tier3=dict(cost=tier3.cost_today, years=tier3.years_gain, p=tier3.base_prob, g=tier3.growth_per_year, cap=tier3.cap_prob),
            seed=int(seed)
        )
        inputs_json = _to_jsonable(inputs_json.model_dump() if hasattr(inputs_json, "model_dump") else inputs_json.dict())

        # 3) Outputs payload
        outputs_json = RunOutputs(
            summary=_summarize_for_save(out, age),
            figs_data=_fig_payload(out)
        )
        outputs_json = _to_jsonable(outputs_json.model_dump() if hasattr(outputs_json, "model_dump") else outputs_json.dict())

        # 4) Save snapshot with org + operator
        r = Run(
            org_id=org_id,
            operator_id=user_id,     # ← match storage.py column name
            client_id=client.id,
            inputs=inputs_json,
            outputs=outputs_json,
        )
        sess.add(r); sess.commit()

        # 5) Signed link (label-based, survives file renames)
        link = _client_link_for(r.id)

        st.success("Saved snapshot.")
        st.markdown("**Client link**")
        st.code(link, language="text")
        st.markdown(f"[Open client view]({link})")

    # Tiny spacer before the overview
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ------------------ Results Overview ------------------

    st.subheader("Results Overview")
    st.caption("How long you're expected to live, what you'll be worth, and where your extra years come from")

    # 1. SIMULATION VIEW TOGGLE (Moved to top)
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

    # 2. CALCULATE DYNAMIC METRICS
    # Determine which data powers the headline metric
    if "Range" in hist_mode:
        current_life_data = out["projected_life_mc"]
        metric_label = "Simulated Lifespan (Median outcome)"
    else:
        current_life_data = out.get("projected_life_frac", out["projected_life"])
        metric_label = "Projected Lifespan (Biological)"

    median_life = float(np.median(current_life_data))
    p5, p95 = np.percentile(out["projected_life"], [5,95])
    median_net = float(np.median(out["net_worth"]))

    # --- Calculate Static "Benefit" Metrics ---
    yrs_from_habits = float(np.sum(out["yrs_added_interventions"]))

    # Alive-weighted expected years from future treatments
    le_series = out.get("threshold_series")
    if le_series is None:
        le_series = out.get("le_threshold_series")
    
    alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)
    exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)
    yrs_from_tech = float(np.sum(exp_tech_by_age))

    # 3. RENDER METRICS (Aligned with Client View)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(metric_label, f"{median_life:.0f} years")
        # Dynamic years remaining based on the toggle result
        years_remaining = int(round(median_life - age))
        st.metric("Years remaining", f"{years_remaining:d} years")

    with col2:
        st.metric("Net worth (median, $MM)", f"{median_net:,.2f}")
        st.metric(
            "Benefit from Habits",
            f"+{yrs_from_habits:.1f} years",
            help="The expected extra time your habits add to your life, regardless of luck."
        )

    with col3:
        st.metric(
            "90% Confidence Range",
            f"{p5:.0f}-{p95:.0f} years",
            help="Middle 90% of simulated lifespans; 5% of draws fall below the left value and 5% above the right"
        )
        st.metric(
            "Benefit from Treatments",
            f"+{yrs_from_tech:.1f} years",
            help="The expected extra time future treatments add, accounting for their arrival probability."
        )

# ------------------ Health Spending Outcomes ------------------

# tiny vertical spacer between the two metric groups
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

st.subheader("Spending Outcomes")
st.caption("What your health purchases cost, how many years they buy, and what's left for your estate")

# 1) Pull arrays safely
age_grid      = np.asarray(out["chrono_age"], dtype=float)                  # (T,)
bal_with      = np.asarray(out["balance_path"], dtype=float)                # (D, T) in $
bal_without   = np.asarray(out["balance_no_tech_path"], dtype=float)        # (D, T) in $
path_with     = bal_with / 1e6                                              # $MM for plotting
path_without  = bal_without / 1e6

def _first_present(d, *keys):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

thr = _first_present(out, "threshold_series", "le_threshold_series")
if thr is None:
    raise KeyError("Missing threshold series: expected 'threshold_series' or 'le_threshold_series'.")
thr = np.asarray(thr, dtype=float).ravel()

bio_age           = np.asarray(out["bio_age"], dtype=float)                 # (D, T)
tech_years_by_age = np.asarray(out["tech_years_by_age"], dtype=float)       # (D, T)

# 2) Alive masks (baseline = WITHOUT; treatments reduce biological age)
cum_tech    = np.cumsum(tech_years_by_age, axis=1)
bio_with    = np.maximum(0.0, bio_age - cum_tech)                            # WITH treatments
alive_with  = (bio_with  < thr[None, :])
alive_without = (bio_age < thr[None, :])                                     # WITHOUT = baseline

# 3) Scenario-specific median death ages (first age with <=50% alive)
def _median_death_age(alive_mask):
    surv = alive_mask.mean(axis=0)  # proportion alive by age
    idx  = np.where(surv <= 0.5)[0]
    return float(age_grid[idx[0]]) if idx.size else float(age_grid[-1])

med_age_with    = _median_death_age(alive_with)
med_age_without = _median_death_age(alive_without)

mask_with    = (age_grid <= med_age_with)
mask_without = (age_grid <= med_age_without)

# 4) Alive-weighted medians for portfolio paths (center lines)
def _median_alive(path, alive_mask):
    D, T = path.shape
    med = np.full(T, np.nan)
    for t in range(T):
        vals = path[alive_mask[:, t], t]
        if vals.size:
            med[t] = np.median(vals)
    return med

m_with    = _median_alive(path_with,    alive_with)
m_without = _median_alive(path_without, alive_without)

# ----- KPI foundations: compute per-draw totals up to WITH median age -----
# window mask
m = mask_with  # ages <= WITH median death age

# Safety arrays
tech_spend = out.get("tech_costs_by_age")  # (D, T) dollars
tech_spend = np.asarray(tech_spend, dtype=float) if tech_spend is not None else None
tyba = np.asarray(out["tech_years_by_age"], dtype=float)                  # (D, T)
alive_w = alive_with                                                       # (D, T)

# Per-draw totals (undiscounted) to WITH median age
if tech_spend is None:
    spend_by_draw = np.zeros(bio_age.shape[0], dtype=float)
else:
    spend_by_draw = (tech_spend[:, m] * alive_w[:, m]).sum(axis=1)        # $ per draw

yrs_by_draw = (tyba[:, m] * alive_w[:, m]).sum(axis=1)                    # yrs per draw

# Typical (median) and expected (mean) totals
typ_cost_total   = float(np.median(spend_by_draw))                         # headline
exp_cost_total   = float(spend_by_draw.mean())                             # optional in caption

# Present value (so the dollars mean something)
# Use user's portfolio return if available, else 3% real
_r = float(locals().get("ret", 0.03))
years = age_grid[m] - age_grid[m][0]
df = (1.0 / (1.0 + _r)) ** years
if tech_spend is None:
    pv_cost_median = 0.0
else:
    pv_by_draw = (tech_spend[:, m] * alive_w[:, m] * df[None, :]).sum(axis=1)
    pv_cost_median = float(np.median(pv_by_draw))

# ROI as a distribution, then report median
roi_draw = np.full_like(yrs_by_draw, np.nan, dtype=float)
nz = spend_by_draw > 0
roi_draw[nz] = yrs_by_draw[nz] / (spend_by_draw[nz] / 100000.0)
roi_median = float(np.nanmedian(roi_draw))                                 # headline ROI

# ----- Bequest at death (medians by scenario, then delta) -----
def _terminal_wealth_at_death(bal, alive_mask):
    D, T = bal.shape
    tw = np.zeros(D, dtype=float)
    for d in range(D):
        idx = np.where(alive_mask[d])[0]
        t = idx[-1] if idx.size else 0
        tw[d] = bal[d, t]
    return tw

tw_with    = _terminal_wealth_at_death(bal_with,    alive_with)
tw_without = _terminal_wealth_at_death(bal_without, alive_without)

tw_with_med    = float(np.median(tw_with))
tw_without_med = float(np.median(tw_without))
bequest_delta_med = tw_with_med - tw_without_med

# 6) KPI row (1 x 3): Spend | ROI | Bequest
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Expected treatment costs", f"${typ_cost_total:,.0f}")

with c2:
    st.metric("Years gained for every $100k", f"{roi_median:.2f} yrs")

def _fmt_signed_currency(x):
    sign = "+" if x > 0 else ""  # minus sign will come from format itself
    return f"{sign}${abs(x):,.0f}" if x < 0 else f"{sign}${x:,.0f}"

with c3:
    st.metric("Money you leave behind", _fmt_signed_currency(bequest_delta_med))

# ---------- Impact analysis (local counterfactual) ----------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.expander("What helped or hurt most?", expanded=False):
    st.caption("We re-run your profile once per factor, setting that factor to **baseline (risk ×1.00)**, and report how your " \
    "**median lifespan** would change. “Risk ×1.10” means 10% higher yearly death risk; “×0.90” means 10% lower.")

    # Keep runs fast but stable
    draws_impact = int(min(draws, 4000))  # smaller than main run, same seed for low noise

    # Helper to rebuild Inputs with a different lifestyle_HRs dict
    def _inputs_with(lhr: dict) -> Inputs:
        return Inputs(
            start_age=int(age),
            sex=sex,
            draws=draws_impact,
            investment_return=float(ret),
            start_capital=float(start_capital),
            discretionary_income=float(di0),
            income_growth=float(income_growth),
            annual_contrib=0.0,
            contrib_growth=0.0,
            lambdaP=float(lambda_plateau),
            frontier_drift_days=float(drift_days),
            le_trend=float(le_improve),
            max_age_today=int(max_age_today),
            hc_inflation=float(hc_infl),
            lifestyle_HRs=lhr,
            adherence=1.0,
            tiers=[tier1, tier2, tier3],
            tier_repeatable=True,
            intervention_costs=intervention_costs,
            intervention_on=intervention_on,
            grid_max_age=max(int(age) + 126, 170),
            seed=int(seed),  # same seed → differences reflect the factor
        )

    # Build label map once
    LABEL_FOR = {v: k for k, v in CANON.items()}

    rows = []
    for key, hr_now in lifestyle_HRs.items():
        # Skip neutral factors and disabled ones
        if not intervention_on.get(key, False):
            continue
        if abs(float(hr_now) - 1.0) < 1e-9:
            continue

        lhr2 = dict(lifestyle_HRs)
        lhr2[key] = 1.0  # neutralize this factor only

        out_i = run_monte_carlo(_inputs_with(lhr2))
        med_i = float(np.median(out_i["projected_life"]))

        # Effect if we removed the factor: positive = it’s hurting you now; negative = it’s helping you now.
        effect = med_i - median_life

        rows.append({
            "Factor": LABEL_FOR.get(key, key).replace("Weight status", "BMI"),
            "Risk ×": f"×{float(hr_now):.2f}",
            "Δ Median years if removed": effect
        })

    if not rows:
        st.info("No active factors to analyze yet")
    else:
        df_imp = pd.DataFrame(rows)

        # Rank lists
        bad  = df_imp.sort_values("Δ Median years if removed", ascending=False).head(5)
        good = df_imp.sort_values("Δ Median years if removed", ascending=True).head(5)

        # Drop the pandas row index so the leftmost 0–4 vanishes
        bad  = bad.reset_index(drop=True)
        good = good.reset_index(drop=True)

        # Lock the visible column order (optional but nice)
        cols = ["Factor", "Risk ×", "Δ Median years if removed"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Biggest drag right now**")
            st.dataframe(
                bad[cols],
                use_container_width=True,
                hide_index=True
            )
        with c2:
            st.markdown("**Biggest boost right now**")
            st.dataframe(
                good[cols],
                use_container_width=True,
                hide_index=True
            )
    
# ================== Lifespan + Wealth controls (row 1) ==================
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# We only need the Wealth View toggle here now, 
# because "Simulation View" (hist_mode) is already set at the top of the page.
view = st.radio("Wealth View", ["Scatter", "Heatmap"], index=0, horizontal=True)

# Tiny spacer so charts don’t collide with radios
st.write("")

# ================== Prep data for both charts ==================
# Left chart data
if "Range" in hist_mode:  # Check the variable from the top of the page
    life_data = out["projected_life_mc"]
    life_title = "Simulated Ages at Death"
    nbins = int(np.ptp(life_data)) + 1 if np.ptp(life_data) >= 1 else 10
    fig_life = px.histogram(
        life_data, nbins=nbins, title=life_title,
        labels={"value": "Age at death (years)", "Count": "Simulations"}
    )
else:
    life_data = out.get("projected_life_frac", out["projected_life"])
    life_title = "Predicted Lifespan"
    fig_life = px.histogram(
        life_data, title=life_title,
        labels={"value": "Age at death (years)", "Count": "Simulations"}
    )
    fig_life.update_traces(xbins=dict(size=0.5))

# ... rest of the chart code remains the same ...
# Just make sure line 700 uses the new check string:
# if "Range" in hist_mode:

x_lo, x_hi = _axis_from_data(life_data)
fig_life.update_xaxes(range=[x_lo, x_hi])
fig_life.update_layout(
    title=dict(text=life_title, x=0, xanchor="left", font=dict(size=16, color="#444")),
    margin=dict(t=40, r=0, l=0, b=0),
    height=420,
    showlegend=False
)

# --- Right chart data (scatter/heatmap) ---
# Use stochastic lifespans for x when dice mode is selected; otherwise deterministic threshold life.
life_for_net = out["projected_life_mc"] if hist_mode == "Random chance each year" \
               else out.get("projected_life_frac", out["projected_life"])

if hist_mode == "Random chance each year":
    # align net worth to the MC death year
    idx = np.clip((life_for_net - age).astype(int), 0, out["balance_path"].shape[1] - 1)
    nw = out["balance_path"][np.arange(idx.size), idx] / 1e6  # scale to $MM
else:
    # engine already computed threshold-mode net worth at death
    nw = out["net_worth"]

# Count DISTINCT tech start events per draw (not years active).
ty = out["tech_years_by_age"]  # shape (D, T); 0 before any tech starts, >0 once something is active
# A start is when we go from 0 last year to >0 this year.
starts = (ty > 0) & np.concatenate([np.ones((ty.shape[0], 1), dtype=bool), ty[:, :-1] == 0], axis=1)
purchases = starts.sum(axis=1)

# Bucket for clearer legend
bucket = np.where(purchases == 0, "0",
          np.where(purchases == 1, "1",
          np.where(purchases <= 3, "2-3", "4+")))

df_nw = pd.DataFrame({"Life": life_for_net, "NetWorth": nw, "Purchases": bucket})

if view == "Scatter":
    fig_nw = px.scatter(
        df_nw, x="Life", y="NetWorth", color="Purchases",
        title="Simulated Wealth Outcomes by Lifespan",
        labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"},
        opacity=0.55
    )
else:
    fig_nw = px.density_heatmap(
        df_nw, x="Life", y="NetWorth",
        nbinsx=40, nbinsy=40, color_continuous_scale="Blues",
        title="Simulated Wealth Outcomes by Lifespan",
        labels={"Life": "Age at death (years)", "NetWorth": "Net worth at death ($MM)"}
    )
x_lo2, x_hi2 = _axis_from_data(df_nw["Life"])
y_lo2, y_hi2 = _axis_from_data(df_nw["NetWorth"], pad_frac=0.04, min_span=0.2)
fig_nw.update_xaxes(range=[x_lo2, x_hi2])
fig_nw.update_yaxes(range=[y_lo2, y_hi2])
title_text = "Simulated Wealth Outcomes by Lifespan" if view == "Heatmap" else "Simulated Wealth Outcomes by Lifespan"
fig_nw.update_layout(title=dict(text=title_text, x=0, xanchor="left", font=dict(size=16, color="#444")),
                     margin=dict(t=40, r=0, l=0, b=0), height=420)

# ================== Charts (row 2, perfectly aligned) ==================
col_l, col_r = st.columns(2)

with col_l:
    st.plotly_chart(fig_life, use_container_width=True)

with col_r:
    st.plotly_chart(fig_nw, use_container_width=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# Years-added chart: Excel-match mode by default with options
mode = st.selectbox(
    "Added Years Breakdown",
    ["From Health Habits", "From Habits + Treatments (survivors only)", "From Future Treatments (survivors only)"],
    index=0,
    help="Shows how different health habits and treatments contribute to extra years of life. Values " \
    "are weighted only for people still alive in each simulated year"
)

yrs_int = out["yrs_added_interventions"]                           # (T,)

# Alive-weighted expected tech by age
# Use the 1-D threshold series returned by the engine
le_series = out.get("threshold_series")  # shape (T,)
# Back-compat if you ever run an older engine that used a different name
if le_series is None:
    le_series = out.get("le_threshold_series")

alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)  # (D, T)
exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)

if mode.startswith("From Health Habits"):
    y = yrs_int
elif mode.startswith("From Habits + Treatments (survivors only)"):
    y = yrs_int + exp_tech_by_age
else:  # Tech only
    y = exp_tech_by_age

fig_yrs = px.area(x=out["chrono_age"], y=y,
                  labels={"x": "Age", "y": "Years Added by Age"},
                  title=mode)
st.plotly_chart(fig_yrs, use_container_width=True)

# Optional diagnostic
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.expander("Health vs Life Expectancy", expanded=False):
    mean_bio = out["bio_age"].mean(axis=0)        # (T,)

    le_series = out.get("threshold_series")
    if le_series is None:
        le_series = out.get("le_threshold_series")

    # sanity: make sure lengths match your age axis
    assert len(le_series) == len(out["chrono_age"]), "LE series length mismatch"

    df = pd.DataFrame({
        "Age": out["chrono_age"],
        "Biological Age (mean)": mean_bio,
        "Life Expectancy": le_series,
    })

    fig_diag = px.line(
        df,
        x="Age",
        y=["Biological Age (mean)", "Life Expectancy"],
        title="Biological Age vs Societal Life Expectancy",
        labels={"Age": "Age (years)", "value": "Years", "variable": ""},
    )
    fig_diag.update_layout(legend_title_text="")
    st.plotly_chart(fig_diag, use_container_width=True)

# Finance diagnostics
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.expander("Personal Finances, First 20 Years"):
    df_fin = pd.DataFrame({
        "Age": out["chrono_age"],
        "Discretionary Income": out["discretionary_income_by_year"],
        "Total Healthcare Spending": out["health_spend_by_year"],
        "Excess Cash for Investments": out["contrib_by_year"],
    })
    st.dataframe(df_fin.head(20), use_container_width=True, height=480)

# 7) Portfolio overlay (optional, after KPIs). Trim at each scenario's own median age.
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

x_end = max(med_age_with, med_age_without)
fig_bal = go.Figure()
fig_bal.add_trace(go.Scatter(x=age_grid[mask_with],    y=m_with[mask_with],       mode="lines", name="With treatments"))
fig_bal.add_trace(go.Scatter(x=age_grid[mask_without], y=m_without[mask_without], mode="lines", name="Without treatments"))
fig_bal.update_layout(title="Portfolio value by age", xaxis_title="Age (years)", yaxis_title="Balance ($MM)")
fig_bal.update_xaxes(range=[float(age_grid[0]), x_end])
st.plotly_chart(fig_bal, use_container_width=True)

# 8) Cash-flow bars (optional)
contrib = np.asarray(out["contrib_by_year"], dtype=float)                # (T,)
spend_mean = np.zeros_like(contrib)
if tech_spend is not None:
    spend_mean = (tech_spend * alive_with).mean(axis=0)
    # optional smoothing, comment out if you want raw events
    spend_mean = pd.Series(spend_mean).rolling(3, center=True, min_periods=1).mean().to_numpy()

m = mask_with
df_cf = pd.DataFrame({
    "Age": age_grid[m],
    "Contributions (DI - health)": contrib[m],
    "Expected treatment spend": -spend_mean[m],
})
fig_cf = px.bar(df_cf, x="Age",
                y=["Contributions (DI - health)", "Expected treatment spend"],
                barmode="relative",
                title="Cash flows into portfolio (excluding market returns)",
                labels={"value": "$ per year"})
fig_cf.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
st.plotly_chart(fig_cf, use_container_width=True)
