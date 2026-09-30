from __future__ import annotations
import math, os, urllib.request, json
from typing import Dict, List, Optional, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import defer

import engine
from engine import Inputs, Tier, IntervCost
import storage
from storage import Session, Org, User, Client, Run

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://povsczqljpbvermijxrh.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "sb_publishable_j-YL3zOHlJhPvNSsa1lAtw_22wMlTnt")

app = FastAPI(
    title="Hazard Curve API",
    description="Longevity & Wealth Planning Simulation Engine",
    version="2.0.0",
    docs_url=None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard habit base hazard ratios from Admin.py
BASE_RISK_MULT: Dict[str, float] = {
    # Hemodynamic
    "Frequent Exercise": 0.68,
    "Daily Movement": 0.72,
    "Frequent Sauna": 0.77,

    # Metabolic
    "Caloric Restriction": 0.75,
    "Mediterranean Diet": 0.77,
    "Low Sugar / Fiber": 0.85,

    # Restorative
    "Consistent Sleep": 0.88,
    "Meditation": 0.93,

    # Cellular
    "Cold Plunge": 0.96,
    "Red-Light Therapy": 0.98,
    "Supplements": 0.95,

    # Chemical (Harmful)
    "Heavy Smoking": 2.50,
    "Heavy Drinking": 1.35,
    "Poor Air Quality": 1.10,
}

CANON: Dict[str, str] = {
    "Frequent Exercise": "exercise",
    "Daily Movement": "steps",
    "Frequent Sauna": "sauna",
    "Caloric Restriction": "fasting",
    "Mediterranean Diet": "mediterraneandiet",
    "Low Sugar / Fiber": "sugar",
    "Consistent Sleep": "sleep",
    "Meditation": "meditation",
    "Cold Plunge": "cold",
    "Red-Light Therapy": "redlight",
    "Supplements": "supplements",
    "Heavy Smoking": "smoker",
    "Heavy Drinking": "heavyalcohol",
    "Poor Air Quality": "airquality",
    "Weight status": "weight"
}

BUCKETS = {
    "hemo": {
        "label": "Cardiovascular",
        "type": "beneficial",
        "habits": ["Frequent Exercise", "Daily Movement", "Frequent Sauna"]
    },
    "meta": {
        "label": "Metabolic Health",
        "type": "beneficial",
        "habits": ["Caloric Restriction", "Mediterranean Diet", "Low Sugar / Fiber"]
    },
    "rest": {
        "label": "Recovery & Neuro",
        "type": "beneficial",
        "habits": ["Consistent Sleep", "Meditation"]
    },
    "cell": {
        "label": "Cellular Repair",
        "type": "beneficial",
        "habits": ["Cold Plunge", "Red-Light Therapy", "Supplements"]
    },
    "chem": {
        "label": "Toxic Exposure",
        "type": "harmful",
        "habits": ["Heavy Smoking", "Heavy Drinking", "Poor Air Quality"]
    }
}

ORDER = [
    "exercise", "steps", "sauna", 
    "fasting", "mediterraneandiet", "sugar",
    "sleep", "meditation",
    "cold", "redlight", "supplements",
    "smoker", "heavyalcohol", "airquality", "weight"
]

ANNUAL_DEFAULT = {
    "exercise": 240.0, "steps": 0.0, "sauna": 300.0,
    "fasting": -1500.0, "mediterraneandiet": 5400.0, "sugar": 0.0,
    "sleep": 50.0, "meditation": 70.0,
    "cold": 100.0, "redlight": 150.0, "supplements": 600.0,
    "smoker": 800.0, "heavyalcohol": 1000.0, "airquality": 0.0,
    "weight": 0.0
}

# BMI anchors matching Admin.py
BMI_HR_ANCHORS = [
    (15.0, 2.76), (18.5, 1.13),
    (22.5, 1.00),
    (25.0, 1.07), (27.5, 1.20),
    (30.0, 1.45), (35.0, 1.94), (40.0, 2.76)
]
BMI_EVIDENCE_MIN = 15.0
BMI_EVIDENCE_MAX = 40.0
BMI_VALID_MIN = 12.0
BMI_VALID_MAX = 60.0
BMI_HR_CAP = 3.0

def hr_bmi_continuous(bmi: float):
    if not (BMI_VALID_MIN <= bmi <= BMI_VALID_MAX):
        return None, "invalid"
    xs = np.array([x for x, _ in BMI_HR_ANCHORS], dtype=float)
    ys = np.log(np.array([y for _, y in BMI_HR_ANCHORS], dtype=float))
    if bmi <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y = ys[0] + slope * (bmi - xs[0])
    elif bmi >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        y = ys[-1] + slope * (bmi - xs[-1])
    else:
        y = float(np.interp(bmi, xs, ys))
    hr = float(np.exp(y))
    hr = min(hr, BMI_HR_CAP)
    flag = "valid" if (BMI_EVIDENCE_MIN <= bmi <= BMI_EVIDENCE_MAX) else "extrapolated"
    return hr, flag

def _scale_hr(base_hr: float, adherence: float, mode: str = "log") -> float:
    a = max(0.0, min(1.0, float(adherence)))
    if mode == "linear":
        return 1.0 - a * (1.0 - float(base_hr))
    eff = a ** 1.0
    return float(base_hr) ** eff

class HabitInputItem(BaseModel):
    enabled: bool = False
    val: float = 0.75  # 0 to 1

class TechTierPayload(BaseModel):
    cost: float
    years: float
    p0: float
    g_pp: float
    cap: float

class FullSimulationRequest(BaseModel):
    # Demographics
    age: int = Field(30, ge=18, le=95)
    sex: str = Field("Male", pattern="^(Male|Female)$")
    
    # Body Mass
    bmi: Optional[float] = None
    
    # Scaling mode
    scaling_mode: str = "log"
    
    # Habit states keyed by habit name (e.g. "Frequent Exercise")
    habits: Dict[str, HabitInputItem] = Field(default_factory=dict)
    lifestyle_costs: Dict[str, float] = Field(default_factory=dict)
    
    # Financials
    start_capital: float = 10000.0
    di0: float = 10000.0
    ret: float = 0.05
    income_growth: float = 0.03
    hc_infl: float = 0.03
    
    # Breakthroughs
    use_tech: bool = True
    tier1: TechTierPayload = Field(default_factory=lambda: TechTierPayload(cost=12000, years=0.7, p0=0.03, g_pp=0.0015, cap=0.10))
    tier2: TechTierPayload = Field(default_factory=lambda: TechTierPayload(cost=550000, years=2.5, p0=0.006, g_pp=0.0015, cap=0.10))
    tier3: TechTierPayload = Field(default_factory=lambda: TechTierPayload(cost=2400000, years=7.0, p0=0.0012, g_pp=0.0015, cap=0.10))
    
    # Lifespan
    lambda_plateau: float = 0.6
    drift_days: float = 15.0
    le_improve: float = 0.002
    max_age_today: float = 119.0
    
    # Simulation
    draws: int = Field(5000, ge=1000, le=50000)
    seed: int = 49

@app.post("/api/simulate_full")
def simulate_full(req: FullSimulationRequest):
    try:
        # 1. Winning hand & harmful multiplier resolution exactly from Admin.py lines 640-675
        active_bucket_habits = {bid: [] for bid in BUCKETS}
        
        for bid, bdata in BUCKETS.items():
            for name in bdata["habits"]:
                h_item = req.habits.get(name)
                if h_item and h_item.enabled:
                    base_hr = BASE_RISK_MULT[name]
                    eff_hr = _scale_hr(base_hr, h_item.val, mode=req.scaling_mode)
                    active_bucket_habits[bid].append({
                        "name": name,
                        "key": CANON[name],
                        "base_hr": base_hr,
                        "effective_hr": eff_hr,
                        "val": h_item.val
                    })

        lifestyle_HRs = {}
        intervention_on = {}

        bucket_summaries = {}
        for bid, bdata in BUCKETS.items():
            active_items = active_bucket_habits[bid]
            if not active_items:
                bucket_summaries[bid] = {"impact": 1.0, "driver": "None", "count": 0}
                continue

            if bdata["type"] == "beneficial":
                winner = min(active_items, key=lambda x: x["effective_hr"])
                lifestyle_HRs[winner["key"]] = winner["effective_hr"]
                intervention_on[winner["key"]] = True
                bucket_summaries[bid] = {
                    "impact": round(winner["effective_hr"], 2),
                    "driver": winner["name"],
                    "count": len(active_items)
                }
                for item in active_items:
                    if item["key"] != winner["key"]:
                        intervention_on[item["key"]] = False
                        lifestyle_HRs[item["key"]] = 1.0
            else:
                combined_hr = 1.0
                for item in active_items:
                    lifestyle_HRs[item["key"]] = item["effective_hr"]
                    intervention_on[item["key"]] = True
                    combined_hr *= item["effective_hr"]
                bucket_summaries[bid] = {
                    "impact": round(combined_hr, 2),
                    "driver": "Stacked Exposures",
                    "count": len(active_items)
                }

        # Weight / BMI
        bmi_hr = None
        bmi_flag = None
        if req.bmi is not None:
            bmi_hr, bmi_flag = hr_bmi_continuous(req.bmi)
            if bmi_hr is not None:
                lifestyle_HRs["weight"] = float(bmi_hr)
                intervention_on["weight"] = True
            else:
                intervention_on["weight"] = False
        else:
            intervention_on["weight"] = False

        # Lifestyle costs
        interv_costs_dict = {}
        for k in ORDER:
            rec_cost = req.lifestyle_costs.get(k, ANNUAL_DEFAULT.get(k, 0.0))
            interv_costs_dict[k] = IntervCost(horizon=1, one_time=0.0, recurring=float(rec_cost))

        # Tiers
        if not req.use_tech:
            tiers = [Tier(0, 0, 0, 0, 0) for _ in range(3)]
        else:
            t1 = req.tier1
            t2 = req.tier2
            t3 = req.tier3
            tiers = [
                Tier(t1.cost, t1.years, t1.p0, t1.g_pp, t1.cap),
                Tier(t2.cost, t2.years, t2.p0, t2.g_pp, t2.cap),
                Tier(t3.cost, t3.years, t3.p0, t3.g_pp, t3.cap),
            ]

        # Engine Inputs instantiation
        inp = Inputs(
            start_age=int(req.age),
            sex=req.sex,
            draws=int(req.draws),
            investment_return=float(req.ret),
            start_capital=float(req.start_capital),
            discretionary_income=float(req.di0),
            income_growth=float(req.income_growth),
            annual_contrib=0.0,
            contrib_growth=0.0,
            lambdaP=float(req.lambda_plateau),
            frontier_drift_days=float(req.drift_days),
            le_trend=float(req.le_improve),
            max_age_today=int(req.max_age_today),
            hc_inflation=float(req.hc_infl),
            lifestyle_HRs=lifestyle_HRs,
            adherence=1.0,
            tiers=tiers,
            tier_repeatable=True,
            intervention_costs=interv_costs_dict,
            intervention_on=intervention_on,
            grid_max_age=max(int(req.age) + 126, 170),
            seed=int(req.seed),
        )

        out = engine.run_monte_carlo(inp)

        # ---------------- KPIs Calculation (matching Admin.py lines 1120-1320) ----------------
        proj_life_mc = np.asarray(out["projected_life_mc"], dtype=float)
        proj_life_det = np.asarray(out.get("projected_life_frac", out["projected_life"]), dtype=float)
        net_worth = np.asarray(out["net_worth"], dtype=float)

        median_life_det = float(np.median(proj_life_det))
        median_life_mc = float(np.median(proj_life_mc))
        
        p5, p95 = np.percentile(out["projected_life"], [5, 95])
        median_net = float(np.median(net_worth))

        yrs_from_habits = float(np.sum(out["yrs_added_interventions"]))

        le_series = out.get("threshold_series")
        if le_series is None:
            le_series = out.get("le_threshold_series")
        alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)
        exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)
        yrs_from_tech = float(np.sum(exp_tech_by_age))

        # Spending Outcomes & Bequest (Admin.py lines 1220-1310)
        age_grid = np.asarray(out["chrono_age"], dtype=float)
        bal_with = np.asarray(out["balance_path"], dtype=float)
        bal_without = np.asarray(out["balance_no_tech_path"], dtype=float)
        bio_age = np.asarray(out["bio_age"], dtype=float)
        tech_years_by_age = np.asarray(out["tech_years_by_age"], dtype=float)

        cum_tech = np.cumsum(tech_years_by_age, axis=1)
        bio_with = np.maximum(0.0, bio_age - cum_tech)
        alive_with = (bio_with < le_series[None, :])
        alive_without = (bio_age < le_series[None, :])

        def _med_death_age(a_mask):
            surv = a_mask.mean(axis=0)
            idx = np.where(surv <= 0.5)[0]
            return float(age_grid[idx[0]]) if idx.size else float(age_grid[-1])

        med_age_with = _med_death_age(alive_with)
        med_age_without = _med_death_age(alive_without)
        mask_with = (age_grid <= med_age_with)

        tech_spend = out.get("tech_costs_by_age")
        tech_spend = np.asarray(tech_spend, dtype=float) if tech_spend is not None else None
        tyba = np.asarray(out["tech_years_by_age"], dtype=float)

        if tech_spend is None:
            spend_by_draw = np.zeros(bio_age.shape[0], dtype=float)
        else:
            spend_by_draw = (tech_spend[:, mask_with] * alive_with[:, mask_with]).sum(axis=1)

        yrs_by_draw = (tyba[:, mask_with] * alive_with[:, mask_with]).sum(axis=1)

        typ_cost_total = float(np.median(spend_by_draw))
        nz = spend_by_draw > 0
        roi_draw = np.full_like(yrs_by_draw, np.nan, dtype=float)
        if np.any(nz):
            roi_draw[nz] = yrs_by_draw[nz] / (spend_by_draw[nz] / 100000.0)
            roi_median = float(np.nanmedian(roi_draw))
        else:
            roi_median = 0.0
        if not np.isfinite(roi_median):
            roi_median = 0.0

        def _terminal_wealth_at_death(bal, a_mask):
            D, T = bal.shape
            tw = np.zeros(D, dtype=float)
            for d in range(D):
                idx = np.where(a_mask[d])[0]
                t = idx[-1] if idx.size else 0
                tw[d] = bal[d, t]
            return tw

        tw_with = _terminal_wealth_at_death(bal_with, alive_with)
        tw_without = _terminal_wealth_at_death(bal_without, alive_without)
        tw_with_med = float(np.median(tw_with))
        tw_without_med = float(np.median(tw_without))
        bequest_delta_med = tw_with_med - tw_without_med

        # ---------------- Forest Plot Impact Analysis (Admin.py lines 1340-1420) ----------------
        LABEL_FOR = {v: k for k, v in CANON.items()}
        forest_rows = []
        draws_impact = int(min(req.draws, 4000))
        
        def _inputs_with(lhr: dict) -> Inputs:
            new_inp = Inputs(**{k: v for k, v in inp.__dict__.items() if k != "lifestyle_HRs"})
            new_inp.lifestyle_HRs = lhr
            new_inp.draws = draws_impact
            return new_inp

        baseline_median = median_life_det

        for key, hr_now in lifestyle_HRs.items():
            if not intervention_on.get(key, False):
                continue
            if abs(float(hr_now) - 1.0) < 1e-9:
                continue

            lhr2 = dict(lifestyle_HRs)
            lhr2[key] = 1.0  # neutralize

            out_i = engine.run_monte_carlo(_inputs_with(lhr2))
            med_i = float(np.median(out_i["projected_life"]))
            effect = baseline_median - med_i

            factor_name = LABEL_FOR.get(key, key).replace("Weight status", "BMI")
            forest_rows.append({
                "factor": factor_name,
                "hr": round(float(hr_now), 2),
                "delta": round(effect, 1),
                "is_harmful": float(hr_now) > 1.0
            })

        forest_rows.sort(key=lambda r: r["hr"], reverse=True)

        # Histogram data for MC & Deterministic
        bins_mc = list(range(int(np.min(proj_life_mc)), int(np.max(proj_life_mc)) + 2))
        counts_mc, edges_mc = np.histogram(proj_life_mc, bins=bins_mc)

        bins_det = list(range(int(np.min(proj_life_det)), int(np.max(proj_life_det)) + 2))
        counts_det, edges_det = np.histogram(proj_life_det, bins=bins_det)

        # Thin scatter sample for wealth vs life
        ty = out["tech_years_by_age"]
        starts = (ty > 0) & np.concatenate([np.ones((ty.shape[0], 1), dtype=bool), ty[:, :-1] == 0], axis=1)
        purchases = starts.sum(axis=1)
        
        sample_size = min(2000, req.draws)
        s_idx = np.random.default_rng(req.seed).choice(req.draws, size=sample_size, replace=False)
        scatter_sample = [
            {
                "life": round(float(proj_life_det[i]), 1),
                "life_mc": round(float(proj_life_mc[i]), 1),
                "nw": round(float(net_worth[i]), 2),
                "purchases": int(purchases[i])
            }
            for i in s_idx
        ]

        # Years added by age curve
        y_int = out["yrs_added_interventions"].tolist()
        y_tech = exp_tech_by_age.tolist()
        chrono = out["chrono_age"].tolist()

        return {
            "status": "success",
            "bucket_summaries": bucket_summaries,
            "metrics": {
                "det_lifespan": round(median_life_det, 0),
                "mc_lifespan": round(median_life_mc, 0),
                "det_years_remaining": int(round(median_life_det - req.age)),
                "mc_years_remaining": int(round(median_life_mc - req.age)),
                "median_net_worth_mm": round(median_net, 2),
                "benefit_from_habits": round(yrs_from_habits, 1),
                "conf_90_p5": round(p5, 0),
                "conf_90_p95": round(p95, 0),
                "benefit_from_treatments": round(yrs_from_tech, 1),
                "expected_treatment_costs": round(typ_cost_total, 0),
                "years_per_100k": round(roi_median, 2),
                "terminal_estate": round(tw_with_med, 0),
                "bequest_delta": round(bequest_delta_med, 0),
            },
            "forest_plot": forest_rows,
            "histograms": {
                "det": {"bins": edges_det[:-1].tolist(), "counts": counts_det.tolist()},
                "mc": {"bins": edges_mc[:-1].tolist(), "counts": counts_mc.tolist()}
            },
            "wealth_scatter": scatter_sample,
            "timeline": {
                "chrono_age": chrono,
                "habits_years": y_int,
                "tech_years": y_tech
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def get_current_user_and_org(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication token required")
    
    token = authorization.split("Bearer ", 1)[1].strip()
    try:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}"
            }
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            sb_user = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.")

    user_id = sb_user.get("id")
    email = sb_user.get("email") or ""
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user session")

    # Mirror to Postgres User & Org if first time (exactly matching Admin.py)
    sess = Session()
    try:
        u = sess.get(User, user_id)
        if not u:
            org_name = f"{email.split('@')[0]}'s Firm" if email else "Advisor Practice"
            org = Org(name=org_name)
            sess.add(org)
            sess.flush()
            u = User(id=user_id, email=email, org_id=org.id)
            sess.add(u)
            sess.commit()
        return {"user_id": u.id, "email": u.email, "org_id": u.org_id}
    finally:
        sess.close()

class SaveRunRequest(BaseModel):
    client_name: str
    client_email: Optional[str] = ""
    client_id: Optional[str] = None
    inputs: Dict[str, Any]
    metrics: Dict[str, Any]
    timeline: Optional[Dict[str, Any]] = None
    bucket_summaries: Optional[Dict[str, Any]] = None

@app.get("/api/auth/me")
def auth_me(auth: dict = Depends(get_current_user_and_org)):
    return {
        "status": "authenticated",
        "user_id": auth["user_id"],
        "email": auth["email"],
        "org_id": auth["org_id"]
    }

@app.get("/api/runs")
def list_runs(search: Optional[str] = None, auth: dict = Depends(get_current_user_and_org)):
    sess = Session()
    try:
        org_id = auth["org_id"]
        q = (sess.query(Run, Client)
             .options(defer(Run.inputs), defer(Run.outputs))
             .join(Client, Run.client_id == Client.id, isouter=True)
             .filter(Run.org_id == org_id))

        if search:
            like = f"%{search.strip()}%"
            q = q.filter(or_(Client.name.ilike(like), Client.email.ilike(like), Run.id.ilike(like)))

        rows = q.order_by(Run.created_at.desc()).limit(30).all()
        res = []
        for r, c in rows:
            res.append({
                "id": r.id,
                "client_id": r.client_id,
                "client_name": c.name if c else "Unknown",
                "client_email": c.email if c else "",
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else ""
            })
        return {"runs": res}
    finally:
        sess.close()

@app.get("/api/runs/{run_id}")
def get_run(run_id: str, auth: dict = Depends(get_current_user_and_org)):
    sess = Session()
    try:
        r = sess.query(Run).filter(Run.id == run_id, Run.org_id == auth["org_id"]).first()
        if not r:
            raise HTTPException(status_code=404, detail="Run not found")
        client = sess.get(Client, r.client_id)
        return {
            "id": r.id,
            "client_name": client.name if client else "Unknown",
            "client_email": client.email if client else "",
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            "inputs": r.inputs,
            "outputs": r.outputs
        }
    finally:
        sess.close()

@app.post("/api/runs")
def save_run(payload: SaveRunRequest, auth: dict = Depends(get_current_user_and_org)):
    sess = Session()
    try:
        org_id = auth["org_id"]
        user_id = auth["user_id"]
        
        # Resolve or create Client
        if payload.client_id:
            client = sess.query(Client).filter(Client.id == payload.client_id, Client.org_id == org_id).first()
            if not client:
                client = Client(org_id=org_id, name=payload.client_name.strip(), email=(payload.client_email or "").strip())
                sess.add(client)
                sess.flush()
        else:
            if not payload.client_name.strip():
                raise HTTPException(status_code=400, detail="Client name is required")
            client = Client(org_id=org_id, name=payload.client_name.strip(), email=(payload.client_email or "").strip())
            sess.add(client)
            sess.flush()

        # Package outputs
        outputs_payload = {
            "summary": payload.metrics,
            "timeline": payload.timeline or {},
            "bucket_summaries": payload.bucket_summaries or {}
        }

        run = Run(
            org_id=org_id,
            operator_id=user_id,
            client_id=client.id,
            inputs=payload.inputs,
            outputs=outputs_payload
        )
        sess.add(run)
        sess.commit()

        return {
            "status": "success",
            "run_id": run.id,
            "client_id": client.id,
            "client_name": client.name,
            "created_at": run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else ""
        }
    finally:
        sess.close()

@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str, auth: dict = Depends(get_current_user_and_org)):
    sess = Session()
    try:
        r = sess.query(Run).filter(Run.id == run_id, Run.org_id == auth["org_id"]).first()
        if not r:
            raise HTTPException(status_code=404, detail="Run not found")
        sess.delete(r)
        sess.commit()
        return {"status": "success", "deleted_id": run_id}
    finally:
        sess.close()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/docs")
def serve_docs():
    return FileResponse(str(STATIC_DIR / "docs.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

