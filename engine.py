
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from ssa_tables import male as SSA_MALE, female as SSA_FEMALE
except Exception:
    SSA_MALE = np.linspace(0.00586, 0.906532, 120)
    SSA_FEMALE = np.linspace(0.005063, 0.906532, 120)


@dataclass
class Tier:
    cost_today: float
    years_gain: float
    base_prob: float
    growth_per_year: float
    cap_prob: float

@dataclass
class IntervCost:
    horizon: int          # E: replace every N years (0 or 1 = every year)
    one_time: float       # F: one-time cost
    recurring: float      # G: recurring per year

@dataclass
class Inputs:
    start_age: int = 30
    sex: str = "Male"
    draws: int = 10_000
    investment_return: float = 0.05
    start_capital: float = 10_000.0
    annual_contrib: float = 10_000.0
    contrib_growth: float = 0.00   # kept for back-compat (unused when discretionary_income is provided)
    # Excel-style: discretionary income grows by V23; contribution = max(0, DI - health_spend)
    discretionary_income: float = 0.0
    income_growth: float = 0.0


    # Longevity params
    lambdaP: float = 0.6                # plateau hazard parameter
    frontier_drift_days: float = 15.0   # days per calendar year (for hazard shape and frontier mode)
    le_trend: float = 0.002             # 0.2% per year LE improvement (used for societal LE mode)
    max_age_today: float = 119.0

    # Threshold mode for death
    threshold_mode: str = "societal"    # "societal" (default) or "frontier"

    # Optional override for today's societal LE; 0 means compute from SSA
    societal_le_today: float = 0.0

    # Tech / cost params
    hc_inflation: float = 0.03
    lifestyle_HRs: Dict[str, float] = None
    adherence: float = 1.0
    tiers: List[Tier] = None
    tier_repeatable: bool = False       # one purchase per tier per draw (default False)
    # Costs for lifestyle interventions (keys normalized: sleep, exercise, mediterraneandiet, meditation, redlight)
    intervention_costs: Dict[str, IntervCost] = None
    intervention_on: Dict[str, bool] = None

    grid_max_age: int = 170
    seed: int = 42

    def ensure_defaults(self) -> None:
        if self.lifestyle_HRs is None:
            self.lifestyle_HRs = {
                "sleep": 0.88,
                "exercise": 0.68,
                "diet": 0.77,
                "meditation": 0.93,
                "redlight": 1.00,
            }
        if self.tiers is None:
            self.tiers = [
                Tier(12_000.0, 0.7,   0.03,  0.0015, 0.10),
                Tier(550_000.0, 2.5,  0.006, 0.0015, 0.10),
                Tier(2_400_000.0, 7.0,0.0012,0.0015, 0.10),
            ]
        if self.intervention_costs is None:
            self.intervention_costs = {
                "sleep":            IntervCost(horizon=3, one_time=80.0,   recurring=0.0),
                "exercise":         IntervCost(horizon=1, one_time=0.0,    recurring=240.0),
                "redlight":         IntervCost(horizon=3, one_time=500.0,  recurring=0.0),
                "mediterraneandiet":IntervCost(horizon=1, one_time=0.0,    recurring=5400.0),
                "meditation":       IntervCost(horizon=1, one_time=0.0,    recurring=70.0),
            }
        if self.intervention_on is None:
            # Default: ON if supplied HR < 1; OFF otherwise
            self.intervention_on = {k: (float(v) < 1.0 - 1e-12) for k, v in self.lifestyle_HRs.items()}


# ----------------------- utilities -----------------------

def _q_table(sex: str) -> np.ndarray:
    return SSA_MALE if sex.lower().startswith("m") else SSA_FEMALE


def _frontier_age_series(inp: Inputs, T: int) -> np.ndarray:
    years_offset = np.arange(T, dtype=float)
    return float(inp.max_age_today) + (float(inp.frontier_drift_days) / 365.0) * years_offset


def _q_piecewise(age: np.ndarray, fAge: np.ndarray, q_tab: np.ndarray, lambdaP: float) -> np.ndarray:
    """
    Baseline annual death probability q(age) with:
      - SSA table up to 119,
      - exponential glide 119 -> frontier max-age,
      - plateau beyond frontier with q = 1 - exp(-lambdaP).
    age : (D,T)
    fAge: (T,)
    """
    q_plat = 1.0 - np.exp(-lambdaP)

    a_idx = np.clip(np.floor(age).astype(int), 0, 119)
    q_ssa = q_tab[a_idx]

    a0 = float(q_tab[119])
    denom = np.maximum(fAge - 119.0, 1e-12)
    b0 = np.log(a0 / q_plat) / denom          # (T,)
    q_exp = a0 * np.exp(-(age - 119.0) * b0[None, :])  # (D,T)

    q_core = np.where(age <= 119.0, q_ssa, q_exp)
    q = np.where(age < fAge[None, :], q_core, q_plat)
    return np.clip(q, 0.0, 1.0)

def _compute_contrib_schedule(inp: Inputs, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Excel-parity finance:
      discretionary_income_t = DI0 * (1+income_growth)^t
      health_spend_t = sum(recurring*HC^t + one_time*HC^t on t%horizon==0 for active interventions)
      contribution_t = max(0, discretionary_income_t - health_spend_t)
    Returns (contrib_by_year, health_spend_by_year, di_series).
    """
    years = np.arange(T)
    hc_factor = (1.0 + float(inp.hc_inflation)) ** years
    di_series = float(inp.discretionary_income) * (1.0 + float(inp.income_growth)) ** years
    health = np.zeros(T, dtype=float)
    if inp.intervention_costs is not None and inp.intervention_on is not None:
        for key, cfg in inp.intervention_costs.items():
            if not inp.intervention_on.get(key, False):
                continue
            # Recurring every year
            health += cfg.recurring * hc_factor
            # One-time at t % horizon == 0 (t=0 counts)
            h = max(1, int(cfg.horizon))  # guard horizon=0 -> treat as every year
            mask = (years % h == 0).astype(float)
            health += cfg.one_time * hc_factor * mask
    contrib = np.maximum(0.0, di_series - health)
    return contrib, health, di_series

def _excel_style_intervention_years(inp: Inputs, chrono_age: np.ndarray, fAge: np.ndarray, q_tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Excel parity: per-intervention probability-scale HR, sum deltas vs baseline.
    Returns:
      yrs_added_sum : (T,) expected years added by interventions (sum across active interventions)
      S_base        : (T,) baseline survival
    """
    T = chrono_age.size
    age_col = chrono_age[None, :]
    q_base = _q_piecewise(age_col, fAge, q_tab, inp.lambdaP)[0]   # (T,)

    # Baseline survival (Excel indexing): first column = 1, multiply starting next year
    S_base = np.empty(T, dtype=float)
    S_base[0] = 1.0
    for t in range(1, T):
        S_base[t] = S_base[t-1] * (1.0 - q_base[t])

    yrs_added_sum = np.zeros(T, dtype=float)

    # Include both beneficial and harmful items (negative years allowed)
    multipliers = []
    for _, m in inp.lifestyle_HRs.items():
        m = float(m)
        if abs(m - 1.0) > 1e-12:
            multipliers.append(np.clip(m, 0.0, 4.0))

    for m in multipliers:
        q_k = np.clip(q_base * m, 0.0, 1.0)
        S_k = np.empty(T, dtype=float)
        S_k[0] = 1.0
        for t in range(1, T):
            S_k[t] = S_k[t-1] * (1.0 - q_k[t])
        yrs_added_sum += (S_k - S_base)

    return yrs_added_sum, S_base


def _societal_le_today_from_ssa(inp: Inputs, q_tab: np.ndarray) -> float:
    """
    Compute today's societal life expectancy at birth from SSA q_tab and plateau λ.
    We use a static frontier equal to max_age_today for this derivation.
    """
    T_age = 200  # generous horizon
    age = np.arange(T_age, dtype=float)[None, :]         # (1, T_age)
    fAge_static = np.full((T_age,), float(inp.max_age_today))  # (T_age,)
    q = _q_piecewise(age, fAge_static, q_tab, inp.lambdaP)[0]  # (T_age,)

    S = np.empty(T_age, dtype=float)
    S[0] = 1.0 * (1.0 - q[0])
    for t in range(1, T_age):
        S[t] = S[t-1] * (1.0 - q[t])

    LE_today = S.sum()  # life expectancy at birth
    return float(LE_today)


def _le_threshold_series_societal(inp: Inputs, T: int, q_tab: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Societal LE threshold series = LE_today * (1 + le_trend)^t, independent of the frontier drift.
    Returns series and the LE_today anchor used.
    """
    LE0 = inp.societal_le_today if inp.societal_le_today > 0 else _societal_le_today_from_ssa(inp, q_tab)
    years = np.arange(T, dtype=float)
    LE_series = LE0 * (1.0 + float(inp.le_trend)) ** years
    return LE_series, float(LE0)


# ----------------------- core simulation -----------------------

def run_monte_carlo(inp: Inputs) -> Dict[str, Any]:
    inp.ensure_defaults()
    rng = np.random.default_rng(inp.seed)

    # Grid
    T = max(1, int(inp.grid_max_age - inp.start_age))
    chrono_age = inp.start_age + np.arange(T, dtype=float)  # (T,)
    fAge = _frontier_age_series(inp, T)                      # (T,) for hazard shapes
    q_tab = _q_table(inp.sex)

    # Excel-style interventions (years added series + baseline survival)
    yrs_added_interventions, S_base = _excel_style_intervention_years(inp, chrono_age, fAge, q_tab)
    cum_int = np.cumsum(yrs_added_interventions)

    # Finance: contribution schedule from DI minus health spend
    contrib_by_year, health_by_year, di_series = _compute_contrib_schedule(inp, T)

    # Simulate tech tiers + balance path
    draws = int(inp.draws)
    balance = np.full(draws, float(inp.start_capital), dtype=float)
    balance_no_tech = np.full(draws, float(inp.start_capital), dtype=float)   # ← NEW

    balance_path = np.zeros((draws, T), dtype=float)
    balance_no_tech_path = np.zeros((draws, T), dtype=float)                  # ← NEW

    tech_years_by_age = np.zeros((draws, T), dtype=float)
    tech_costs_by_age = np.zeros((draws, T), dtype=float)                     # ← NEW: dollars spent on tech
    miss_streak = np.zeros((draws, len(inp.tiers)), dtype=int)
    available = np.ones((draws, len(inp.tiers)), dtype=bool)


    # Purchase order: cheapest-first by today's (uninflated) cost
    order = np.argsort([t.cost_today for t in inp.tiers]).tolist()
    for t in range(T):
        infl = (1.0 + inp.hc_inflation) ** t
        # Start-of-year budget = StartingCapital only (per draw)
        budget = balance.copy()
        total_cost = np.zeros(draws, dtype=float)
        total_years = np.zeros(draws, dtype=float)

        # Precompute arrivals and per-tier inflated costs/years
        arrived_list, cost_list, years_list = [], [], []
        for j, tier in enumerate(inp.tiers):
            p_raw = np.minimum(tier.cap_prob, tier.base_prob + tier.growth_per_year * miss_streak[:, j])
            p_t = np.where(available[:, j], p_raw, 0.0)
            arrived = rng.uniform(size=draws) < p_t
            arrived_list.append(arrived)
            # Reset miss-streak on trigger; else increment
            miss_streak[:, j] = np.where(arrived, 0, miss_streak[:, j] + 1)
            cost_list.append(np.where(arrived, tier.cost_today * infl, 0.0))
            years_list.append(np.where(arrived, tier.years_gain, 0.0))

        # Buy tier-by-tier, cheapest-first, gated by budget
        for j in order:
            cost_t = cost_list[j]
            yrs_t  = years_list[j]
            purchase = (cost_t > 0) & available[:, j] & (budget >= cost_t)
            buy_cost  = np.where(purchase, cost_t, 0.0)
            buy_years = np.where(purchase, yrs_t, 0.0)
            total_cost  += buy_cost
            total_years += buy_years
            budget      -= buy_cost
            if not inp.tier_repeatable:
                available[purchase, j] = False

        # Contributions from DI minus health spend (scalar per year)
        balance += contrib_by_year[t]
        balance -= total_cost                     # tech purchases reduce balance
        balance *= (1.0 + inp.investment_return)
        balance_path[:, t] = balance
        tech_years_by_age[:, t] = total_years
        tech_costs_by_age[:, t] = total_cost           # ← NEW

        # Parallel path with NO treatments (same DI/returns; never subtract cost)
        balance_no_tech += contrib_by_year[t]
        balance_no_tech *= (1.0 + inp.investment_return)
        balance_no_tech_path[:, t] = balance_no_tech   # ← NEW

    # Biological age
    cum_tech = np.cumsum(tech_years_by_age, axis=1)
    cum_added = cum_tech + cum_int[None, :]
    bio_age = chrono_age[None, :] - cum_added

    # --------- Threshold series (societal or frontier) ---------
    if inp.threshold_mode.lower().startswith("soc"):
        le_series, LE0 = _le_threshold_series_societal(inp, T, q_tab)
        threshold_series = le_series
        threshold_label = "societal"
    else:
        threshold_series = fAge
        LE0 = np.nan
        threshold_label = "frontier"

    # Death index by crossing (deterministic per draw)
    cross = (bio_age >= threshold_series[None, :])
    died = cross.any(axis=1)
    death_idx_cross = np.where(died, cross.argmax(axis=1), T - 1)
    projected_life_threshold = inp.start_age + death_idx_cross

    # ---- Fractional crossing (linear interpolation inside the year) ----
    # t = first crossed year; t_prev = last not‑yet‑crossed year
    t = death_idx_cross
    t_prev = np.maximum(t - 1, 0)

    # distance to threshold just before crossing and at crossing
    delta_prev = threshold_series[t_prev] - bio_age[np.arange(draws), t_prev]
    delta_curr = threshold_series[t]      - bio_age[np.arange(draws), t]

    # fraction of the year between t_prev and t where crossing occurs
    eps = 1e-12
    frac = np.zeros_like(delta_prev, dtype=float)
    mask = (t > 0) & died
    frac[mask] = np.clip(delta_prev[mask] / (delta_prev[mask] - delta_curr[mask] + eps), 0.0, 1.0)

    # fractional projected life (fallback to integer if no crossing)
    projected_life_threshold_frac = np.where(
    died, inp.start_age + t_prev + frac, inp.start_age + t
    ).astype(float)

    # Net worth at death (threshold crossing)
    idx_clamped = np.clip(death_idx_cross, 0, T - 1)
    net_at_death = balance_path[np.arange(draws), idx_clamped]


    # --------- Optional MC hazard sampling (distribution) ---------
    q_eff = _q_piecewise(bio_age, fAge, q_tab, inp.lambdaP)
    alive_year = (rng.uniform(size=q_eff.shape) >= q_eff).astype(np.int8)
    alive_status = np.cumprod(alive_year, axis=1)
    died_any = (alive_status == 0).any(axis=1)
    death_idx_mc = np.where(died_any, (alive_status == 0).argmax(axis=1), T - 1)
    projected_life_mc = inp.start_age + death_idx_mc

    return {
        "projected_life": projected_life_threshold.astype(float),  # threshold mode result
        "projected_life_frac": projected_life_threshold_frac,
        "projected_life_mc": projected_life_mc.astype(float),      # MC distribution (optional)
        "threshold_series": threshold_series,                       # (T,) societal or frontier
        "threshold_label": threshold_label,
        "LE_today_anchor": float(LE0),
        "net_worth": net_at_death / 1e6,
        "yrs_added_interventions": yrs_added_interventions,
        "S_base": S_base,
        "tech_years_by_age": tech_years_by_age,
        "bio_age": bio_age,
        "chrono_age": chrono_age,
        "balance_path": balance_path,
        "balance_no_tech_path": balance_no_tech_path,
        "tech_costs_by_age": tech_costs_by_age,
        # Diagnostics
        "health_spend_by_year": health_by_year,
        "contrib_by_year": contrib_by_year,
        "discretionary_income_by_year": di_series,
        "balance_path": balance_path,                     # ← NEW (with tech)
        "balance_no_tech_path": balance_no_tech_path,     # ← NEW (without tech)
        "tech_costs_by_age": tech_costs_by_age,           # ← NEW
    }
