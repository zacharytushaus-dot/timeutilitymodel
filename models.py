from pydantic import BaseModel
from typing import Any, Dict

class RunInputs(BaseModel):
    age: int
    sex: str
    draws: int
    ret: float
    start_capital: float
    di0: float
    income_growth: float
    hc_infl: float
    lambdaP: float
    drift_days: float
    le_trend: float
    max_age_today: int
    lifestyle_HRs: Dict[str, float]
    intervention_on: Dict[str, bool]
    tier1: Dict[str, Any]
    tier2: Dict[str, Any]
    tier3: Dict[str, Any]
    seed: int

class RunOutputs(BaseModel):
    summary: Dict[str, Any]    # small metric payload
    figs_data: Dict[str, Any]  # arrays needed to render the charts