"""
Phase 4: FastAPI wrapper for the PenaltySimulator.

Exposes the simulator over HTTP so the dashboard (or any client) can
call it. Auto-generated docs available at /docs after starting the server.

Usage:
    uvicorn src.api.main:app --reload
"""

import math
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.simulator.penalty_simulator import PenaltySimulator

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Penalty Simulator API",
    description=(
        "Predict the outcome of football penalty kicks using ML models "
        "trained on 1,000+ historical penalties. Built for the FIFA "
        "World Cup 2026."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the simulator ONCE at startup
sim = PenaltySimulator()


# ============================================================
# HELPERS
# ============================================================

def _clean_for_json(d: dict) -> dict:
    """
    Convert a dict's values to JSON-safe types:
    - numpy scalars → native Python types
    - NaN / Inf floats → None (JSON null)
    """
    cleaned = {}
    for k, v in d.items():
        if hasattr(v, 'item'):  # numpy scalar
            v = v.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

GameState = Literal['LEADING', 'LEVEL', 'TRAILING', 'SHOOTOUT']
ShootoutPressure = Literal['CAN_WIN', 'STANDARD', 'MUST_SCORE', 'N/A']


class SimulateRequest(BaseModel):
    taker_id: int = Field(..., description="Taker's StatsBomb player ID")
    keeper_id: int = Field(..., description="Keeper's StatsBomb player ID")
    is_shootout: bool = Field(False, description="Is this a shootout penalty?")
    shootout_pressure: ShootoutPressure = Field(
        "N/A",
        description="CAN_WIN, STANDARD, MUST_SCORE, or N/A (in-game)",
    )
    game_state: Optional[GameState] = Field(
        "LEVEL",
        description="LEADING, LEVEL, TRAILING, or SHOOTOUT (auto-set if omitted)",
    )
    score_diff_at_penalty: int = Field(0, description="Taker's team score minus opponent's")
    minute: int = Field(60, ge=0, le=130, description="Minute of the match")
    shootout_kick_num: Optional[int] = Field(None, description="Kick number for the taker's team in shootout")
    is_left_foot: Optional[bool] = Field(None, description="Override foot (default: taker's preferred)")
    is_right_foot: Optional[bool] = Field(None, description="Override foot")
    n_simulations: int = Field(10000, ge=100, le=100000, description="Monte Carlo iterations")
    random_state: Optional[int] = Field(None, description="Seed for reproducible Monte Carlo")

    @model_validator(mode='after')
    def coerce_and_validate(self):
        """
        Auto-correct context inconsistencies (e.g., shootout at minute 60),
        then enforce hard rules that can't be expressed via type annotations.
        """
        # ----- Auto-correct contextual contradictions -----
        if self.is_shootout:
            if self.minute < 120:
                self.minute = 120
            if self.game_state !='SHOOTOUT':
                self.game_state = 'SHOOTOUT'
            if self.score_diff_at_penalty != 0:
                self.score_diff_at_penalty = 0
            if self.shootout_pressure == 'N/A':
                self.shootout_pressure = 'STANDARD'
        else:
            if self.shootout_pressure != 'N/A':
                self.shootout_pressure = 'N/A'
            if self.shootout_kick_num is not None and self.shootout_kick_num > 0:
                self.shootout_kick_num = None
            if self.game_state == 'SHOOTOUT':
                self.game_state = 'LEVEL'

        # ----- Hard rules: foot booleans must not contradict -----
        if self.is_left_foot is True and self.is_right_foot is True:
            raise ValueError(
                "is_left_foot and is_right_foot cannot both be True. "
                "Pick one, or omit both to use the taker's preferred foot."
            )
        if self.is_left_foot is False and self.is_right_foot is False:
            raise ValueError(
                "is_left_foot and is_right_foot cannot both be False. "
                "A penalty must be taken with one foot or the other."
            )

        return self


class SimulateResponse(BaseModel):
    p_goal: float
    zone_probs: dict
    outcome_probs: dict
    outcome_by_zone: dict
    simulations: dict
    taker_name: str
    keeper_name: str
    taker_reliability: str
    keeper_reliability: str
    taker_total_penalties: int
    keeper_total_penalties: int
    context: dict


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Quick health check + API overview."""
    return {
        "name": "Penalty Simulator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /simulate": "Run a penalty simulation",
            "GET /takers": "List all known takers",
            "GET /keepers": "List all known keepers",
            "GET /takers/search?q=...": "Search takers by name",
            "GET /keepers/search?q=...": "Search keepers by name",
            "GET /docs": "Interactive API documentation",
        },
        "n_takers": len(sim.takers),
        "n_keepers": len(sim.keepers),
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """Run a complete penalty simulation between the given taker and keeper."""
    try:
        return sim.simulate(
            taker_id=req.taker_id,
            keeper_id=req.keeper_id,
            is_shootout=req.is_shootout,
            shootout_pressure=req.shootout_pressure,
            game_state=req.game_state,
            score_diff_at_penalty=req.score_diff_at_penalty,
            minute=req.minute,
            shootout_kick_num=req.shootout_kick_num,
            is_left_foot=req.is_left_foot,
            is_right_foot=req.is_right_foot,
            n_simulations=req.n_simulations,
            random_state=req.random_state,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/takers")
def list_takers(
    limit: int = Query(50, ge=1, le=500, description="Number of results to return"),
    reliability: Optional[Literal['low', 'medium', 'high']] = Query(None),
):
    """List takers sorted by number of penalties taken."""
    df = sim.list_takers()
    if reliability:
        df = df[df['reliability'] == reliability]
    return df.head(limit).to_dict(orient='records')


@app.get("/keepers")
def list_keepers(
    limit: int = Query(50, ge=1, le=500),
    reliability: Optional[Literal['low', 'medium', 'high']] = Query(None),
):
    """List keepers sorted by penalties faced."""
    df = sim.list_keepers()
    if reliability:
        df = df[df['reliability'] == reliability]
    return df.head(limit).to_dict(orient='records')


@app.get("/takers/search")
def search_takers(q: str = Query(..., min_length=2, description="Name substring (case-insensitive)")):
    """Search takers by name."""
    df = sim.list_takers()
    matches = df[df['taker_name'].str.contains(q, case=False, na=False)]
    return matches.to_dict(orient='records')


@app.get("/keepers/search")
def search_keepers(q: str = Query(..., min_length=2)):
    """Search keepers by name."""
    df = sim.list_keepers()
    matches = df[df['keeper_name'].str.contains(q, case=False, na=False)]
    return matches.to_dict(orient='records')


@app.get("/taker/{taker_id}")
def get_taker_profile(taker_id: int):
    """Return the full profile for a specific taker."""
    taker = sim._lookup_taker(taker_id)
    if taker is None:
        raise HTTPException(status_code=404, detail=f"Taker {taker_id} not found")
    return _clean_for_json(taker)


@app.get("/keeper/{keeper_id}")
def get_keeper_profile(keeper_id: int):
    """Return the full profile for a specific keeper."""
    keeper = sim._lookup_keeper(keeper_id)
    if keeper is None:
        raise HTTPException(status_code=404, detail=f"Keeper {keeper_id} not found")
    return _clean_for_json(keeper)
