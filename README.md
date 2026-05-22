# ⚽ FIFA Penalty Simulator — World Cup 2026

> Predict the outcome of any penalty kick matchup using ML models trained on 1,000+ historical penalties from StatsBomb open data.

**[🌐 Live Dashboard](https://fifa-penalty-simulator.streamlit.app/)** · **[🔌 API Docs](https://nabs7-fifa-penalty-simulator-api.hf.space/docs)** · **[🤖 Models on HF](https://huggingface.co/nabs7/penalty-simulator-models)**

![Dashboard screenshot](assets\dashboard_picture\dash_pic.png)

---

## What it does

Pick a taker, pick a keeper, set the context (in-game vs shootout, pressure, minute), and the simulator returns:

- **Probability of a goal** — the headline number
- **Predicted shot location** — six-zone heatmap overlaid on an illustrated goal mouth
- **Outcome breakdown** — full distribution over GOAL / SAVED / POST / WIDE / OVER
- **Player profiles** — sample sizes and reliability indicators for both players

The dashboard runs on Streamlit Cloud, calls a FastAPI backend on Hugging Face Spaces, which loads ML models hosted on Hugging Face Model Hub. Three decoupled services, all free-tier.

---

## How it works

### The pipeline

```
StatsBomb open data
       ↓
1,065 penalties extracted (FIFA WC, Euros, top 5 leagues, etc.)
       ↓
Player profiles built with Bayesian smoothing
       ↓
41 features engineered per penalty (leave-one-out to prevent leakage)
       ↓
Two XGBoost models trained + isotonic-calibrated
       ↓
FastAPI service exposes /simulate endpoint
       ↓
Streamlit dashboard calls API and visualizes predictions
```

### The two-model architecture

The simulator uses **two complementary models** that work together:

| Model | Predicts | Output |
|---|---|---|
| **Shot placement** | Where the shot will go (if on target) | 6 zones: TL, TC, TR, BL, BC, BR |
| **Outcome** | What happens to the shot | 5 classes: GOAL, SAVED, POST, WIDE, OVER |

At inference time, the simulator:
1. Predicts the zone distribution from the matchup features
2. For each zone, predicts the outcome distribution
3. Marginalizes: `P(goal) = Σ P(zone) × P(goal | zone)`
4. Runs Monte Carlo (default 10,000 iterations) for distributional uncertainty

### Feature engineering highlights

The hardest part of this project wasn't training models — it was building features that work on small, sparse data without leaking the target.

- **Leave-one-out aggregations** for per-player features. For each penalty, the player's "career conversion rate" is computed excluding that row. Without this, the model would effectively cheat on training data.
- **Bayesian smoothing** on per-player rates. 93% of takers in our dataset have ≤2 penalties — raw rates are too noisy. We smooth toward population priors with a strength parameter tuned to the data size.
- **Matchup-aware features**, e.g., `taker_zone_TR_prob × keeper_save_rate_TR`. These capture interaction effects that aren't visible from looking at either player alone.
- **Engineered context features**: `is_high_pressure` flag combining shootout state and trailing-score-late-game scenarios, mirroring the psychological reality penalty takers face.

### Calibration matters more than accuracy

Tree models like XGBoost are notoriously overconfident — they often output probabilities like 99% when the actual rate is 80%. For a dashboard showing "Probability of Goal," this is a deal-breaker.

Three layers of calibration handle this:

1. **Per-class isotonic regression** fit on out-of-fold predictions
2. **Probability floors** ([0.02, 0.98]) prevent the calibrator from outputting impossible 0% or 100% predictions
3. **Profile blending** for high-volume takers (≥5 penalties): 50/50 weighted average of model prediction and player's smoothed profile. This grounds predictions in player history rather than letting the model extrapolate aggressively.

On the held-out 2022 World Cup test set: **mean predicted goal probability 69.2% vs actual 67.2%** — calibration within 2 percentage points.

---

## Performance

On 64 held-out penalties from the 2022 FIFA World Cup:

| Metric | Result | Baseline |
|---|---|---|
| Outcome log-loss (calibrated) | **0.45** | 0.86 (zone-conditional baseline) |
| Binary goal accuracy | **84.4%** | 67.2% (always predict goal) |
| Predicted vs actual goal rate | **69.2% vs 67.2%** | n/a |

For shot placement (6-class): test log-loss 1.35 vs baselines of 1.55 (population) and 1.56 (taker-prior).

---

## Tech stack

**Data & modeling**
- Python 3.12
- StatsBomb open data via `statsbombpy`
- pandas, numpy, scipy
- XGBoost (tree-based gradient boosting)
- scikit-learn (calibration, evaluation)

**API**
- FastAPI with Pydantic validation
- uvicorn ASGI server
- Custom request validators (auto-corrects contextual contradictions)
- Auto-generated OpenAPI docs at `/docs`

**Dashboard**
- Streamlit
- Plotly (interactive charts)
- Inline SVG for goal-mouth illustration and player silhouettes
- Base64 image embedding (no static file serving needed)

**Deployment**
- Dashboard → Streamlit Cloud
- API → Hugging Face Spaces (Docker)
- Models → Hugging Face Model Hub (downloaded on API startup)

---

## Project structure

```
penalty-simulator/
├── src/
│   ├── data/
│   │   ├── extract_penalties.py        # Pull penalties from StatsBomb
│   │   ├── clean_penalties.py          # Map shots to 3×2 zones, classify outcomes
│   │   ├── enrich_score_state.py       # Add game state and shootout pressure
│   │   ├── build_profiles.py           # Bayesian-smoothed player profiles
│   │   └── scrape_wikipedia_photos.py  # Player headshots for the dashboard
│   ├── features/
│   │   └── build_features.py           # Leave-one-out feature engineering
│   ├── models/
│   │   ├── calibration.py              # IsotonicMultiClassCalibrator
│   │   ├── train_shot_placement.py     # 6-zone classifier
│   │   └── train_outcome.py            # 5-class outcome predictor
│   ├── simulator/
│   │   └── penalty_simulator.py        # Inference engine combining both models
│   └── api/
│       └── main.py                     # FastAPI app
├── dashboard/
│   ├── app.py                          # Streamlit UI
│   └── api_client.py                   # HTTP wrapper for the API
├── notebooks/
│   ├── 00_explore_statsbomb.ipynb      # Data exploration
│   └── 01_eda.ipynb                    # EDA findings
├── data/processed/                     # Smoothed profiles, priors, metadata
├── assets/                             
│   ├── photos/                         # Wikipedia-scraped player headshots
│   └── dashboard_picture/              # Dashboard screenshot
├── Dockerfile                          # For Hugging Face Spaces deployment
└── requirements.txt
```

---

## Run it locally

### Prerequisites
- Python 3.12
- A Python virtual environment

### Setup

```bash
git clone https://github.com/nabs7/penalty-simulator.git
cd penalty-simulator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Generate models (optional — if not present locally, the simulator downloads them from HF on startup)

```bash
python -m src.data.extract_penalties      # ~10 min, fetches StatsBomb data
python -m src.data.clean_penalties
python -m src.data.enrich_score_state     # ~5 min
python -m src.data.build_profiles
python -m src.features.build_features
python -m src.models.train_shot_placement # ~3 min
python -m src.models.train_outcome        # ~3 min
```

### Run the API and dashboard

In two terminals:

```bash
# Terminal 1
uvicorn src.api.main:app --reload

# Terminal 2
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`. API docs at `http://localhost:8000/docs`.

### Configuration

The dashboard reads `API_URL` from a `.env` file. For local development:

```
API_URL=http://127.0.0.1:8000
```

For deployment, set this as an environment variable / secret on the hosting platform.

---

## Design decisions worth highlighting

A few choices that took thought:

### Two models instead of one

A single end-to-end model "predict the outcome given the matchup" loses signal about *where* the shot is going — which is genuinely interesting (it's the dashboard's most engaging visualization). Splitting into placement + outcome lets users see the model's reasoning, not just its conclusion.

### Profile blending for elite players

Without blending, the shot-placement model occasionally produced extreme predictions (one player aiming top-right 80% of the time) because it learned patterns from a handful of similar-profile takers. Blending 50/50 with the player's actual smoothed profile keeps predictions grounded in player history while still letting context (shootout, pressure) shift them.

### Models on Hugging Face Model Hub, not in Git

Hugging Face Spaces blocks binary files (.pkl, .parquet) from regular Git pushes — they require Xet storage. Rather than navigate that, models are hosted on HF Model Hub and downloaded by the API on first startup. Caches locally afterward. Cleaner separation anyway: code in Git, artifacts on a model registry.

### Auto-correction in API validation

The API accepts contradictory inputs gracefully — if you set `is_shootout=true` but `minute=60`, it auto-corrects minute to 120. Better UX than rejecting with a 422 error, and the response echoes back the corrected values so users can see what was assumed.

---

## Limitations

Honest about what this model can and can't do:

- **Small dataset**: 1,065 penalties is small for ML. Even with smoothing and LOO, predictions for rare scenarios (e.g., top-corner attempts) are noisier than headline metrics suggest.
- **Class imbalance**: GOAL dominates at 75% of outcomes. The model rarely predicts POST/WIDE/OVER because there are only 26-36 examples of each in training.
- **Player coverage**: 83% of takers in the dataset have ≤2 historical penalties. The model uses Bayesian priors to handle these, but predictions for low-data players are essentially regression to the population mean.
- **No World Cup 2026 data yet**: The 2026 tournament hadn't started at training time. Predictions for new WC2026 players default to population priors plus whatever StatsBomb open data has on them.
- **Calibration ceiling**: Even after all calibration work, predictions above 90% should be viewed skeptically. Penalty kicks are inherently noisy.

---

## Acknowledgments

- **StatsBomb** for making their event-level football data freely available under their open data license
- **Wikipedia** for player photos via the Wikimedia Commons API
- **Hugging Face** for free-tier Spaces, Model Hub, and the `huggingface_hub` library
- **Streamlit** for the dashboard framework and free Cloud hosting

---

## License

MIT — feel free to fork, modify, learn from, or adapt.

---

*Built as a portfolio project exploring end-to-end ML engineering: data extraction, feature engineering with leakage prevention, model calibration, and decoupled deployment.*
