---
title: Penalty Simulator API
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Penalty Simulator API

FastAPI service that predicts penalty kick outcomes using ML models trained on 1,000+ historical penalties from StatsBomb open data.

Built for the FIFA World Cup 2026.

## Endpoints

Once running, see `/docs` for the auto-generated Swagger UI.

Key endpoints:
- `POST /simulate` — run a penalty simulation between a taker and keeper
- `GET /takers`, `GET /keepers` — list known players
- `GET /takers/search?q=...`, `GET /keepers/search?q=...` — search by name
- `GET /taker/{id}`, `GET /keeper/{id}` — individual player profiles

## Companion dashboard

This API powers the [Penalty Simulator dashboard]<streamlit-url-post-deployment> — a Streamlit interface for exploring predictions visually.

## Tech stack

- Python 3.12, FastAPI, XGBoost, scikit-learn
- Deployed via Docker on Hugging Face Spaces
