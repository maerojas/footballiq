"""
API local de predicciones FootballIQ.

  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import FootballPredictor, TeamStats, standing_row_to_stats

FD_API = "https://api.football-data.org/v4"
FILES_DIR = Path(__file__).resolve().parent.parent / "files"

app = FastAPI(title="FootballIQ Predictor", version="1.0.0")
predictor = FootballPredictor()


@app.on_event("startup")
def warmup_model() -> None:
    """Primera predicción en startup para evitar demora en el primer click."""
    sample = TeamStats(1, "Home", 1, 10, 6, 2, 2, 18, 8, 20, "W,W,D,W,W")
    rival = TeamStats(2, "Away", 8, 10, 4, 2, 4, 12, 14, 14, "L,W,L,D,W")
    predictor.predict(sample, rival, [sample, rival])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TeamInput(BaseModel):
    id: int
    name: str = ""


class StandingInput(BaseModel):
    team: TeamInput
    position: int
    playedGames: int = 1
    won: int = 0
    draw: int = 0
    lost: int = 0
    goalsFor: int = 0
    goalsAgainst: int = 0
    points: int = 0
    form: str = ""


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    standings: list[StandingInput]


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "poisson+ml" if predictor.classifier else "poisson",
    }


@app.api_route("/api/fd/{path:path}", methods=["GET"])
def proxy_football_data(path: str, request: Request) -> Response:
    """Proxy a football-data.org para evitar CORS en el navegador."""
    api_key = (request.headers.get("X-Auth-Token") or "").strip()
    if not api_key:
        raise HTTPException(401, detail="Ingresá tu API key de football-data.org")

    url = f"{FD_API}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    try:
        resp = requests.get(
            url,
            headers={"X-Auth-Token": api_key},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, detail=f"Sin conexión con football-data.org: {exc}") from exc

    if resp.status_code >= 400:
        try:
            body = resp.json()
            msg = body.get("message") or body.get("detail") or f"Error HTTP {resp.status_code}"
        except ValueError:
            msg = f"Error HTTP {resp.status_code}"
        if resp.status_code == 403:
            msg = "API key inválida o sin acceso a esta liga"
        elif resp.status_code == 429:
            msg = "Límite de requests alcanzado (plan gratis: 10/min)"
        raise HTTPException(resp.status_code, detail=msg)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type", "application/json"),
    )


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if not req.standings:
        raise HTTPException(400, "Se requiere la tabla de posiciones")

    all_stats: list[TeamStats] = [
        standing_row_to_stats(s.model_dump()) for s in req.standings
    ]
    by_id = {s.team_id: s for s in all_stats}

    home = by_id.get(req.home_team_id)
    away = by_id.get(req.away_team_id)
    if not home or not away:
        raise HTTPException(404, "Equipo no encontrado en la tabla")

    return predictor.predict(home, away, all_stats)


# Frontend estático (index.html) — debe ir al final
if FILES_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FILES_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
