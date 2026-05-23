"""
Entrena el clasificador ML con partidos finalizados de football-data.org.

Uso:
  FOOTBALL_API_KEY=tu_key python train.py
  FOOTBALL_API_KEY=tu_key python train.py --league PL PD SA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from model import META_PATH, SKLEARN_PATH, TeamStats, standing_row_to_stats

API_BASE = "https://api.football-data.org/v4"
LEAGUES = {
    "PL": 2021,
    "PD": 2014,
    "SA": 2019,
    "BL1": 2002,
    "FL1": 2015,
}


def fetch(api_key: str, endpoint: str) -> dict:
    r = requests.get(
        f"{API_BASE}/{endpoint}",
        headers={"X-Auth-Token": api_key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def match_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def build_features(home: TeamStats, away: TeamStats, total: int) -> list[float]:
    return [
        home.position / total,
        away.position / total,
        home.ppg,
        away.ppg,
        home.gf_per_game,
        away.gf_per_game,
        home.ga_per_game,
        away.ga_per_game,
        home.form_points,
        away.form_points,
        (total - home.position) / total,
        (total - away.position) / total,
    ]


def train_league(api_key: str, league_code: str) -> tuple[list, list]:
    league_id = LEAGUES[league_code]
    print(f"  → {league_code} (id {league_id})")

    standings_data = fetch(api_key, f"competitions/{league_id}/standings")
    table = next(s for s in standings_data["standings"] if s["type"] == "TOTAL")
    stats_by_id = {
        row["team"]["id"]: standing_row_to_stats(row) for row in table["table"]
    }
    total_teams = len(stats_by_id)

    matches_data = fetch(
        api_key,
        f"competitions/{league_id}/matches?status=FINISHED&limit=100",
    )
    X, y = [], []
    for m in matches_data.get("matches", []):
        hid = m["homeTeam"]["id"]
        aid = m["awayTeam"]["id"]
        if hid not in stats_by_id or aid not in stats_by_id:
            continue
        hs = m["score"]["fullTime"]["home"]
        aws = m["score"]["fullTime"]["away"]
        if hs is None or aws is None:
            continue
        home = stats_by_id[hid]
        away = stats_by_id[aid]
        X.append(build_features(home, away, total_teams))
        y.append(match_result(hs, aws))

    print(f"    Partidos usados: {len(y)}")
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar modelo FootballIQ")
    parser.add_argument(
        "--league",
        nargs="+",
        default=list(LEAGUES.keys()),
        choices=list(LEAGUES.keys()),
    )
    args = parser.parse_args()

    api_key = os.environ.get("FOOTBALL_API_KEY", "").strip()
    if not api_key:
        print("Error: definí FOOTBALL_API_KEY con tu key de football-data.org")
        sys.exit(1)

    all_X, all_y = [], []
    for code in args.league:
        try:
            X, y = train_league(api_key, code)
            all_X.extend(X)
            all_y.extend(y)
        except Exception as e:
            print(f"    ⚠ Error en {code}: {e}")

    if len(all_y) < 20:
        print(f"Error: solo {len(all_y)} partidos — se necesitan al menos 20")
        sys.exit(1)

    X_arr = np.array(all_X)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(all_y)

    model = LogisticRegression(max_iter=2000, multi_class="multinomial")
    scores = cross_val_score(model, X_arr, y_enc, cv=min(5, len(set(all_y))), scoring="accuracy")
    model.fit(X_arr, y_enc)

    SKLEARN_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "encoder": encoder}, SKLEARN_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "samples": len(all_y),
                "leagues": args.league,
                "cv_accuracy": round(float(scores.mean()), 3),
                "league_avg_goals": 1.35,
            },
            indent=2,
        )
    )

    print(f"\n✅ Modelo guardado en {SKLEARN_PATH}")
    print(f"   Muestras: {len(all_y)} | Accuracy CV: {scores.mean():.1%}")


if __name__ == "__main__":
    main()
