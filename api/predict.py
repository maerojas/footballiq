"""Vercel serverless: modelo Poisson (stdlib only)."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

HOME_ADVANTAGE = 1.28
MAX_GOALS = 6
LEAGUE_AVG_GOALS = 1.35


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


@dataclass
class TeamStats:
    team_id: int
    name: str
    position: int
    played: int
    won: int
    draw: int
    lost: int
    goals_for: int
    goals_against: int
    points: int
    form: str = ""

    @property
    def ppg(self) -> float:
        return self.points / self.played if self.played else 0.0

    @property
    def gf_per_game(self) -> float:
        return self.goals_for / self.played if self.played else 1.0

    @property
    def ga_per_game(self) -> float:
        return self.goals_against / self.played if self.played else 1.0

    @property
    def form_points(self) -> float:
        mapping = {"W": 3, "D": 1, "L": 0}
        recent = [f.strip() for f in (self.form or "").split(",") if f.strip()][-5:]
        if not recent:
            return self.ppg * 5 / 3
        return sum(mapping.get(r, 0) for r in recent)


def standing_row_to_stats(row: dict[str, Any]) -> TeamStats:
    team = row.get("team", {})
    played = row.get("playedGames", 0) or 1
    return TeamStats(
        team_id=team.get("id", 0),
        name=team.get("name", "Unknown"),
        position=row.get("position", 10),
        played=played,
        won=row.get("won", 0),
        draw=row.get("draw", 0),
        lost=row.get("lost", 0),
        goals_for=row.get("goalsFor", 0),
        goals_against=row.get("goalsAgainst", 0),
        points=row.get("points", 0),
        form=row.get("form") or "",
    )


def _form_to_recent(form: str) -> list[str]:
    return [f.strip() for f in (form or "").split(",") if f.strip()][-5:]


def _attack_defense(team: TeamStats, league_gf: float, league_ga: float) -> tuple[float, float]:
    attack = (team.gf_per_game / league_gf) if league_gf else 1.0
    defense = (team.ga_per_game / league_ga) if league_ga else 1.0
    return max(0.4, attack), max(0.4, defense)


def _expected_goals(home: TeamStats, away: TeamStats, all_teams: list[TeamStats]) -> tuple[float, float]:
    league_gf = statistics.mean([t.gf_per_game for t in all_teams]) if all_teams else LEAGUE_AVG_GOALS
    league_ga = statistics.mean([t.ga_per_game for t in all_teams]) if all_teams else LEAGUE_AVG_GOALS

    h_att, h_def = _attack_defense(home, league_gf, league_ga)
    a_att, a_def = _attack_defense(away, league_gf, league_ga)

    xg_home = h_att * a_def * league_gf * HOME_ADVANTAGE
    xg_away = a_att * h_def * league_gf

    form_boost_home = 1 + (home.form_points - away.form_points) * 0.015
    form_boost_away = 1 + (away.form_points - home.form_points) * 0.015
    xg_home *= max(0.85, min(1.15, form_boost_home))
    xg_away *= max(0.85, min(1.15, form_boost_away))

    return max(0.3, xg_home), max(0.3, xg_away)


def _poisson_outcomes(xg_home: float, xg_away: float) -> dict[str, float | int]:
    home_win = draw = away_win = btts = over25 = 0.0
    score_matrix: list[tuple[int, int, float]] = []

    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = poisson_pmf(hg, xg_home) * poisson_pmf(ag, xg_away)
            score_matrix.append((hg, ag, p))
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p
            if hg > 0 and ag > 0:
                btts += p
            if hg + ag > 2:
                over25 += p

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total

    best = max(score_matrix, key=lambda x: x[2])
    return {
        "home_win": home_win * 100,
        "draw": draw * 100,
        "away_win": away_win * 100,
        "btts": btts * 100,
        "over25": over25 * 100,
        "score_home": best[0],
        "score_away": best[1],
    }


def _build_analysis(home: TeamStats, away: TeamStats, probs: dict[str, Any]) -> dict[str, Any]:
    home_form = _form_to_recent(home.form or "")
    away_form = _form_to_recent(away.form or "")
    home_wins = home_form.count("W")
    away_wins = away_form.count("W")
    xg_total = probs["xg_home"] + probs["xg_away"]

    context = (
        f"{home.name} ocupa la posición {home.position} con {home.points} puntos "
        f"({home.won}V-{home.draw}E-{home.lost}D). "
        f"{away.name} va {away.position}° con {away.points} pts. "
        f"El local tiene ventaja de campo (+{int((HOME_ADVANTAGE - 1) * 100)}% en ataque)."
    )

    strengths = []
    if home.gf_per_game > away.gf_per_game:
        strengths.append(f"{home.name} anota más ({home.gf_per_game:.2f} vs {away.gf_per_game:.2f} GF/partido).")
    else:
        strengths.append(f"{away.name} tiene mejor producción ofensiva ({away.gf_per_game:.2f} GF/partido).")
    if home.ga_per_game < away.ga_per_game:
        strengths.append(f"{home.name} es más sólido defensivamente ({home.ga_per_game:.2f} GC/partido).")
    else:
        strengths.append(f"{away.name} concede menos goles ({away.ga_per_game:.2f} GC/partido).")
    if home_wins > away_wins:
        strengths.append(f"Forma reciente favorable al local ({home_wins} victorias en 5).")
    elif away_wins > home_wins:
        strengths.append(f"{away.name} llega en mejor racha ({away_wins}V en últimos 5).")

    outcomes = sorted(
        [
            ("Victoria local", probs["home_win"]),
            ("Empate", probs["draw"]),
            ("Victoria visitante", probs["away_win"]),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    prediction = (
        f"Resultado más probable: {outcomes[0][0]} ({outcomes[0][1]:.0f}%). "
        f"Marcador sugerido: {probs['score_home']}-{probs['score_away']}. "
        f"xG total: {xg_total:.1f} goles. "
        f"BTTS: {probs['btts']:.0f}% · Over 2.5: {probs['over25']:.0f}%."
    )

    factors = [
        f"Diferencia en tabla: {abs(home.position - away.position)} puestos",
        f"Ataque local vs defensa visitante: {home.gf_per_game:.2f} vs {away.ga_per_game:.2f}",
        f"Puntos por partido: {home.ppg:.2f} (local) vs {away.ppg:.2f} (visitante)",
        f"Forma reciente: {'-'.join(home_form) or 'N/D'} vs {'-'.join(away_form) or 'N/D'}",
    ]

    return {
        "context": context,
        "strengths": " ".join(strengths),
        "prediction": prediction,
        "factors": factors,
    }


def run_prediction(home_id: int, away_id: int, standings: list[dict[str, Any]]) -> dict[str, Any]:
    teams = [standing_row_to_stats(row) for row in standings]
    home = next((t for t in teams if t.team_id == home_id), None)
    away = next((t for t in teams if t.team_id == away_id), None)
    if not home or not away:
        raise ValueError("Equipos no encontrados en la tabla de standings")

    xg_home, xg_away = _expected_goals(home, away, teams)
    poisson = _poisson_outcomes(xg_home, xg_away)

    max_prob = max(poisson["home_win"], poisson["draw"], poisson["away_win"])
    if max_prob >= 55:
        confidence = "Alta"
    elif max_prob >= 42:
        confidence = "Media"
    else:
        confidence = "Baja"

    analysis = _build_analysis(
        home,
        away,
        {
            **poisson,
            "xg_home": xg_home,
            "xg_away": xg_away,
        },
    )

    return {
        "win": round(float(poisson["home_win"])),
        "draw": round(float(poisson["draw"])),
        "lose": round(float(poisson["away_win"])),
        "xg": f"{xg_home + xg_away:.1f}",
        "btts": round(float(poisson["btts"])),
        "over25": round(float(poisson["over25"])),
        "score": f"{poisson['score_home']}-{poisson['score_away']}",
        "confidence": confidence,
        "analysis": analysis,
    }


class handler(BaseHTTPRequestHandler):
    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_standings(self) -> list[dict[str, Any]]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"[]"
        return json.loads(raw.decode("utf-8") or "[]")

    def _parse_ids(self) -> tuple[int, int]:
        qs = parse_qs(urlparse(self.path).query)
        home_id = int(qs.get("home_id", [""])[0])
        away_id = int(qs.get("away_id", [""])[0])
        return home_id, away_id

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def _handle(self) -> None:
        try:
            home_id, away_id = self._parse_ids()
            standings = self._read_standings()
            result = run_prediction(home_id, away_id, standings)
            self._respond(200, result)
        except Exception as exc:
            self._respond(400, {"error": str(exc)})
