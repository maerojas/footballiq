"""
Modelo predictivo de fútbol basado en Poisson + regresión logística.

Usa estadísticas de tabla (goles, puntos, forma) para estimar:
- Probabilidades 1X2
- Goles esperados (xG)
- BTTS y Over 2.5
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
SKLEARN_PATH = MODEL_DIR / "match_classifier.joblib"
META_PATH = MODEL_DIR / "meta.json"

HOME_ADVANTAGE = 1.28
MAX_GOALS = 6


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
        """Puntos de los últimos 5 partidos (W=3, D=1, L=0)."""
        mapping = {"W": 3, "D": 1, "L": 0}
        recent = [f.strip() for f in self.form.split(",") if f.strip()][-5:]
        if not recent:
            return self.ppg * 5 / 3
        return sum(mapping.get(r, 0) for r in recent)


@dataclass
class Prediction:
    home_win: float
    draw: float
    away_win: float
    xg_home: float
    xg_away: float
    xg_total: float
    btts: float
    over25: float
    score_home: int
    score_away: int
    confidence: str
    analysis: dict[str, str]


def _form_to_recent(form: str) -> list[str]:
    return [f.strip() for f in (form or "").split(",") if f.strip()][-5:]


def _build_analysis(home: TeamStats, away: TeamStats, pred: "Prediction") -> dict[str, str]:
    home_form = _form_to_recent(home.form)
    away_form = _form_to_recent(away.form)
    home_wins = home_form.count("W")
    away_wins = away_form.count("W")

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
            ("Victoria local", pred.home_win),
            ("Empate", pred.draw),
            ("Victoria visitante", pred.away_win),
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    prediction_text = (
        f"Resultado más probable: {outcomes[0][0]} ({outcomes[0][1]:.0f}%). "
        f"Marcador sugerido: {pred.score_home}-{pred.score_away}. "
        f"xG total: {pred.xg_total:.1f} goles. "
        f"BTTS: {pred.btts:.0f}% · Over 2.5: {pred.over25:.0f}%."
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
        "prediction": prediction_text,
        "factors": factors,
    }


class FootballPredictor:
    """Modelo híbrido: Poisson para goles + logistic regression opcional para 1X2."""

    def __init__(self) -> None:
        self.classifier: LogisticRegression | None = None
        self.label_encoder = LabelEncoder()
        self.league_avg_goals = 1.35
        self._load_sklearn()

    def _load_sklearn(self) -> None:
        if SKLEARN_PATH.exists():
            bundle = joblib.load(SKLEARN_PATH)
            self.classifier = bundle["model"]
            self.label_encoder = bundle["encoder"]
            meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            self.league_avg_goals = meta.get("league_avg_goals", 1.35)

    def _attack_defense(self, team: TeamStats, league_gf: float, league_ga: float) -> tuple[float, float]:
        attack = (team.gf_per_game / league_gf) if league_gf else 1.0
        defense = (team.ga_per_game / league_ga) if league_ga else 1.0
        return max(0.4, attack), max(0.4, defense)

    def _expected_goals(
        self, home: TeamStats, away: TeamStats, all_teams: list[TeamStats]
    ) -> tuple[float, float]:
        league_gf = np.mean([t.gf_per_game for t in all_teams]) or self.league_avg_goals
        league_ga = np.mean([t.ga_per_game for t in all_teams]) or self.league_avg_goals

        h_att, h_def = self._attack_defense(home, league_gf, league_ga)
        a_att, a_def = self._attack_defense(away, league_gf, league_ga)

        xg_home = h_att * a_def * league_gf * HOME_ADVANTAGE
        xg_away = a_att * h_def * league_gf

        # Ajuste por forma reciente
        form_boost_home = 1 + (home.form_points - away.form_points) * 0.015
        form_boost_away = 1 + (away.form_points - home.form_points) * 0.015
        xg_home *= max(0.85, min(1.15, form_boost_home))
        xg_away *= max(0.85, min(1.15, form_boost_away))

        return max(0.3, xg_home), max(0.3, xg_away)

    def _poisson_outcomes(self, xg_home: float, xg_away: float) -> dict[str, float]:
        home_win = draw = away_win = btts = over25 = 0.0
        score_matrix: list[tuple[int, int, float]] = []

        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                p = poisson.pmf(hg, xg_home) * poisson.pmf(ag, xg_away)
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

    def _feature_vector(
        self, home: TeamStats, away: TeamStats, total_teams: int
    ) -> np.ndarray:
        return np.array(
            [
                home.position / total_teams,
                away.position / total_teams,
                home.ppg,
                away.ppg,
                home.gf_per_game,
                away.gf_per_game,
                home.ga_per_game,
                away.ga_per_game,
                home.form_points,
                away.form_points,
                (total_teams - home.position) / total_teams,
                (total_teams - away.position) / total_teams,
            ]
        ).reshape(1, -1)

    def _blend_with_sklearn(
        self, poisson_probs: dict[str, float], home: TeamStats, away: TeamStats, total_teams: int
    ) -> dict[str, float]:
        if not self.classifier:
            return poisson_probs

        X = self._feature_vector(home, away, total_teams)
        proba = self.classifier.predict_proba(X)[0]
        labels = self.label_encoder.classes_

        ml = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for label, p in zip(labels, proba):
            if label == "H":
                ml["home_win"] = p * 100
            elif label == "D":
                ml["draw"] = p * 100
            elif label == "A":
                ml["away_win"] = p * 100

        # 60% Poisson + 40% ML entrenado
        blend = {}
        for key in ("home_win", "draw", "away_win"):
            blend[key] = 0.6 * poisson_probs[key] + 0.4 * ml[key]
        blend["btts"] = poisson_probs["btts"]
        blend["over25"] = poisson_probs["over25"]
        blend["score_home"] = poisson_probs["score_home"]
        blend["score_away"] = poisson_probs["score_away"]
        return blend

    def predict(
        self,
        home: TeamStats,
        away: TeamStats,
        all_standings: list[TeamStats] | None = None,
    ) -> dict[str, Any]:
        teams = all_standings or [home, away]
        xg_home, xg_away = self._expected_goals(home, away, teams)
        poisson_probs = self._poisson_outcomes(xg_home, xg_away)
        probs = self._blend_with_sklearn(poisson_probs, home, away, len(teams))

        max_prob = max(probs["home_win"], probs["draw"], probs["away_win"])
        if max_prob >= 55:
            confidence = "Alta"
        elif max_prob >= 42:
            confidence = "Media"
        else:
            confidence = "Baja"

        pred = Prediction(
            home_win=probs["home_win"],
            draw=probs["draw"],
            away_win=probs["away_win"],
            xg_home=xg_home,
            xg_away=xg_away,
            xg_total=xg_home + xg_away,
            btts=probs["btts"],
            over25=probs["over25"],
            score_home=probs["score_home"],
            score_away=probs["score_away"],
            confidence=confidence,
            analysis={},
        )
        pred.analysis = _build_analysis(home, away, pred)

        return {
            "win": round(pred.home_win),
            "draw": round(pred.draw),
            "lose": round(pred.away_win),
            "xg": f"{pred.xg_total:.1f}",
            "xg_home": round(float(xg_home), 2),
            "xg_away": round(float(xg_away), 2),
            "btts": round(pred.btts),
            "over25": round(pred.over25),
            "score": f"{pred.score_home}-{pred.score_away}",
            "confidence": pred.confidence,
            "analysis": pred.analysis,
            "model": "poisson+ml" if self.classifier else "poisson",
        }


def standing_row_to_stats(row: dict[str, Any]) -> TeamStats:
    team = row.get("team", {})
    return TeamStats(
        team_id=team.get("id", 0),
        name=team.get("name", "Unknown"),
        position=row.get("position", 10),
        played=row.get("playedGames", 0) or 1,
        won=row.get("won", 0),
        draw=row.get("draw", 0),
        lost=row.get("lost", 0),
        goals_for=row.get("goalsFor", 0),
        goals_against=row.get("goalsAgainst", 0),
        points=row.get("points", 0),
        form=row.get("form", ""),
    )
