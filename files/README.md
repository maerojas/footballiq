# ⚽ FootballIQ — Predictor ML

App de análisis de fútbol con **modelo predictivo en Python** (sin IA / sin tokens).

## Arquitectura

```
footballiq/
├── backend/          ← API Python (FastAPI + Poisson + sklearn)
│   ├── app.py        ← Servidor en puerto 8000
│   ├── model.py      ← Modelo predictivo
│   ├── train.py      ← Entrenamiento con datos reales
│   └── requirements.txt
└── files/
    └── index.html    ← Frontend (puerto 8080)
```

## Cómo correrla

### 1. Backend (modelo predictivo)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### 2. (Opcional) Entrenar el modelo ML

Con tu API key de [football-data.org](https://www.football-data.org/client/register):

```bash
cd backend
FOOTBALL_API_KEY=tu_key python train.py
```

Esto entrena una regresión logística con partidos finalizados y la combina con el modelo Poisson.

### 3. Frontend

```bash
cd files
python3 -m http.server 8080
```

Abrí http://localhost:8080, pegá tu API key y listo.

## Qué predice el modelo

| Métrica | Método |
|---------|--------|
| 1X2 (local/empate/visitante) | Distribución de Poisson + ML opcional |
| xG esperado | Fuerza de ataque/defensa por equipo |
| BTTS / Over 2.5 | Suma de probabilidades de marcadores |
| Marcador sugerido | Marcador con mayor probabilidad |
| Análisis textual | Generado localmente desde stats (sin IA) |

## Escudos de equipos

Los escudos vienen directo de la API de football-data.org (`team.crest`) en partidos y tabla.

## Ligas disponibles (plan gratuito)

| Liga | Código |
|------|--------|
| Premier League | PL |
| La Liga | PD |
| Serie A | SA |
| Bundesliga | BL1 |
| Ligue 1 | FL1 |
