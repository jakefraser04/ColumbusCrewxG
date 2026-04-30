# Columbus Crew xG Dashboard — Multi-Season Analysis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-222222?style=flat-square&logo=github&logoColor=white)](https://jakefraser04.github.io/Crew2025xG/)

**[→ View Live Dashboard](https://jakefraser04.github.io/ColumbusCrewxG/)**

---

## Background

The Columbus Crew are a Major League Soccer team located in Columbus, Ohio. This project showcases data wrangling and visualization skills by analyzing whether the team's match results reflect their performance quality using Expected Goals (xG).

Expected Goals (xG) cuts through scorelines and tells you which results a team *earned* and which they *got away with*. This project builds an xG model from scratch and applies it to Columbus Crew matches from 2025 to the present.

---

## Project Structure

```
ColumbusCrewxG/
├── assets/                 # Generated visuals for README and reports
│   ├── 2025/               # Historical season visualizations
│   └── 2026/               # Current season visualizations
├── dashboard/              # Interactive Web Interface
│   └── data/               # JSON endpoints feeding the dashboard
├── data/
│   ├── raw/                # (Gitignored) Source match data
│   └── processed/          # Cleaned datasets for modeling
├── models/                 # Production-ready model & metadata
│   ├── model_meta.json     # Performance metrics (AUC, Log-Loss)
│   └── xg_model.pkl        # Serialized Logistic Regression model
├── notebooks/              # Research & Development
│   ├── crew_xG_analysis.ipynb
│   └── xG_model.ipynb
├── index.html              # Dashboard Entry Point
├── update_season.py        # Automated Data Pipeline Script
├── requirements.txt        # Project Dependencies
└── README.md
```

---

## Automation & Updates

The dashboard is designed to be updated after every match using the `update_season.py` script.

**To update a season:**
```bash
python update_season.py --season 2026 --push
```
This script:
1. Pulls the latest match data from the American Soccer Analysis API.
2. Applies the trained xG model to the new matches.
3. Regenerates all season-specific charts in `assets/`.
4. Exports updated JSON data to `dashboard/data/`.
5. Commits and pushes the changes to GitHub.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | Python · pandas · NumPy · scikit-learn · joblib |
| Visualization | matplotlib · mplsoccer |
| Data | StatsBomb open data · American Soccer Analysis API |
| Frontend | Vanilla JavaScript · HTML/CSS |
| Hosting | GitHub Pages |

---

## About the xG Model

This model uses logistic regression trained on StatsBomb shot event data with the following features: distance to goal, shot angle, body part, defensive pressure, and play pattern context.

A shot with xG = 0.20 means shots taken from that location and context score approximately 20% of the time historically.

---

## Author

**Jake Fraser** · [github.com/jakefraser04](https://github.com/jakefraser04)
