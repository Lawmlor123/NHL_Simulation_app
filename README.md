🏒 NHL Predictive Simulation App
This Streamlit‑based application serves as the interactive front end to a custom XGBoost‑powered predictive model that forecasts daily NHL game outcomes. It automatically loads the day’s schedule, runs each matchup through a machine‑learning pipeline built from multi‑season team statistics, and displays projected win probabilities in a clean, data‑driven dashboard. Hockey enthusiasts can browse all games at a glance or click into individual matchups for deeper insights such as comparative team form, recent performance, and historical context. Designed for the experienced fan who values analytics, this tool blends real‑time data with machine learning to provide an engaging, stats‑focused perspective on each day’s NHL slate.

# NHL Quant Modeling – Data & Model Overview

_Last updated: 2026‑01‑26_

---

## 1. Feature Inventory

### 1.1 Core Game Fields (from `CLEAN_ALL_GAMES_MASTER.py`)
| Field | Description | Notes |
| --- | --- | --- |
| `game_id`, `gameid` | Source-provided identifiers | ⚠️ TBD: confirm if both needed |
| `date` | Game date |  |
| `hometeam`, `awayteam` | Raw team strings | Normalized to `home_key`, `away_key` |
| `home_key`, `away_key` | Canonical 3-letter codes | 35 unique keys detected |
| `hometeamscore`, `awayteamscore` | Final scores | Used to derive `target` |
| `status`, `isclosed`, `played` | Game completion flags | `played=True` when scores present |
| `target` | Home-win indicator | 1 = home win, 0 = home loss |
| `season`, `seasontype` | Context fields |  |
| `source`, `team` | Origin + extra column | ⚠️ TBD: clarify `team` usage |

### 1.2 Builder Metadata & Betting Inputs (`builder.py`, SportsData feed)
- Moneylines: `AwayTeamMoneyLine`, `HomeTeamMoneyLine`
- Spread & totals: `PointSpread`, `OverUnder`, `Over/Under Payouts`, spread moneylines
- Attendance, broadcast channel, stadium IDs
- Period/clock tracking, `SeriesInfo`, `LastPlay`
- Rotation numbers, `NeutralVenue`
- Multiple ID systems (`GlobalGameID`, etc.)

### 1.3 Seasonal Team Aggregates (HockeyReference CSVs)
| Category | Fields |
| --- | --- |
| Standings | `W`, `L`, `OL`, `PTS`, `PTS%`, `SRS`, `SOS` |
| Scoring | `GF`, `GA`, `GF/G`, `GA/G` |
| Special Teams | `PP`, `PPO`, `PP%`, `PPA`, `PPOA`, `PK%` |
| Shooting/Defense | `S`, `S%`, `SA`, `SV%`, `SO` |
| Discipline | `PIM/G`, `oPIM/G` |
| Misc | `AvAge`, `Source`, `Season` |

### 1.4 Engineered Features (`feature_assembler.py`, `phase_3_gm.py`)
| Feature Group | Fields | Notes |
| --- | --- | --- |
| Differential stats | `goal_diff`, `wins_diff`, `losses_diff`, `goals_for_diff`, `goals_against_diff`, `rolling_goals_for_diff`, `rolling_goals_against_diff` | Derived from normalized master dataset |
| Rest / recent games | ⚠️ TBD – counts per window (log indicates rolling counts but window length not documented) | Need explicit definitions |
| Odds-based features | _Not currently merged_ | Awaiting odds file availability |

---

## 2. Model Performance

### 2.1 Training & Validation
- **Training rows:** 1,262 labeled games (home win rate 58.2%).
- **Validation scheme:** Most recent 300 played games (adaptive window ensuring ≥30 positives).
- **Latest validation metrics:**
  - Accuracy: **0.633**
  - ROC-AUC: **0.543**
  - Class distribution: 0 = 107 (35.7%), 1 = 193 (64.3%)

### 2.2 Post-hoc Evaluation (`Modeleffective.py`, games through 2026‑01‑25)
| Segment | n | Accuracy | Brier |
| --- | --- | --- | --- |
| All predictions | 206 | 0.529 | 0.2704 |
| Old model (< 2025‑12‑09) | 24 | 0.542 | 0.3954 |
| New model (≥ 2025‑12‑09) | 182 | 0.527 | 0.2539 |

### 2.3 Baselines & Diagnostics
- Baseline comparisons: ⚠️ TBD (coin flip, always-home, bookmaker implied odds).
- Failure modes / error analysis: ⚠️ Not yet documented.

---

## 3. Data Sources & Pipeline

| Stage | Script / File | Description | Output |
| --- | --- | --- | --- |
| Data collection | `phase_3_dc.py` | Loads `nhl_20XX_team_stats.csv` (2021‑2025) + current season 2025 stats | `combined_all_teamstats_history`, `teamseason_features_2025`, `games_YYYY-MM-DD` |
| Master builder | `builder.py` | Merges SportsData feed (2,683 rows) with master schedule (1,312 rows) | `all_games_master_<timestamp>.csv` (3,995 rows) |
| Cleaner | `phase_3_cm.py` | Normalizes teams, drops malformed rows, splits played/upcoming | `all_games_master_clean.csv`, `all_games_played_clean.csv`, `all_games_upcoming.csv`, `unmapped_teams.csv` |
| Normalizer & diffs | `phase_3_gm.py` | Consolidates 15,316 rows across history, creates differential features | Updated `all_games_master.csv` + clean sync |
| Feature assembler | `feature_assembler.py` | Adds rest features & (when available) odds; outputs modeling table | `games_features_v1.csv` |
| Prediction & odds | `phase_5_prediction_xgb.py` | Trains XGB, validates, pulls Odds API data, generates daily predictions | `quant_nhl_xgb_diff.pkl`, validation/pred CSVs, `odds_snapshots.csv` |
| Effectiveness tracking | `Modeleffective.py` | Merges predictions with ESPN results for rolling KPIs | Accuracy/Brier summary |

### External Feeds
- **SportsData** – schedule + betting lines (builder input).
- **Master schedule CSV** – supplemental schedule info.
- **HockeyReference** – season team stats (2021–2025).
- **The Odds API** – daily H2H odds (prediction stage).
- **ESPN results** – actual outcomes for evaluation.

---

### Open Items / Next Actions
1. Document exact definitions for rest/rolling features (window lengths, formulas).  
2. Capture baseline comparisons and any qualitative failure notes.  
3. Confirm whether additional data sources (injuries, goalie starters, etc.) are planned; add to catalog when available.

---

_This document should be updated whenever new features, data feeds, or evaluation metrics are introduced._
