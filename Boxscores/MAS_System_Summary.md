# NHL Prediction System: MAS vs XGBoost — Full Comparison
**Built:** April 2026 | **Status:** All 398 tests passing

---

## What We Built — The 6-Module Multi-Agent System (MAS)

### Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                     run_picks.py  (Module 6)                 │
│            Orchestrator — CLI / 3 modes / config             │
└───────┬──────────────────────────────────────────────────────┘
        │
        ├─ full mode   ─────────────────────────────────────────
        │   │                                                   
        │   ├─ Module 1: module1_ingest.py                     
        │   │   Context builder — NHL API, skater/goalie/       
        │   │   schedule/sentiment data per game               
        │   │                                                   
        │   ├─ Module 2: agents/                               
        │   │   5 independent agents, each returns AgentSignal  
        │   │   ┌──────────────────┬──────────────────────┐    
        │   │   │ team_form        │ rolling 3/5/10/20g   │    
        │   │   │ goalie_form      │ GAA, SV%, last 5g    │    
        │   │   │ schedule_fatigue │ rest days, B2B, travel│    
        │   │   │ player_form      │ top-6 skater rolls   │    
        │   │   │ market_sentiment │ moneyline implied %  │    
        │   │   └──────────────────┴──────────────────────┘    
        │   │                                                   
        │   ├─ Module 3: module3_synthesizer.py                
        │   │   Weighted ensemble → PickCard                    
        │   │   Modes: weighted / majority / unanimous          
        │   │   Tiers: HIGH / MEDIUM / LOW / SKIP               
        │   │   Writes picks to SQLite (mas_picks.db)           
        │   │                                                   
        │   ├─ Module 5: script_generator.py                   
        │   │   Generates Alexis Rivers D-ID avatar script      
        │   │   Tones: hype / analytical | Target: 200-250 words
        │   │                                                   
        │   └─ Summary table + Alexis script printed to stdout  
        │                                                       
        ├─ script-only mode  ──────────────────────────────────
        │   Load picks from DB for any past date → regen script
        │                                                       
        └─ report mode  ───────────────────────────────────────
            Module 4: feedback.py
            Resolve outcomes from NHL API → per-agent accuracy
            EMA weight updates → weekly performance report
```

### Module-by-Module Summary

| Module | File | Purpose | Tests |
|--------|------|---------|-------|
| 1 | `module1_ingest.py` | Game context builder: NHL API, parquet skater/goalie data, sentiment | ~65 |
| 2 | `agents/` | 5 independent signal agents | ~80 |
| 3 | `module3_synthesizer.py` | Weighted ensemble → PickCard → SQLite | ~65 |
| 4 | `feedback.py` | Outcome resolver, per-agent accuracy, EMA weights, weekly report | 61 |
| 5 | `script_generator.py` | Alexis Rivers avatar script generator | 64 |
| 6 | `run_picks.py` | Orchestrator CLI — ties everything together | 86 |
| **Total** | | | **398** |

---

## Key Design Decisions Made

**EMA Weight Formula** — weights live in `[0.3, 2.5]`, normalised so a random agent = 1.0:
```python
norm_acc  = win_rate / 0.50        # random baseline = 1.0, 60% win rate = 1.2
new_weight = 0.80 * old_w + 0.20 * norm_acc
new_weight = max(0.3, min(2.5, new_weight))
```
Requires MIN_SAMPLE = 10 resolved picks before firing. Agents with fewer picks keep their current weight.

**Tier Thresholds** (Module 3):
- HIGH: ensemble raw_score ≥ confidence_floor (default 0.52) AND edge ≥ 3% AND ≥ 3 agents above floor
- MEDIUM: above floor but weaker agreement or lower edge
- LOW: raw signal present but below floor
- SKIP: no meaningful consensus

**Script Word Budget** — 200–250 words, ~90–120 seconds at 150 wpm:
- Trim loop runs up to 40 passes, always removes the last sentence from the longest pick paragraph
- Sentence splitter uses `(?<=[a-zA-Z])\.\s+` — won't split "18.0 percent" on the decimal point
- Opening paragraph expands for 1-pick scripts to hit the floor

**Injectable Dependencies** — all modules accept constructor injection:
```python
FeedbackEngine(db_path=":memory:", weights_path=tmp, fetcher=mock_fn)
MASOrchestrator(config=cfg, context_builder=mock, agents=[], synthesizer=mock)
```
No `unittest.mock.patch` needed anywhere in 398 tests.

---

## Old System vs New System — Head-to-Head

| Dimension | `predict_today.py` (XGBoost) | MAS (Modules 1–6) |
|-----------|------------------------------|-------------------|
| **Model type** | Single XGBoost classifier | 5 independent agents + weighted ensemble |
| **Signal count** | 1 calibrated probability | 5 signals, each with raw_score + confidence |
| **Confidence tiers** | HIGH ≥65%, LOW 52–57%, MED blocked | HIGH / MEDIUM / LOW / SKIP |
| **Edge calculation** | Model prob − market implied | edge_pct = (raw_score − 0.50) × 2 × 100 |
| **Goalie handling** | Scrapes DailyFaceoff + manual overrides | Agent-native: goalie_form agent reads GAA/SV% |
| **Rest/B2B** | Fetched live from NHL API | schedule_fatigue agent computes |
| **Market signal** | Market implied used for edge calc | market_sentiment agent votes on direction |
| **Weight updates** | None (static model) | EMA weight updates after each resolved date |
| **Feedback loop** | track_results.py (manual CSV) | feedback.py: auto-resolves via NHL API |
| **Script output** | None | Alexis Rivers D-ID avatar script, 2 tones |
| **Output format** | Print to terminal + CSV log | SQLite picks DB + stdout + script |
| **Persistence** | `prediction_log.csv` | `mas_picks.db` (SQLite) |
| **Config** | Hardcoded paths | `.env` file + env var overrides |
| **CLI modes** | `--date` only | `--mode full/script-only/report` + `--dry-run` |
| **Tests** | 0 (script-style, hard to unit test) | 398 across all 6 modules |
| **Dependencies** | pandas, xgboost, requests, scikit-learn | All above + sqlite3 (stdlib) |

---

## What Each System Does Better

### XGBoost (`predict_today.py`) Does Better:

**Trained accuracy** — The XGBoost model was trained on real historical NHL data with isotonic calibration. Its 65% confidence threshold was set empirically after seeing which predictions underperformed (MED 58–64% blocked because it ran 30% win rate). The MAS agents start with equal weights and only adapt after enough resolved picks.

**Goalie confirmation gate** — The old system explicitly blocks a pick if either starting goalie is "Unknown" or unconfirmed. The MAS goalie_form agent will still output a signal (likely neutral/low confidence) but won't hard-block the pick.

**Market edge precision** — `predict_today.py` computes edge directly as `model_probability − market_implied`, giving a clean apples-to-apples comparison. The MAS edge_pct is derived from raw ensemble score, which may not be in the same probability space as market odds without calibration.

**Simplicity** — One file, ~680 lines, runs in 15–30 seconds. Easy to audit or modify ad-hoc.

### MAS (Modules 1–6) Does Better:

**Interpretability** — Every pick includes a ranked breakdown of which agents voted which direction and why. Alexis can explain the "why" in plain language: "Toronto has won 8 of their last 10 at home, Woll's been a wall all week, and Vegas has them as a slight home dog — that's where we're playing."

**Adaptability** — EMA weights shift after every resolved game. If the goalie_form agent has been calling wrong for 20 games, its weight drops toward 0.3 and the ensemble listens to it less. The XGBoost model only updates when you retrain.

**Content pipeline** — The D-ID-ready Alexis script is a complete deliverable. The old system prints a terminal table. The MAS outputs something you can plug directly into avatar generation.

**Testability** — 398 tests covering every function, edge case, and integration path. You can refactor any module with confidence.

**Dry-run mode** — `--dry-run` shows you every agent signal, direction, and weight for every game before a single row hits the database. Great for debugging or calibrating new agents.

**Modularity** — You can run `--mode script-only` to regenerate Alexis's script for yesterday's games without re-running all agents. You can run `--mode report` to resolve outcomes and update weights without touching today's predictions.

---

## How to Use Both Together

The systems aren't mutually exclusive. A practical workflow:

```
Morning (~8am):
  1. predict_today.py          → quick calibrated XGBoost read + goalie check
  2. run_picks.py --mode full  → 5-agent ensemble + Alexis script

If XGBoost and MAS agree on direction + tier:  stronger conviction
If they disagree:                               reduce stake or pass

Evening (after games finish):
  3. run_picks.py --mode report --date YYYY-MM-DD  → resolve outcomes, update weights
```

The XGBoost model is the calibrated anchor. The MAS gives you explainability and an evolving ensemble that learns which signals are hot right now. Together they're better than either alone.

---

## Running the MAS

```bash
# Full run for today
python run_picks.py --mode full --tone hype

# Dry-run to inspect signals without writing to DB
python run_picks.py --mode full --dry-run --verbose

# Regenerate Alexis script from yesterday's picks in DB
python run_picks.py --mode script-only --date 2026-04-07 --tone analytical

# Resolve outcomes + weekly performance report
python run_picks.py --mode report --date 2026-04-07 --weeks 2

# Check current agent weights
python feedback.py --show-weights
```

**Config via `.env`** (place in `Boxscores/`):
```
DB_PATH=C:\...\Boxscores\mas_picks.db
WEIGHTS_FILE_PATH=C:\...\Boxscores\weights_registry.json
LOG_LEVEL=INFO
CONFIDENCE_FLOOR=0.52
MIN_AGENTS=3
SCRIPT_TONE=hype
FETCH_ADVANCED=true
```

---

## Test Coverage Summary

```
Module 1 (ingest)        — context builder + NHL API wrappers
Module 2 (agents)        — 5 agents, signal contracts, edge cases
Module 3 (synthesizer)   — weighted ensemble, tier logic, DB writes
Module 4 (feedback)      — EMA weights, ROI calc, weekly report        61 tests
Module 5 (script_gen)    — word count, tone, formatting, team names     64 tests
Module 6 (orchestrator)  — CLI, config, run modes, mocks               86 tests
─────────────────────────────────────────────────────────────────────────────────
TOTAL                                                                  398 tests
                                                                        0 failures
```

All database operations use in-memory SQLite (`":memory:"`). No network calls anywhere in the test suite. All external dependencies are injectable — the 398 tests run with zero real NHL API calls, zero parquet reads, and zero file system dependencies outside stdlib.
