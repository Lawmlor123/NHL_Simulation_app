"""
module3_synthesizer.py
----------------------
Module 3: Master Synthesizer — NHL MAS meta-learner.

Entry point:
    card = synthesizer.synthesize(signals, game_context)

Two synthesis modes:
    1. weighted_avg  — weighted mean of agent raw_scores + sentiment multiplier
    2. meta_learner  — trained sklearn LogisticRegression on historical signals

SKIP logic:
    Fewer than `min_agents` (default 3) agents above confidence_floor → SKIP.

SQLite log:
    Every PickCard is written to mas_picks.db.
    Outcomes can be added later (by track_results.py) to enable meta-learner training
    and weight recalibration.

Weights registry:
    {agent_id: float} — start from DEFAULT_WEIGHTS, updated by update_weights()
    after each batch of resolved outcomes.
"""

import json
import math
import sqlite3
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

# sklearn — required for meta-learner mode only
try:
    from sklearn.linear_model  import LogisticRegression
    from sklearn.calibration   import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from agents.agent_base import AgentSignal

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
DB_PATH = BASE / "mas_picks.db"

# ── Constants ─────────────────────────────────────────────────────────────────
AGENT_ORDER = ["team_form", "player_form", "goalie_form", "schedule", "sentiment"]

DEFAULT_WEIGHTS: dict[str, float] = {
    "team_form":   1.00,
    "player_form": 0.90,
    "goalie_form": 1.20,   # highest — goalies drive outcomes
    "schedule":    0.80,
    "sentiment":   0.50,
}

# Confidence tier thresholds (match existing predict_today.py thresholds)
TIER_HIGH   = 0.65
TIER_MEDIUM = 0.58
TIER_LOW    = 0.52

# Sentinel for SKIP confidence
SKIP_SCORE = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# PickCard — Module 3 output contract
# ═══════════════════════════════════════════════════════════════════════════════

ConfidenceTier = Literal["HIGH", "MEDIUM", "LOW", "SKIP"]


@dataclass
class PickCard:
    # ── Game identity ─────────────────────────────────────────────────
    game_id:    int   = 0
    game_date:  str   = ""
    home_team:  str   = ""
    away_team:  str   = ""

    # ── Pick ──────────────────────────────────────────────────────────
    pick:           str = "SKIP"       # team abbreviation or "SKIP"
    pick_direction: str = "skip"       # "home" / "away" / "skip"

    # ── Scores ────────────────────────────────────────────────────────
    raw_score: float = 0.50            # 0–1, >0.5 = home lean
    edge_pct:  float = 0.0             # |raw_score − 0.5| × 100

    # ── Tier & metadata ───────────────────────────────────────────────
    confidence_tier:      ConfidenceTier = "SKIP"
    agent_agreement_score: float         = 0.0   # 0–1 (1.0 = unanimous)
    agents_above_floor:    int           = 0
    mode_used:             str           = "weighted_avg"

    # ── Agent breakdown (for SQLite log & downstream modules) ─────────
    signal_summary: dict = field(default_factory=dict)
    # {agent_id: {raw_score, confidence, direction, reasoning, weight}}
    weights_used:   dict = field(default_factory=dict)

    # ── Skip reason (populated when confidence_tier == SKIP) ──────────
    skip_reason: str = ""

    # ── Timestamp ────────────────────────────────────────────────────
    produced_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Derived helpers ───────────────────────────────────────────────

    @property
    def is_playable(self) -> bool:
        return self.confidence_tier != "SKIP"

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    def tier_emoji(self) -> str:
        return {"HIGH": "★★★", "MEDIUM": "★★ ", "LOW": "★  ", "SKIP": "⛔ "}.get(
            self.confidence_tier, "   ")

    def __str__(self) -> str:
        if self.confidence_tier == "SKIP":
            return (f"[SKIP] {self.matchup}  ({self.skip_reason})")
        direction = "HOME" if self.pick_direction == "home" else "AWAY"
        return (
            f"{self.tier_emoji()} {self.matchup:<15s}  "
            f"→ {self.pick} ({direction})  "
            f"edge={self.edge_pct:+.1f}%  "
            f"agree={self.agent_agreement_score:.0%}  "
            f"[{self.mode_used}]"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite schema
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT    NOT NULL,
    game_id               INTEGER,
    game_date             TEXT,
    home_team             TEXT,
    away_team             TEXT,
    pick                  TEXT,
    pick_direction        TEXT,
    raw_score             REAL,
    edge_pct              REAL,
    confidence_tier       TEXT,
    agent_agreement       REAL,
    agents_above_floor    INTEGER,
    mode_used             TEXT,
    skip_reason           TEXT,
    signals_json          TEXT,
    weights_json          TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     INTEGER UNIQUE,
    game_date   TEXT,
    home_team   TEXT,
    away_team   TEXT,
    home_win    INTEGER,
    added_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_picks_date    ON picks(game_date);
CREATE INDEX IF NOT EXISTS idx_picks_game    ON picks(game_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_game ON outcomes(game_id);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering (shared by fit() and meta-learner predict)
# ═══════════════════════════════════════════════════════════════════════════════

def _signals_to_feature_vector(signals: list[AgentSignal]) -> list[float]:
    """
    Build a fixed-length feature vector from a list of AgentSignals.
    Order: [agent1_raw, agent1_conf, agent2_raw, agent2_conf, ...]
    Missing agents → (0.5, 0.0) — neutral signal, zero confidence.
    Result length: len(AGENT_ORDER) × 2 = 10 features.
    """
    sig_map = {s.agent_id: s for s in signals}
    feats   = []
    for aid in AGENT_ORDER:
        s = sig_map.get(aid)
        if s is not None:
            feats.extend([s.raw_score, s.confidence])
        else:
            feats.extend([0.5, 0.0])
    return feats


def _build_feature_matrix(
    all_signals: list[list[AgentSignal]],
) -> np.ndarray:
    """Convert list-of-signal-lists to numpy feature matrix (n_games × 10)."""
    return np.array([_signals_to_feature_vector(sigs) for sigs in all_signals],
                    dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# MasterSynthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class MasterSynthesizer:
    """
    Master Synthesizer — combines all 5 agent signals into a final PickCard.

    Args:
        weights_registry:  {agent_id: weight} — updated by update_weights()
        confidence_floor:  Minimum agent confidence to count toward ensemble
        min_agents:        Minimum agents above floor before issuing a pick
        db_path:           SQLite path for logging (None = in-memory only)
    """

    def __init__(
        self,
        weights_registry: Optional[dict] = None,
        confidence_floor: float          = 0.52,
        min_agents:       int            = 3,
        db_path:          Optional[Path] = None,
    ):
        self.weights_registry = dict(weights_registry or DEFAULT_WEIGHTS)
        self.confidence_floor = confidence_floor
        self.min_agents       = min_agents
        self._db_path         = db_path or DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

        # Meta-learner (populated by fit())
        self._model:     Optional[Pipeline] = None
        self._is_fitted: bool               = False

    # ── DB connection (lazy) ──────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = _init_db(self._db_path)
            except Exception:
                # Fallback to in-memory DB if path not writable (e.g. sandbox)
                self._conn = sqlite3.connect(":memory:")
                self._conn.executescript(_SCHEMA)
                self._conn.commit()
        return self._conn

    # ── Public API ────────────────────────────────────────────────────

    def synthesize(
        self,
        signals:      list[AgentSignal],
        game_context  = None,
        mode:         str = "auto",
        log:          bool = True,
    ) -> PickCard:
        """
        Main entry point. Returns a PickCard for one game.

        mode options:
            "auto"         — use meta_learner if fitted, else weighted_avg
            "weighted_avg" — always use weighted average
            "meta_learner" — always use meta-learner (must call fit() first)
        """
        effective_mode = mode
        if mode == "auto":
            effective_mode = "meta_learner" if self._is_fitted else "weighted_avg"

        if effective_mode == "meta_learner":
            if not self._is_fitted:
                warnings.warn("Meta-learner not fitted — falling back to weighted_avg")
                effective_mode = "weighted_avg"

        if effective_mode == "meta_learner":
            card = self._synthesize_meta(signals, game_context)
        else:
            card = self._synthesize_weighted(signals, game_context)

        if log:
            self._log_pick(card)

        return card

    def fit(
        self,
        historical_signals: list[list[AgentSignal]],
        outcomes:           list[int],
    ) -> "MasterSynthesizer":
        """
        Train the meta-learner on historical signal data.

        Args:
            historical_signals: List of signal lists, one per game
            outcomes:           1 = home won, 0 = away won, per game

        Returns:
            self (for chaining)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for meta-learner mode. "
                              "Run: pip install scikit-learn")

        X = _build_feature_matrix(historical_signals)
        y = np.array(outcomes, dtype=int)

        if len(X) < 20:
            warnings.warn(f"Only {len(X)} training samples — meta-learner may be unreliable. "
                          "Recommend 100+ games for stable weights.")

        # Pipeline: scale features → logistic regression → isotonic calibration
        base_lr = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(base_lr, cv=min(5, len(X) // 5 or 2),
                                               method="isotonic")),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X, y)

        self._is_fitted = True
        print(f"  ✅ Meta-learner fitted on {len(X)} games")
        return self

    def fit_from_db(self) -> "MasterSynthesizer":
        """
        Train the meta-learner using picks + outcomes already in the SQLite DB.
        Only uses rows where both a pick AND outcome exist for the same game_id.
        """
        conn = self._get_conn()

        query = """
            SELECT p.game_id, p.signals_json, o.home_win
            FROM picks p
            JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.confidence_tier != 'SKIP'
        """
        rows = conn.execute(query).fetchall()
        if not rows:
            print("  ⚠  No resolved picks in DB — meta-learner not trained")
            return self

        historical_signals = []
        outcomes = []

        for game_id, signals_json, home_win in rows:
            try:
                sig_dicts = json.loads(signals_json)
                # Reconstruct minimal AgentSignal objects for feature building
                signals = [
                    AgentSignal(
                        agent_id       = aid,
                        pick_direction = v.get("direction", "neutral"),
                        raw_score      = float(v.get("raw_score", 0.5)),
                        confidence     = float(v.get("confidence", 0.0)),
                        reasoning      = "",
                    )
                    for aid, v in sig_dicts.items()
                ]
                historical_signals.append(signals)
                outcomes.append(int(home_win))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if historical_signals:
            n_classes = len(set(outcomes))
            if n_classes < 2:
                print(f"  ⚠  Only 1 outcome class in {len(outcomes)} resolved games "
                      f"— skipping meta-learner fit (need both wins and losses)")
                return self
            try:
                self.fit(historical_signals, outcomes)
            except Exception as exc:
                print(f"  ⚠  Meta-learner fit failed ({exc}) — continuing without it")
        return self

    def update_weights(self) -> dict:
        """
        Recalibrate agent weights based on historical accuracy in the DB.

        Method:
            For each agent, compute the correlation between its raw_score
            direction (>0.5 = home pick) and actual outcome (home_win).
            Agents with higher correlation get higher weights.

        Returns:
            Updated weights_registry dict.
        """
        conn = self._get_conn()

        query = """
            SELECT p.signals_json, o.home_win
            FROM picks p
            JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.confidence_tier != 'SKIP'
        """
        rows = conn.execute(query).fetchall()
        if len(rows) < 10:
            print(f"  ⚠  Only {len(rows)} resolved games — weights not updated (need 10+)")
            return self.weights_registry

        # Per-agent: collect (predicted_direction, actual_outcome)
        agent_correct: dict[str, list[int]] = {aid: [] for aid in AGENT_ORDER}

        for signals_json, home_win in rows:
            try:
                sig_dicts = json.loads(signals_json)
                for aid in AGENT_ORDER:
                    if aid in sig_dicts:
                        raw = float(sig_dicts[aid].get("raw_score", 0.5))
                        conf = float(sig_dicts[aid].get("confidence", 0.0))
                        if conf >= self.confidence_floor:
                            predicted_home = int(raw > 0.5)
                            correct = int(predicted_home == int(home_win))
                            agent_correct[aid].append(correct)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        updated = {}
        for aid in AGENT_ORDER:
            outcomes_for_agent = agent_correct[aid]
            if len(outcomes_for_agent) < 5:
                updated[aid] = self.weights_registry.get(aid, DEFAULT_WEIGHTS.get(aid, 1.0))
                continue

            accuracy = sum(outcomes_for_agent) / len(outcomes_for_agent)
            n        = len(outcomes_for_agent)

            # Weight = accuracy / 0.5 (normalized so random = 1.0)
            # Clamp to [0.3, 2.5] to avoid runaway weights
            new_weight = max(0.30, min(2.50, accuracy / 0.5))

            # Exponential smoothing: 70% old, 30% new
            old_weight = self.weights_registry.get(aid, 1.0)
            blended    = 0.70 * old_weight + 0.30 * new_weight

            updated[aid] = round(blended, 4)
            print(f"  [{aid:<18s}]  acc={accuracy:.1%} over {n:3d} games  "
                  f"weight: {old_weight:.3f} → {blended:.3f}")

        self.weights_registry = updated
        return updated

    def log_outcome(self, game_id: int, game_date: str,
                    home_team: str, away_team: str, home_win: int) -> None:
        """Add an actual game outcome to the DB for meta-learner training."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO outcomes
                (game_id, game_date, home_team, away_team, home_win, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (game_id, game_date, home_team, away_team, int(home_win),
              datetime.utcnow().isoformat()))
        conn.commit()

    # ── Weighted average synthesis ────────────────────────────────────

    def _synthesize_weighted(
        self,
        signals:     list[AgentSignal],
        game_context = None,
    ) -> PickCard:
        """
        Weighted average ensemble.
        SentimentAgent contributes as both a direct vote AND a multiplier.
        """
        meta = _extract_game_meta(game_context)

        # Partition: sentiment separate from directional agents
        directional = [s for s in signals if s.agent_id != "sentiment"]
        sentiment   = next((s for s in signals if s.agent_id == "sentiment"), None)

        # ── SKIP check ────────────────────────────────────────────────
        above_floor = [s for s in directional if s.confidence >= self.confidence_floor]
        n_above     = len(above_floor)

        if n_above < self.min_agents:
            return self._make_skip_card(
                signals, meta,
                reason=f"Only {n_above}/{self.min_agents} agents above confidence floor",
                mode="weighted_avg",
            )

        # ── Weighted average over directional agents ──────────────────
        total_w = 0.0
        composite = 0.0
        for s in directional:
            w = self.weights_registry.get(s.agent_id, 1.0)
            composite += s.raw_score * w
            total_w   += w

        composite = composite / total_w if total_w > 0 else 0.5

        # ── Apply sentiment multiplier ────────────────────────────────
        if sentiment is not None:
            mult = sentiment.factors.get("multiplier", 1.0)
            # Multiplier shifts composite toward/away from 0.5
            edge_before = composite - 0.5
            edge_after  = edge_before * mult
            composite   = max(0.0, min(1.0, 0.5 + edge_after))

        # ── Include sentiment as a direct vote (lower weight) ─────────
        if sentiment is not None:
            w = self.weights_registry.get("sentiment", 0.5)
            all_w = total_w + w
            composite = (composite * total_w + sentiment.raw_score * w) / all_w

        return self._make_pick_card(
            composite, signals, meta, mode="weighted_avg"
        )

    # ── Meta-learner synthesis ────────────────────────────────────────

    def _synthesize_meta(
        self,
        signals:     list[AgentSignal],
        game_context = None,
    ) -> PickCard:
        """
        LogisticRegression meta-learner synthesis.
        Inputs: raw_score + confidence per agent (10 features).
        Output: P(home_win).
        """
        meta = _extract_game_meta(game_context)

        # SKIP check still applies
        above_floor = [s for s in signals if s.confidence >= self.confidence_floor]
        if len(above_floor) < self.min_agents:
            return self._make_skip_card(
                signals, meta,
                reason=f"Only {len(above_floor)}/{self.min_agents} agents above floor",
                mode="meta_learner",
            )

        feat_vec = np.array([_signals_to_feature_vector(signals)], dtype=np.float32)

        try:
            prob_home = float(self._model.predict_proba(feat_vec)[0, 1])
        except Exception as e:
            warnings.warn(f"Meta-learner predict failed ({e}) — falling back to weighted_avg")
            return self._synthesize_weighted(signals, game_context)

        return self._make_pick_card(
            prob_home, signals, meta, mode="meta_learner"
        )

    # ── Shared PickCard assembly ──────────────────────────────────────

    def _make_pick_card(
        self,
        composite: float,
        signals:   list[AgentSignal],
        meta:      dict,
        mode:      str,
    ) -> PickCard:
        edge_pct   = abs(composite - 0.5) * 100.0
        pick_dir   = "home" if composite > 0.5 else ("away" if composite < 0.5 else "neutral")
        pick_team  = (meta.get("home_team", "HOME") if pick_dir == "home"
                      else meta.get("away_team", "AWAY"))

        tier = _score_to_tier(composite)

        # SKIP if tier is too low even after floor check
        if tier == "SKIP":
            return self._make_skip_card(
                signals, meta,
                reason=f"Composite {composite:.3f} below pick threshold",
                mode=mode,
            )

        agreement = _compute_agreement(signals, self.confidence_floor)
        above     = sum(1 for s in signals if s.confidence >= self.confidence_floor)

        return PickCard(
            game_id               = meta.get("game_id", 0),
            game_date             = meta.get("game_date", ""),
            home_team             = meta.get("home_team", ""),
            away_team             = meta.get("away_team", ""),
            pick                  = pick_team,
            pick_direction        = pick_dir,
            raw_score             = round(composite, 4),
            edge_pct              = round(edge_pct, 2),
            confidence_tier       = tier,
            agent_agreement_score = round(agreement, 3),
            agents_above_floor    = above,
            mode_used             = mode,
            signal_summary        = _build_signal_summary(signals, self.weights_registry),
            weights_used          = dict(self.weights_registry),
        )

    def _make_skip_card(
        self,
        signals:  list[AgentSignal],
        meta:     dict,
        reason:   str,
        mode:     str,
    ) -> PickCard:
        above = sum(1 for s in signals if s.confidence >= self.confidence_floor)
        return PickCard(
            game_id               = meta.get("game_id", 0),
            game_date             = meta.get("game_date", ""),
            home_team             = meta.get("home_team", ""),
            away_team             = meta.get("away_team", ""),
            pick                  = "SKIP",
            pick_direction        = "skip",
            raw_score             = SKIP_SCORE,
            edge_pct              = 0.0,
            confidence_tier       = "SKIP",
            agent_agreement_score = _compute_agreement(signals, self.confidence_floor),
            agents_above_floor    = above,
            mode_used             = mode,
            signal_summary        = _build_signal_summary(signals, self.weights_registry),
            weights_used          = dict(self.weights_registry),
            skip_reason           = reason,
        )

    # ── SQLite logging ────────────────────────────────────────────────

    def _log_pick(self, card: PickCard) -> None:
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO picks (
                    created_at, game_id, game_date, home_team, away_team,
                    pick, pick_direction, raw_score, edge_pct, confidence_tier,
                    agent_agreement, agents_above_floor, mode_used,
                    skip_reason, signals_json, weights_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                card.produced_at,
                card.game_id,
                card.game_date,
                card.home_team,
                card.away_team,
                card.pick,
                card.pick_direction,
                card.raw_score,
                card.edge_pct,
                card.confidence_tier,
                card.agent_agreement_score,
                card.agents_above_floor,
                card.mode_used,
                card.skip_reason,
                json.dumps(card.signal_summary),
                json.dumps(card.weights_used),
            ))
            conn.commit()
        except Exception as e:
            warnings.warn(f"Failed to log pick to DB: {e}")

    # ── Reporting ─────────────────────────────────────────────────────

    def picks_dataframe(self, days: int = 30) -> pd.DataFrame:
        """Return recent picks as a DataFrame for analysis."""
        conn  = self._get_conn()
        query = """
            SELECT p.*, o.home_win as actual_home_win
            FROM picks p
            LEFT JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.game_date >= date('now', ?)
            ORDER BY p.game_date DESC, p.created_at DESC
        """
        return pd.read_sql_query(query, conn, params=(f"-{days} days",))

    def accuracy_report(self) -> dict:
        """
        Return win rate by confidence tier for picks with known outcomes.
        """
        conn = self._get_conn()
        query = """
            SELECT
                p.confidence_tier,
                COUNT(*) as n,
                SUM(CASE
                    WHEN (p.pick_direction='home' AND o.home_win=1) OR
                         (p.pick_direction='away' AND o.home_win=0)
                    THEN 1 ELSE 0
                END) as wins
            FROM picks p
            JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.confidence_tier != 'SKIP'
            GROUP BY p.confidence_tier
        """
        rows = conn.execute(query).fetchall()
        return {
            tier: {"n": n, "wins": w, "win_rate": round(w / n, 3) if n > 0 else None}
            for tier, n, w in rows
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _score_to_tier(score: float) -> ConfidenceTier:
    """Map a composite 0–1 score to a confidence tier."""
    edge = abs(score - 0.5)
    if edge >= (TIER_HIGH   - 0.5): return "HIGH"
    if edge >= (TIER_MEDIUM - 0.5): return "MEDIUM"
    if edge >= (TIER_LOW    - 0.5): return "LOW"
    return "SKIP"


def _compute_agreement(signals: list[AgentSignal], floor: float) -> float:
    """
    Agreement score: fraction of above-floor agents that agree on direction.
    1.0 = unanimous, 0.5 = split 50/50, 0.0 = no opinionated agents.
    """
    opinionated = [
        s for s in signals
        if s.confidence >= floor and s.pick_direction in ("home", "away")
    ]
    if not opinionated:
        return 0.0
    n_home = sum(1 for s in opinionated if s.pick_direction == "home")
    n_away = len(opinionated) - n_home
    return max(n_home, n_away) / len(opinionated)


def _build_signal_summary(
    signals: list[AgentSignal],
    weights: dict,
) -> dict:
    """Build the signals_json payload for SQLite logging."""
    return {
        s.agent_id: {
            "raw_score":  round(s.raw_score, 4),
            "confidence": round(s.confidence, 4),
            "direction":  s.pick_direction,
            "reasoning":  s.reasoning[:120],
            "weight":     weights.get(s.agent_id, 1.0),
        }
        for s in signals
    }


def _extract_game_meta(game_context) -> dict:
    """Pull identity fields from a GameContext (or return empty dict if None)."""
    if game_context is None:
        return {}
    return {
        "game_id":   getattr(game_context, "game_id",   0),
        "game_date": getattr(game_context, "game_date", ""),
        "home_team": getattr(getattr(game_context, "home", None), "team", ""),
        "away_team": getattr(getattr(game_context, "away", None), "team", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point — run today's synthesis after Module 1 + 2
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(BASE))

    parser = argparse.ArgumentParser(description="NHL MAS Module 3 — Master Synthesizer")
    parser.add_argument("--date",        default=None,           help="YYYY-MM-DD")
    parser.add_argument("--mode",        default="auto",         help="auto/weighted_avg/meta_learner")
    parser.add_argument("--no-advanced", action="store_true",    help="Skip play-by-play")
    parser.add_argument("--update-weights", action="store_true", help="Recalibrate agent weights from DB")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    from module1_ingest import build_today_contexts
    from agents import DEFAULT_AGENTS

    synthesizer = MasterSynthesizer(db_path=DB_PATH)
    synthesizer.fit_from_db()   # no-op if no historical data

    if args.update_weights:
        print("\nRecalibrating agent weights ...")
        synthesizer.update_weights()

    print(f"\n{'='*70}")
    print(f"  MODULE 3 — Master Synthesizer  [{target}]")
    print(f"  Mode: {args.mode}")
    print(f"{'='*70}\n")

    contexts = build_today_contexts(target, fetch_advanced=not args.no_advanced)

    all_cards = []
    for ctx in contexts:
        signals = [agent.analyze(ctx) for agent in DEFAULT_AGENTS]
        card    = synthesizer.synthesize(signals, ctx, mode=args.mode)
        all_cards.append(card)
        print(f"  {card}")

    playable = [c for c in all_cards if c.is_playable]
    print(f"\n{'='*70}")
    print(f"  Total games:    {len(all_cards)}")
    print(f"  Playable picks: {len(playable)}")
    if playable:
        print(f"\n  PLAYABLE PICKS:")
        for c in sorted(playable, key=lambda x: x.edge_pct, reverse=True):
            print(f"    {c}")
    print(f"{'='*70}")
