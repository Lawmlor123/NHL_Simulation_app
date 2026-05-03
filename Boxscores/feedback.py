"""
feedback.py
-----------
Module 4: Feedback Loop + Weight Updater — NHL MAS

Connects game outcomes (from NHL API final scores) to stored predictions,
computes per-agent rolling accuracy, recalibrates weights, and prints
weekly performance reports with ROI at -110.

Weight update formula (configurable, default EMA):
    new_weight = 0.80 * old_weight + 0.20 * rolling_accuracy
    where rolling_accuracy = win_rate / 0.50  (normalised; random agent = 1.0)

CLI:
    python feedback.py --resolve-date 2025-01-15
        Fetch final scores for that date, write outcomes, update weights.

    python feedback.py --weekly-report [--weeks N]
        Print last N*7 days performance report (default 1 week).

    python feedback.py --update-weights [--window N]
        Recalibrate weights from last N resolved picks (default 20).

    python feedback.py --show-weights
        Print current weights_registry.json.

    python feedback.py --resolve-date 2025-01-15 --dry-run
        Preview what would be written without touching the DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
DB_PATH      = BASE / "mas_picks.db"
WEIGHTS_PATH = BASE / "weights_registry.json"

# ── Constants ─────────────────────────────────────────────────────────────────
NHL_API_BASE  = "https://api-web.nhle.com/v1"
FINAL_STATES  = {"OFF", "FINAL"}

AGENT_ORDER: list[str] = [
    "team_form", "player_form", "goalie_form", "schedule", "sentiment"
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "team_form":   1.00,
    "player_form": 0.90,
    "goalie_form": 1.20,  # highest — goalies drive outcomes
    "schedule":    0.80,
    "sentiment":   0.50,
}

# EMA parameters  (spec formula: new = 0.8*old + 0.2*accuracy)
WEIGHT_ALPHA = 0.20   # 20% blended toward new evidence per update cycle
WEIGHT_MIN   = 0.30
WEIGHT_MAX   = 2.50

# ROI at standard -110 vig: risk $110 to win $100
JUICE_RISK    = 110
JUICE_WIN     = 100

# Minimum resolved picks before weight recalibration fires
MIN_SAMPLE    = 10

# ── DB schema ─────────────────────────────────────────────────────────────────
# Mirrors Module 3 schema; outcomes gains home_score / away_score columns.
# The `picks` table is owned by Module 3 — we only read from it here.

_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     INTEGER UNIQUE,
    game_date   TEXT,
    home_team   TEXT,
    away_team   TEXT,
    home_win    INTEGER,
    home_score  INTEGER,
    away_score  INTEGER,
    added_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_game ON outcomes(game_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON outcomes(game_date);
"""

_PICKS_DDL = """
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
CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(game_date);
CREATE INDEX IF NOT EXISTS idx_picks_game ON picks(game_id);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses — Module 4 output contracts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResolveResult:
    """Summary returned by FeedbackEngine.resolve_date()."""
    game_date:        str
    games_fetched:    int  = 0
    games_final:      int  = 0
    outcomes_written: int  = 0
    picks_matched:    int  = 0
    already_logged:   int  = 0
    errors:           list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.game_date}]  fetched={self.games_fetched}  "
            f"final={self.games_final}  written={self.outcomes_written}  "
            f"matched={self.picks_matched}  already={self.already_logged}"
        )


@dataclass
class AgentPerformance:
    """Per-agent accuracy over a rolling window."""
    agent_id:    str
    n_picks:     int   = 0
    n_wins:      int   = 0
    win_rate:    float = 0.0
    old_weight:  float = 1.0
    new_weight:  float = 1.0
    window_n:    int   = 20

    @property
    def accuracy_score(self) -> float:
        """
        Normalised accuracy: win_rate / 0.50
        A random agent (50 % win rate) returns 1.0 — weight unchanged.
        A 60 % agent returns 1.20 — weight pushed up.
        """
        return self.win_rate / 0.50 if self.win_rate > 0 else 0.0

    def __str__(self) -> str:
        if self.n_picks == 0:
            return f"  [{self.agent_id:<18s}]  no data  weight {self.old_weight:.3f} → {self.new_weight:.3f}"
        arrow = "↑" if self.new_weight > self.old_weight + 0.001 else (
                "↓" if self.new_weight < self.old_weight - 0.001 else "→")
        return (
            f"  [{self.agent_id:<18s}]  "
            f"{self.win_rate:.1%} ({self.n_wins}/{self.n_picks})  "
            f"weight {self.old_weight:.3f} {arrow} {self.new_weight:.3f}"
        )


@dataclass
class TierStats:
    """Win/loss/ROI breakdown for one confidence tier."""
    tier:   str
    n:      int   = 0
    wins:   int   = 0
    losses: int   = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def roi_pct(self) -> float:
        return compute_roi(self.wins, self.losses)

    def __str__(self) -> str:
        return (
            f"  {self.tier:<8s}  {self.wins:>3d}W-{self.losses:<3d}L  "
            f"({self.win_rate:.1%})  ROI {self.roi_pct:+.1f}%"
        )


@dataclass
class WeeklyReport:
    """Full weekly performance snapshot."""
    report_date:  str
    period_start: str
    period_end:   str

    # Overall record (resolved only)
    total_picks:   int   = 0   # playable (non-skip)
    total_wins:    int   = 0
    total_losses:  int   = 0
    total_skips:   int   = 0   # SKIP cards issued
    unresolved:    int   = 0   # picks without outcome yet

    # Derived
    win_rate:  float = 0.0
    roi_pct:   float = 0.0

    # Detail
    tier_breakdown:  dict = field(default_factory=dict)  # {tier: TierStats}
    agent_stats:     list = field(default_factory=list)  # [AgentPerformance]
    weight_updates:  dict = field(default_factory=dict)  # {agent_id: new_weight}


# ═══════════════════════════════════════════════════════════════════════════════
# NHL API score fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_scores_for_date(
    game_date: str,
    *,
    fetcher: Optional[Callable] = None,
    timeout: int = 15,
) -> list[dict]:
    """
    Pull final scores from NHL API for a given date.

    Args:
        game_date:  "YYYY-MM-DD"
        fetcher:    Optional override for HTTP GET (used in tests).
                    Callable(url) → dict
        timeout:    Request timeout in seconds.

    Returns:
        List of game dicts:
            game_id, home_team, away_team,
            home_score, away_score, home_win, is_final
    """
    url = f"{NHL_API_BASE}/score/{game_date}"

    if fetcher is not None:
        try:
            data = fetcher(url)
        except Exception as exc:
            raise RuntimeError(f"NHL API request failed for {game_date}: {exc}") from exc
    else:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"NHL API request failed for {game_date}: {exc}") from exc

    results = []
    for g in data.get("games", []):
        state    = g.get("gameState", "")
        home     = g.get("homeTeam", {})
        away     = g.get("awayTeam", {})
        h_score  = home.get("score", 0) or 0
        a_score  = away.get("score", 0) or 0

        results.append({
            "game_id":    g.get("id"),
            "home_team":  home.get("abbrev", ""),
            "away_team":  away.get("abbrev", ""),
            "home_score": h_score,
            "away_score": a_score,
            "home_win":   int(h_score > a_score),
            "is_final":   state in FINAL_STATES,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Utility — ROI at -110
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roi(wins: int, losses: int) -> float:
    """
    ROI % at standard -110 vig.

    Bet $110 to win $100 each pick.
    net  = wins * 100 - losses * 110
    cost = (wins + losses) * 110
    ROI  = net / cost * 100
    """
    total = wins + losses
    if total == 0:
        return 0.0
    net  = wins * JUICE_WIN - losses * JUICE_RISK
    cost = total * JUICE_RISK
    return round(net / cost * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# FeedbackEngine
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackEngine:
    """
    Connects NHL API outcomes to stored predictions and keeps agent weights
    calibrated over time.

    Args:
        db_path:      Path to SQLite DB (mas_picks.db).
        weights_path: Path to JSON weights file (weights_registry.json).
        fetcher:      Optional NHL API HTTP fetcher (Callable(url)->dict).
                      Defaults to requests.get. Inject a mock for tests.
    """

    def __init__(
        self,
        db_path:      Optional[Path]     = None,
        weights_path: Optional[Path]     = None,
        fetcher:      Optional[Callable] = None,
    ):
        self._db_path      = db_path      or DB_PATH
        self._weights_path = weights_path or WEIGHTS_PATH
        self._fetcher      = fetcher       # None → real NHL API
        self._conn: Optional[sqlite3.Connection] = None

    # ── DB connection (lazy) ──────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(str(self._db_path))
                self._conn.row_factory = sqlite3.Row
                self._conn.executescript(_PICKS_DDL)
                self._conn.executescript(_OUTCOMES_DDL)
                # Migrate: add score columns if missing (safe on existing DBs)
                for col, dtype in [("home_score", "INTEGER"), ("away_score", "INTEGER")]:
                    try:
                        self._conn.execute(
                            f"ALTER TABLE outcomes ADD COLUMN {col} {dtype}"
                        )
                    except sqlite3.OperationalError:
                        pass  # column already exists
                self._conn.commit()
            except Exception:
                # Fallback in-memory for sandbox / tests
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
                self._conn.executescript(_PICKS_DDL)
                self._conn.executescript(_OUTCOMES_DDL)
                self._conn.commit()
        return self._conn

    # ── Outcome resolution ────────────────────────────────────────────────────

    def resolve_date(
        self,
        game_date: str,
        *,
        dry_run: bool = False,
    ) -> ResolveResult:
        """
        Pull final scores from NHL API for `game_date`, match to picks,
        write outcomes, then update weights.

        Args:
            game_date:  "YYYY-MM-DD"
            dry_run:    If True, preview without writing to DB.

        Returns:
            ResolveResult summary.
        """
        result = ResolveResult(game_date=game_date)

        # 1. Fetch from NHL API
        try:
            games = fetch_scores_for_date(game_date, fetcher=self._fetcher)
        except RuntimeError as exc:
            result.errors.append(str(exc))
            return result

        result.games_fetched = len(games)
        finals = [g for g in games if g["is_final"]]
        result.games_final   = len(finals)

        if not finals:
            return result

        # 2. Check which game_ids already have outcomes
        conn        = self._get_conn()
        final_ids   = [g["game_id"] for g in finals if g["game_id"] is not None]
        if not final_ids:
            return result

        placeholders = ",".join("?" * len(final_ids))
        existing_ids = {
            row[0] for row in
            conn.execute(
                f"SELECT game_id FROM outcomes WHERE game_id IN ({placeholders})",
                final_ids,
            ).fetchall()
        }
        result.already_logged = len(existing_ids)

        # 3. Match to picks in DB
        new_games = [g for g in finals
                     if g["game_id"] is not None
                     and g["game_id"] not in existing_ids]

        picks_ids = {
            row[0] for row in
            conn.execute(
                f"SELECT DISTINCT game_id FROM picks WHERE game_id IN ({placeholders})",
                final_ids,
            ).fetchall()
        }
        result.picks_matched = sum(1 for g in new_games if g["game_id"] in picks_ids)

        if dry_run:
            print(f"  [DRY RUN] {result}")
            for g in new_games:
                picked = "✓" if g["game_id"] in picks_ids else "·"
                winner = g["home_team"] if g["home_win"] else g["away_team"]
                print(f"    {picked} {g['away_team']:3s}@{g['home_team']:3s}  "
                      f"{g['away_score']}-{g['home_score']}  → {winner}")
            return result

        # 4. Write outcomes
        result.outcomes_written = self._write_outcomes(new_games)

        # 5. Auto-update weights after resolution
        updated = self.update_weights()
        if updated:
            self.persist_weights(updated)

        print(f"  ✅ {result}")
        return result

    def _write_outcomes(self, games: list[dict]) -> int:
        """INSERT OR REPLACE outcomes for each game. Returns count written."""
        conn    = self._get_conn()
        now     = datetime.utcnow().isoformat()
        written = 0

        for g in games:
            game_date_str = (str(g.get("game_id", ""))[:8]  # fallback: parse from ID
                             if not g.get("game_date") else g["game_date"])
            # Derive date from game_id if not provided (NHL ID format: SSSSTTGGGG)
            # e.g., 2024020001 → season 2024 (doesn't encode date directly)
            # Use passed-in date context instead
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes
                    (game_id, game_date, home_team, away_team,
                     home_win, home_score, away_score, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    g["game_id"],
                    g.get("game_date", ""),
                    g["home_team"],
                    g["away_team"],
                    g["home_win"],
                    g.get("home_score"),
                    g.get("away_score"),
                    now,
                ),
            )
            written += 1

        conn.commit()
        return written

    # ── Per-agent accuracy ────────────────────────────────────────────────────

    def compute_agent_stats(
        self,
        window_n:         int   = 20,
        confidence_floor: float = 0.52,
    ) -> list[AgentPerformance]:
        """
        Compute per-agent ML accuracy over the last `window_n` resolved picks.

        A pick is "correct" when the agent's raw_score direction (>0.5 = home)
        matches the actual outcome AND the agent's confidence ≥ floor.

        Returns a list of AgentPerformance, one per AGENT_ORDER entry.
        """
        conn = self._get_conn()
        current_weights = self.load_weights()

        # Pull last window_n resolved picks (non-skip)
        rows = conn.execute(
            """
            SELECT p.signals_json, o.home_win
            FROM picks p
            JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.confidence_tier != 'SKIP'
            ORDER BY p.game_date DESC, p.created_at DESC
            LIMIT ?
            """,
            (window_n,),
        ).fetchall()

        # Per-agent: collect correct/total
        correct_map: dict[str, list[int]] = {aid: [] for aid in AGENT_ORDER}

        for row in rows:
            try:
                sig_dict = json.loads(row["signals_json"])
                home_win = int(row["home_win"])
                for aid in AGENT_ORDER:
                    if aid not in sig_dict:
                        continue
                    raw  = float(sig_dict[aid].get("raw_score", 0.5))
                    conf = float(sig_dict[aid].get("confidence", 0.0))
                    if conf < confidence_floor:
                        continue
                    predicted_home = int(raw > 0.5)
                    correct_map[aid].append(int(predicted_home == home_win))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        stats = []
        for aid in AGENT_ORDER:
            picks  = correct_map[aid]
            n      = len(picks)
            wins   = sum(picks)
            rate   = wins / n if n > 0 else 0.0
            old_w  = current_weights.get(aid, DEFAULT_WEIGHTS.get(aid, 1.0))

            # New weight = EMA blend toward normalised accuracy
            # normalised_accuracy = win_rate / 0.50 (random=1.0)
            if n >= 5:
                norm_acc  = rate / 0.50
                new_w     = (1 - WEIGHT_ALPHA) * old_w + WEIGHT_ALPHA * norm_acc
                new_w     = max(WEIGHT_MIN, min(WEIGHT_MAX, round(new_w, 4)))
            else:
                new_w = old_w  # not enough data — preserve current weight

            stats.append(AgentPerformance(
                agent_id   = aid,
                n_picks    = n,
                n_wins     = wins,
                win_rate   = round(rate, 4),
                old_weight = round(old_w, 4),
                new_weight = round(new_w, 4),
                window_n   = window_n,
            ))

        return stats

    # ── Weight recalibration ──────────────────────────────────────────────────

    def update_weights(
        self,
        window_n:         int   = 20,
        confidence_floor: float = 0.52,
    ) -> dict:
        """
        Recalibrate weights from the last `window_n` resolved picks.
        Returns the updated weights dict (also saves to weights_registry.json).
        Prints a short report if data is sufficient.
        """
        conn = self._get_conn()
        n_resolved = conn.execute(
            """
            SELECT COUNT(*) FROM picks p
            JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.confidence_tier != 'SKIP'
            """
        ).fetchone()[0]

        if n_resolved < MIN_SAMPLE:
            print(f"  ⚠  Only {n_resolved} resolved picks — "
                  f"need {MIN_SAMPLE}+ to update weights")
            return {}

        stats      = self.compute_agent_stats(window_n=window_n,
                                              confidence_floor=confidence_floor)
        new_weights = {}
        for ap in stats:
            new_weights[ap.agent_id] = ap.new_weight
            print(ap)

        return new_weights

    # ── Weight persistence ────────────────────────────────────────────────────

    def load_weights(self) -> dict:
        """
        Load weights from JSON file.
        Falls back to DEFAULT_WEIGHTS if file doesn't exist or is malformed.
        """
        try:
            if self._weights_path.exists():
                data = json.loads(self._weights_path.read_text())
                # Validate: merge with defaults to ensure all agents present
                loaded = {k: float(v) for k, v in data.items() if k in AGENT_ORDER}
                merged = dict(DEFAULT_WEIGHTS)
                merged.update(loaded)
                return merged
        except (json.JSONDecodeError, OSError, ValueError):
            pass
        return dict(DEFAULT_WEIGHTS)

    def persist_weights(self, weights: dict) -> None:
        """
        Save updated weights dict to JSON.
        Only known agent IDs are written (guards against corruption).
        """
        clean = {
            aid: round(float(weights.get(aid, DEFAULT_WEIGHTS[aid])), 4)
            for aid in AGENT_ORDER
        }
        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            self._weights_path.write_text(json.dumps(clean, indent=2))
            print(f"  💾 Weights saved → {self._weights_path.name}")
        except OSError as exc:
            warnings.warn(f"Could not save weights: {exc}")

    # ── Weekly report ─────────────────────────────────────────────────────────

    def weekly_report(
        self,
        weeks:    int   = 1,
        window_n: int   = 20,
    ) -> WeeklyReport:
        """
        Generate a performance report for the last `weeks` weeks.

        Returns:
            WeeklyReport dataclass with tier breakdown, per-agent stats, ROI.
        """
        today  = date.today()
        end_dt = today
        start_dt = today - timedelta(weeks=weeks)

        conn = self._get_conn()

        # All picks in window (including SKIP)
        rows = conn.execute(
            """
            SELECT
                p.pick_direction,
                p.confidence_tier,
                p.signals_json,
                o.home_win
            FROM picks p
            LEFT JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.game_date BETWEEN ? AND ?
            ORDER BY p.game_date DESC
            """,
            (start_dt.isoformat(), end_dt.isoformat()),
        ).fetchall()

        # Partition rows
        report = WeeklyReport(
            report_date  = today.isoformat(),
            period_start = start_dt.isoformat(),
            period_end   = end_dt.isoformat(),
        )

        tier_map: dict[str, TierStats] = {
            t: TierStats(tier=t) for t in ("HIGH", "MEDIUM", "LOW")
        }

        for row in rows:
            tier      = row["confidence_tier"]
            direction = row["pick_direction"]
            home_win  = row["home_win"]

            if tier == "SKIP":
                report.total_skips += 1
                continue

            report.total_picks += 1

            if home_win is None:
                report.unresolved += 1
                continue

            home_win = int(home_win)
            correct  = int(
                (direction == "home" and home_win == 1) or
                (direction == "away" and home_win == 0)
            )

            if correct:
                report.total_wins  += 1
            else:
                report.total_losses += 1

            if tier in tier_map:
                tier_map[tier].n += 1
                if correct:
                    tier_map[tier].wins   += 1
                else:
                    tier_map[tier].losses += 1

        report.tier_breakdown = {
            t: ts for t, ts in tier_map.items()
            if ts.n > 0
        }

        # Derived overall
        w, l = report.total_wins, report.total_losses
        report.win_rate = round(w / (w + l), 4) if (w + l) > 0 else 0.0
        report.roi_pct  = compute_roi(w, l)

        # Per-agent stats (rolling window)
        report.agent_stats    = self.compute_agent_stats(window_n=window_n)
        report.weight_updates = {ap.agent_id: ap.new_weight
                                 for ap in report.agent_stats}

        return report

    # ── Report printer ────────────────────────────────────────────────────────

    def print_report(self, report: WeeklyReport, *, verbose: bool = False) -> None:
        """Print a formatted weekly report to stdout."""
        sep = "═" * 62
        print(f"\n{sep}")
        print(f"  NHL MAS — Weekly Performance Report")
        print(f"  Period : {report.period_start} → {report.period_end}")
        print(f"  As of  : {report.report_date}")
        print(sep)

        # Overall record
        w, l = report.total_wins, report.total_losses
        total_resolved = w + l
        print(f"\n  OVERALL RECORD   {w}W–{l}L   ({report.win_rate:.1%})")
        print(f"  ROI at -110      {report.roi_pct:+.1f}%")
        print(f"  Total picks      {report.total_picks}  "
              f"(resolved: {total_resolved}  "
              f"pending: {report.unresolved}  "
              f"skipped: {report.total_skips})")

        # Tier breakdown
        if report.tier_breakdown:
            print(f"\n  CONFIDENCE TIER BREAKDOWN")
            for tier in ("HIGH", "MEDIUM", "LOW"):
                ts = report.tier_breakdown.get(tier)
                if ts:
                    print(ts)

        # Per-agent accuracy
        if report.agent_stats:
            print(f"\n  PER-AGENT ACCURACY  (rolling last {report.agent_stats[0].window_n} games)")
            for ap in report.agent_stats:
                print(ap)

        # Pick-type breakdown (ML only implemented; spread/total = future)
        print(f"\n  PICK TYPE BREAKDOWN")
        print(f"  {'ML':<12s}  {w}W–{l}L  ({report.win_rate:.1%})  ROI {report.roi_pct:+.1f}%")
        print(f"  {'Spread':<12s}  [requires score data — enhancement pending]")
        print(f"  {'Total':<12s}  [requires over/under line data — enhancement pending]")

        print(f"\n{sep}\n")

    # ── Picks dataframe helper ────────────────────────────────────────────────

    def picks_dataframe(self, days: int = 30):
        """Return recent picks + outcomes as a pandas DataFrame."""
        import pandas as pd
        conn  = self._get_conn()
        query = """
            SELECT p.*, o.home_win, o.home_score, o.away_score
            FROM picks p
            LEFT JOIN outcomes o ON p.game_id = o.game_id
            WHERE p.game_date >= date('now', ?)
            ORDER BY p.game_date DESC, p.created_at DESC
        """
        return pd.read_sql_query(query, conn, params=(f"-{days} days",))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NHL MAS Module 4 — Feedback Loop + Weight Updater"
    )
    p.add_argument(
        "--resolve-date", metavar="YYYY-MM-DD",
        help="Pull final scores for this date and write outcomes",
    )
    p.add_argument(
        "--weekly-report", action="store_true",
        help="Print last N weeks performance report",
    )
    p.add_argument(
        "--weeks", type=int, default=1,
        help="Number of weeks for weekly report (default 1)",
    )
    p.add_argument(
        "--update-weights", action="store_true",
        help="Recalibrate agent weights from DB",
    )
    p.add_argument(
        "--window", type=int, default=20,
        help="Rolling window for weight recalibration (default 20)",
    )
    p.add_argument(
        "--show-weights", action="store_true",
        help="Print current weights_registry.json",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview resolve without writing to DB",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    engine  = FeedbackEngine()

    if args.show_weights:
        weights = engine.load_weights()
        print("\n  Current weights_registry:")
        for aid in AGENT_ORDER:
            print(f"    {aid:<18s} {weights.get(aid, '—'):.3f}")
        print()
        return

    if args.resolve_date:
        print(f"\n  Resolving outcomes for {args.resolve_date} …")
        result = engine.resolve_date(args.resolve_date, dry_run=args.dry_run)
        if result.errors:
            for err in result.errors:
                print(f"  ⚠  {err}")

    if args.update_weights:
        print(f"\n  Recalibrating weights (window={args.window}) …")
        updated = engine.update_weights(window_n=args.window)
        if updated:
            engine.persist_weights(updated)

    if args.weekly_report:
        report = engine.weekly_report(weeks=args.weeks, window_n=args.window)
        engine.print_report(report)

    if not any([args.resolve_date, args.update_weights,
                args.weekly_report, args.show_weights]):
        print("  No action specified. Use --help for options.")


if __name__ == "__main__":
    main()
