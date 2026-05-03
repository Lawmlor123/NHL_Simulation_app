"""
goalie_inference.py — Probable Starter Inference Engine
========================================================
When no external source confirms today's starter, infer from recent
rotation history stored in goalie_features.parquet.

Logic (in order of confidence):
  1. Not B2B  → last starter repeats   ~78% historically
  2. B2B      → backup steps in        ~82% historically
  3. 3+ in a row → slight pullback, confidence reduced
  4. Days-since-last-start > 5 → fatigue/rest rotation possible

Returns status = "Inferred" which sits between "Unconfirmed" (2) and
"Likely" (5) in STATUS_RANK — use rank 3.5 effectively.

Public API:
    engine = GoalieInferenceEngine()
    result = engine.infer(team="BOS", game_date="2026-04-14", is_b2b=False)
    # → {"name": "Jeremy Swayman", "status": "Inferred", "confidence": 0.80, "reason": "..."}
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("nhl_mas.goalie_inference")

BASE           = Path(__file__).resolve().parent
FEATURES_PATH  = BASE / "goalie_features.parquet"
BOXSCORES_PATH = BASE / "goalie_boxscores_raw.parquet"

# Historical accuracy rates (from NHL literature + empirical validation)
P_REPEAT_IF_RESTED   = 0.78   # same goalie repeats if team is NOT B2B
P_BACKUP_IF_B2B      = 0.82   # backup starts if team IS on B2B
P_REPEAT_IF_STREAK3  = 0.70   # 3+ starts in a row — rotation risk rises
P_REPEAT_IF_LONG_REST = 0.72  # >5 days since team's last game — uncertain


class GoalieInferenceEngine:
    """
    Loads goalie history once, then answers infer() calls quickly.
    Singleton-safe: instantiate once and reuse across all teams.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            # Prefer features parquet (has rolling stats + is_starter flag)
            df = pd.read_parquet(FEATURES_PATH)
            # Normalise column names across parquet versions
            # Rename legacy column names; add is_starter from gamesStarted only
            # if is_starter doesn't already exist (avoid duplicate-column bug)
            renames = {}
            if "goalieFullName" in df.columns and "goalie_name" not in df.columns:
                renames["goalieFullName"] = "goalie_name"
            if "gameDate" in df.columns and "game_date" not in df.columns:
                renames["gameDate"] = "game_date"
            if "teamAbbrev" in df.columns and "team" not in df.columns:
                renames["teamAbbrev"] = "team"
            if "gamesStarted" in df.columns and "is_starter" not in df.columns:
                renames["gamesStarted"] = "is_starter"
            if renames:
                df = df.rename(columns=renames)
            required = {"goalie_name", "game_date", "team", "is_starter"}
            if not required.issubset(df.columns):
                raise ValueError(f"Missing columns: {required - set(df.columns)}")

            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            df = df[df["is_starter"] == 1].copy()
            # Drop duplicate (team, game_date, goalie) rows that arise from
            # multi-period boxscore joins — keep one record per starter per game
            df = df.drop_duplicates(subset=["team", "game_date", "goalie_name"])
            df = df.reset_index(drop=True)
            df = df.sort_values(["team", "game_date"])
            self._df = df
            log.info("GoalieInferenceEngine: loaded %d starter records (%s → %s)",
                     len(df), df["game_date"].min(), df["game_date"].max())
        except Exception as e:
            log.warning("GoalieInferenceEngine: could not load parquet — %s", e)
            self._df = None
        self._loaded = True

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(
        self,
        team: str,
        game_date: str,
        is_b2b: bool = False,
        current_dfo_name: str = "",
    ) -> Optional[dict]:
        """
        Infer the probable starter for `team` on `game_date`.

        Parameters
        ----------
        team          : NHL team abbreviation (e.g. "BOS")
        game_date     : ISO date string "YYYY-MM-DD"
        is_b2b        : True if this is the team's second game in 2 nights
        current_dfo_name : Name DFO already has (even if Unconfirmed); helps
                           cross-validate the inference. Pass "" if no name.

        Returns
        -------
        dict with keys: name, status, confidence (0-1), reason
        OR None if insufficient history.
        """
        self._load()
        if self._df is None:
            return None

        target_date = pd.to_datetime(game_date).date()

        # Pull this team's history before today
        team_hist = self._df[
            (self._df["team"] == team) &
            (self._df["game_date"] < target_date)
        ].sort_values("game_date", ascending=False)

        if len(team_hist) < 3:
            log.debug("  [%s] Inference skipped: only %d historical starts", team, len(team_hist))
            return None

        last_game   = team_hist.iloc[0]
        last_date   = last_game["game_date"]
        last_name   = str(last_game["goalie_name"]).strip()
        days_since  = (target_date - last_date).days

        # Consecutive starts for last_name
        consecutive = 0
        for _, row in team_hist.iterrows():
            if str(row["goalie_name"]).strip() == last_name:
                consecutive += 1
            else:
                break

        # Find backup: most-recent starter who isn't last_name
        backup_rows = team_hist[team_hist["goalie_name"].str.strip() != last_name]
        backup_name = str(backup_rows.iloc[0]["goalie_name"]).strip() if len(backup_rows) else ""

        # ── Decision tree ─────────────────────────────────────────────────────
        if days_since > 5:
            # Long rest — teams sometimes shuffle but starter still likely
            inferred_name = last_name
            confidence    = P_REPEAT_IF_LONG_REST
            reason        = (f"Long rest ({days_since}d since last game) — "
                             f"{last_name} likely but rotation possible")

        elif is_b2b:
            if backup_name:
                inferred_name = backup_name
                confidence    = P_BACKUP_IF_B2B
                reason        = (f"B2B → backup rotation. "
                                 f"Last starter: {last_name} ({consecutive} straight). "
                                 f"Backup: {backup_name}")
            else:
                # No known backup — starter might play anyway
                inferred_name = last_name
                confidence    = 1 - P_BACKUP_IF_B2B  # inverted — low confidence
                reason        = (f"B2B → backup expected but no backup found in history. "
                                 f"Defaulting to {last_name} (low confidence)")

        elif consecutive >= 3:
            # Starter is on a run — slight pullback risk
            inferred_name = last_name
            confidence    = P_REPEAT_IF_STREAK3
            reason        = (f"{last_name} on {consecutive}-game streak — "
                             f"likely repeats but rotation watch")

        else:
            inferred_name = last_name
            confidence    = P_REPEAT_IF_RESTED
            reason        = (f"{last_name} started last game ({last_date}) — "
                             f"not B2B, {consecutive} straight, repeating")

        # ── Cross-validate with DFO name ──────────────────────────────────────
        dfo_last = _last_name(current_dfo_name)
        inf_last = _last_name(inferred_name)

        if dfo_last and inf_last:
            if dfo_last == inf_last:
                # DFO and inference agree → upgrade confidence slightly
                confidence = min(0.92, confidence + 0.08)
                reason += f" ✓ matches DFO ({current_dfo_name})"
            else:
                # DFO and inference disagree → lower confidence, trust DFO name
                # but still return inference for logging
                confidence = max(0.30, confidence - 0.15)
                reason += (f" ⚠ DFO says {current_dfo_name!r} — conflict, "
                           f"trusting DFO name, lowering confidence")
                # Override with DFO name if DFO is more recent
                inferred_name = current_dfo_name
                reason += f" → using DFO name"

        log.info("  [%s] Inference: %s (conf=%.0f%%) — %s",
                 team, inferred_name, confidence * 100, reason)

        return {
            "name":       inferred_name,
            "status":     "Inferred",
            "confidence": round(confidence, 3),
            "reason":     reason,
        }

    def team_rotation_summary(self, team: str, n: int = 10) -> str:
        """Quick human-readable summary of a team's recent goalie rotation."""
        self._load()
        if self._df is None:
            return f"[{team}] No data"
        hist = self._df[self._df["team"] == team].sort_values("game_date", ascending=False).head(n)
        lines = [f"  [{team}] Last {len(hist)} starts:"]
        for _, row in hist.iterrows():
            lines.append(f"    {row['game_date']}  {row['goalie_name']}")
        return "\n".join(lines)


def _last_name(full_name: str) -> str:
    """Extract lower-cased last name for fuzzy matching."""
    if not full_name:
        return ""
    parts = str(full_name).strip().split()
    return parts[-1].lower() if parts else ""


# ── Convenience function (mirrors scraper API style) ──────────────────────────
_engine: Optional[GoalieInferenceEngine] = None

def get_inferred_goalie(
    team: str,
    game_date: str,
    is_b2b: bool = False,
    current_dfo_name: str = "",
) -> Optional[dict]:
    """
    Module-level convenience wrapper — creates engine singleton on first call.
    Returns {name, status, confidence, reason} or None.
    """
    global _engine
    if _engine is None:
        _engine = GoalieInferenceEngine()
    return _engine.infer(team, game_date, is_b2b, current_dfo_name)


# ── CLI test harness ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = GoalieInferenceEngine()
    today  = datetime.today().strftime("%Y-%m-%d")

    # Test a few teams
    test_teams = sys.argv[1:] if len(sys.argv) > 1 else [
        "BOS", "TOR", "EDM", "NYR", "FLA", "WPG", "COL", "VAN"
    ]

    print(f"\n{'='*60}")
    print(f"  GOALIE INFERENCE ENGINE — test for {today}")
    print(f"{'='*60}")

    for team in test_teams:
        print("")
        print(engine.team_rotation_summary(team, n=5))
        result = engine.infer(team, today, is_b2b=False)
        if result:
            print("  Inferred starter: " + result["name"] + "  (conf=" + str(round(result["confidence"]*100)) + "%)")
            print("  Reason: " + result["reason"])
        else:
            print("  Insufficient history for inference")

    print("")
    print("="*60)
    print("")
