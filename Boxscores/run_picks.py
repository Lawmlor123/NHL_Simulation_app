"""
run_picks.py
------------
Module 6: Orchestrator / Runner — NHL MAS

Ties Modules 1-5 together into a single CLI entry point.

Commands:
    python run_picks.py --mode full          [--date YYYY-MM-DD] [--dry-run] [--tone hype|analytical]
    python run_picks.py --mode script-only   [--date YYYY-MM-DD] [--tone hype|analytical]
    python run_picks.py --mode report        [--date YYYY-MM-DD] [--weeks N]

Mode details:
    full         Fetch game contexts → run 5 agents per game → synthesize →
                 write PickCards to SQLite → generate Alexis script → print table.
    script-only  Load existing PickCards from DB for the date → regenerate script only.
    report       Run feedback module for completed games, print performance report.

Flags:
    --dry-run    Print all agent signals and synthesizer outputs without writing to DB.
    --verbose    Set log level to DEBUG.
    --quiet      Set log level to WARNING.
    --tone       Script tone: hype (default) or analytical.
    --weeks N    Weeks of history for report mode (default 1).
    --no-advanced  Skip NHL play-by-play (faster, less accurate).

Config (.env):
    DB_PATH, WEIGHTS_FILE_PATH, LOG_LEVEL, CONFIDENCE_FLOOR,
    MIN_AGENTS, SYNTHESIS_MODE, SCRIPT_TONE, FETCH_ADVANCED
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
DB_PATH      = BASE / "mas_picks.db"
WEIGHTS_PATH = BASE / "weights_registry.json"
ENV_PATH     = BASE / ".env"

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("nhl_mas")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamped console output."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log.setLevel(numeric)


# ═══════════════════════════════════════════════════════════════════════════════
# Config — loaded from .env
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    db_path:           Path  = DB_PATH
    weights_file_path: Path  = WEIGHTS_PATH
    log_level:         str   = "INFO"
    confidence_floor:  float = 0.52
    min_agents:        int   = 3
    synthesis_mode:    str   = "auto"
    script_tone:       str   = "hype"
    fetch_advanced:    bool  = True


def load_config(env_path: Optional[Path] = None) -> Config:
    """
    Load config from .env file and environment variables.
    Precedence: env vars > .env file > defaults.
    """
    # Optional python-dotenv support
    _env_file = env_path or ENV_PATH
    try:
        from dotenv import dotenv_values
        env_vars = dotenv_values(str(_env_file))
    except ImportError:
        env_vars = {}
        if _env_file.exists():
            # Minimal manual .env parser (KEY=VALUE lines, ignores # comments)
            for line in _env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env_vars[k.strip()] = v.strip()

    def _get(key: str, default: str = "") -> str:
        """Env vars take precedence over .env file."""
        return os.environ.get(key, env_vars.get(key, default))

    def _path(key: str, default: Path) -> Path:
        v = _get(key)
        return Path(v) if v else default

    def _bool(key: str, default: bool) -> bool:
        v = _get(key, str(default)).lower()
        return v in ("1", "true", "yes", "on")

    return Config(
        db_path           = _path("DB_PATH",           DB_PATH),
        weights_file_path = _path("WEIGHTS_FILE_PATH", WEIGHTS_PATH),
        log_level         = _get("LOG_LEVEL",          "INFO"),
        confidence_floor  = float(_get("CONFIDENCE_FLOOR", "0.52")),
        min_agents        = int(_get("MIN_AGENTS",        "3")),
        synthesis_mode    = _get("SYNTHESIS_MODE",     "auto"),
        script_tone       = _get("SCRIPT_TONE",        "hype"),
        fetch_advanced    = _bool("FETCH_ADVANCED",    True),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RunResult — orchestrator output contract
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Summary returned by every orchestrator run()."""
    run_at:          str   = field(default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc).isoformat())
    mode:            str   = ""
    target_date:     str   = ""
    dry_run:         bool  = False

    n_games:         int   = 0
    n_picks:         int   = 0   # playable (non-skip)
    n_high:          int   = 0   # HIGH-tier picks
    n_in_script:     int   = 0   # picks included in Alexis script

    pick_cards:      list  = field(default_factory=list)
    script:          str   = ""
    elapsed_seconds: float = 0.0
    errors:          list  = field(default_factory=list)

    def __str__(self) -> str:
        tag = " [DRY-RUN]" if self.dry_run else ""
        return (
            f"[{self.mode}{tag}]  {self.target_date}  "
            f"games={self.n_games}  picks={self.n_picks}  "
            f"high={self.n_high}  script={self.n_in_script}  "
            f"({self.elapsed_seconds:.1f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DB helper — reconstruct PickCards from SQLite rows
# ═══════════════════════════════════════════════════════════════════════════════

def _load_picks_from_db(conn: sqlite3.Connection, game_date: str) -> list:
    """
    Load PickCard-compatible objects from the picks table for a given date.
    Imports PickCard from module3_synthesizer; falls back to a minimal dict-
    based proxy if the import fails (for tests without the full pipeline).
    """
    rows = conn.execute(
        """
        SELECT game_id, game_date, home_team, away_team,
               pick, pick_direction, raw_score, edge_pct,
               confidence_tier, agent_agreement, agents_above_floor,
               mode_used, skip_reason, signals_json, weights_json
        FROM picks
        WHERE game_date = ?
        ORDER BY created_at DESC
        """,
        (game_date,),
    ).fetchall()

    # Try to reconstruct real PickCard objects
    try:
        sys.path.insert(0, str(BASE))
        from module3_synthesizer import PickCard
        _use_real = True
    except ImportError:
        _use_real = False

    results = []
    for r in rows:
        if _use_real:
            card = PickCard(
                game_id               = r[0],
                game_date             = r[1],
                home_team             = r[2],
                away_team             = r[3],
                pick                  = r[4],
                pick_direction        = r[5],
                raw_score             = r[6],
                edge_pct              = r[7],
                confidence_tier       = r[8],
                agent_agreement_score = r[9],
                agents_above_floor    = r[10],
                mode_used             = r[11],
                skip_reason           = r[12] or "",
                signal_summary        = json.loads(r[13] or "{}"),
                weights_used          = json.loads(r[14] or "{}"),
            )
        else:
            # Minimal proxy for tests / sandbox
            card = _PickProxy(r)
        results.append(card)

    return results


class _PickProxy:
    """Duck-type PickCard proxy for environments without module3_synthesizer."""
    __slots__ = (
        "game_id", "game_date", "home_team", "away_team",
        "pick", "pick_direction", "raw_score", "edge_pct",
        "confidence_tier", "agent_agreement_score", "agents_above_floor",
        "mode_used", "skip_reason", "signal_summary", "weights_used",
    )

    def __init__(self, row):
        (self.game_id, self.game_date, self.home_team, self.away_team,
         self.pick, self.pick_direction, self.raw_score, self.edge_pct,
         self.confidence_tier, self.agent_agreement_score,
         self.agents_above_floor, self.mode_used, self.skip_reason,
         signals_json, weights_json) = row
        self.signal_summary = json.loads(signals_json or "{}")
        self.weights_used   = json.loads(weights_json or "{}")
        self.skip_reason    = self.skip_reason or ""

    @property
    def is_playable(self) -> bool:
        return self.confidence_tier != "SKIP"

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"


# ═══════════════════════════════════════════════════════════════════════════════
# MASOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class MASOrchestrator:
    """
    Ties all NHL MAS modules together.

    All expensive dependencies (context builder, agents, synthesizer,
    feedback engine) are injectable — pass mocks for testing.
    """

    def __init__(
        self,
        config:           Config,
        db_path:          Optional[Path]     = None,
        weights_path:     Optional[Path]     = None,
        context_builder:  Optional[Callable] = None,
        agents:           Optional[list]     = None,
        synthesizer                          = None,
        feedback_engine                      = None,
    ):
        self.config   = config
        self._db_path = db_path or config.db_path
        self._wt_path = weights_path or config.weights_file_path
        self._conn: Optional[sqlite3.Connection] = None

        # Inject real pipeline lazily (avoids import errors in tests/sandbox)
        self._context_builder = context_builder
        self._agents          = agents
        self._synthesizer     = synthesizer
        self._feedback        = feedback_engine

    # ── Lazy imports ─────────────────────────────────────────────────────────

    def _ensure_pipeline(self) -> None:
        """Load real pipeline modules if not injected."""
        if self._context_builder is None:
            sys.path.insert(0, str(BASE))
            from module1_ingest import build_today_contexts
            self._context_builder = build_today_contexts

        if self._agents is None:
            from agents import DEFAULT_AGENTS
            self._agents = DEFAULT_AGENTS

        if self._synthesizer is None:
            from module3_synthesizer import MasterSynthesizer
            self._synthesizer = MasterSynthesizer(
                weights_registry = self._load_weights(),
                confidence_floor = self.config.confidence_floor,
                min_agents       = self.config.min_agents,
                db_path          = self._db_path,
            )
            self._synthesizer.fit_from_db()

        if self._feedback is None:
            from feedback import FeedbackEngine
            self._feedback = FeedbackEngine(
                db_path      = self._db_path,
                weights_path = self._wt_path,
            )

    def _load_weights(self) -> dict:
        try:
            if self._wt_path.exists():
                return json.loads(self._wt_path.read_text())
        except Exception:
            pass
        from module3_synthesizer import DEFAULT_WEIGHTS
        return dict(DEFAULT_WEIGHTS)

    # ── DB connection ─────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self._db_path))
                self._conn.row_factory = sqlite3.Row
            except Exception:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
        return self._conn

    # ── Full mode ─────────────────────────────────────────────────────────────

    def run_full(
        self,
        target_date: str,
        dry_run:     bool = False,
        tone:        str  = "hype",
    ) -> RunResult:
        """
        Full pipeline: fetch → agents → synthesize → (write) → script → table.
        """
        t0     = time.monotonic()
        result = RunResult(mode="full", target_date=target_date, dry_run=dry_run)
        log.info("=== FULL MODE  date=%s  dry_run=%s ===", target_date, dry_run)

        self._ensure_pipeline()

        # 1 — Fetch game contexts
        log.info("Step 1: fetching game contexts for %s …", target_date)
        try:
            contexts = self._context_builder(
                target_date,
                fetch_advanced=self.config.fetch_advanced,
            )
        except Exception as exc:
            import traceback
            msg = f"Context fetch failed: {exc}"
            log.error(msg)
            log.error("Full traceback:\n%s", traceback.format_exc())
            result.errors.append(msg)
            return result

        result.n_games = len(contexts)
        log.info("  %d games found", result.n_games)

        # 2 — Run agents + synthesize per game
        all_cards  = []
        all_sigs   = {}

        for ctx in contexts:
            game_label = f"{getattr(getattr(ctx,'away',None),'team','?')} @ " \
                         f"{getattr(getattr(ctx,'home',None),'team','?')}"
            log.debug("  Processing %s …", game_label)

            try:
                signals = [agent.analyze(ctx) for agent in self._agents]
                card    = self._synthesizer.synthesize(
                    signals,
                    ctx,
                    mode = self.config.synthesis_mode,
                    log  = not dry_run,      # only write to DB when not dry-run
                )
            except Exception as exc:
                log.warning("  ⚠  %s skipped: %s", game_label, exc)
                result.errors.append(f"{game_label}: {exc}")
                continue

            all_cards.append(card)
            all_sigs[game_label] = signals

            if dry_run:
                self._print_dry_run_signals(game_label, signals, card)

        result.pick_cards = all_cards
        result.n_picks    = sum(1 for c in all_cards if c.is_playable)
        result.n_high     = sum(1 for c in all_cards
                                if getattr(c, "confidence_tier", "") == "HIGH")

        # 3 — Generate Alexis script
        log.info("Step 2: generating Alexis script …")
        try:
            from script_generator import generate_script, filter_picks, preview_picks
            script           = generate_script(all_cards, tone=tone,
                                               game_date=target_date)
            result.script    = script
            result.n_in_script = len(filter_picks(all_cards))
        except Exception as exc:
            log.warning("Script generation failed: %s", exc)
            result.errors.append(f"script: {exc}")

        # 4 — Print summary table
        self._print_summary_table(all_cards, result.n_in_script, target_date,
                                  dry_run=dry_run)

        # 5 — Print script
        if result.script:
            print(f"\n{'═' * 62}")
            print(f"  ALEXIS SCRIPT  [{target_date}]  tone={tone}")
            print(f"{'═' * 62}\n")
            print(result.script)
            print(f"\n{'═' * 62}\n")

        result.elapsed_seconds = time.monotonic() - t0
        log.info("Done. %s", result)
        return result

    # ── Script-only mode ──────────────────────────────────────────────────────

    def run_script_only(
        self,
        target_date: str,
        tone:        str = "hype",
    ) -> str:
        """
        Load existing PickCards from DB for the date, regenerate script.
        """
        log.info("=== SCRIPT-ONLY MODE  date=%s ===", target_date)

        conn   = self._get_conn()
        picks  = _load_picks_from_db(conn, target_date)

        if not picks:
            log.warning("No picks found in DB for %s", target_date)
            print(f"  No picks found in database for {target_date}.")
            return ""

        log.info("  Loaded %d pick cards from DB", len(picks))

        try:
            from script_generator import generate_script, filter_picks
            script    = generate_script(picks, tone=tone, game_date=target_date)
            n_in_script = len(filter_picks(picks))
        except Exception as exc:
            log.error("Script generation failed: %s", exc)
            return ""

        print(f"\n{'═' * 62}")
        print(f"  ALEXIS SCRIPT  [{target_date}]  tone={tone}  (from DB)")
        print(f"{'═' * 62}\n")
        print(script)
        print(f"\n{'═' * 62}")
        print(f"  {n_in_script} picks included  |  {len(picks)} total in DB")
        print(f"{'═' * 62}\n")

        return script

    # ── Report mode ───────────────────────────────────────────────────────────

    def run_report(
        self,
        target_date: str,
        weeks:       int = 1,
    ):
        """
        Resolve outcomes for every date in the window (target_date back N weeks),
        then print the performance report.
        """
        from datetime import date as _date, timedelta as _td

        log.info("=== REPORT MODE  date=%s  weeks=%d ===", target_date, weeks)
        self._ensure_pipeline()

        # Build date range: target_date back N*7 days
        try:
            end = _date.fromisoformat(target_date)
        except ValueError:
            log.error("Bad --date format: %s (need YYYY-MM-DD)", target_date)
            return None
        start = end - _td(days=weeks * 7)

        # Only hit dates that actually have picks in the DB (fast + avoids API noise)
        conn = self._get_conn() if hasattr(self, "_get_conn") else None
        try:
            from sqlite3 import connect as _connect
            _conn = _connect(str(self._db_path))
            rows = _conn.execute(
                "SELECT DISTINCT game_date FROM picks "
                "WHERE game_date BETWEEN ? AND ? ORDER BY game_date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            dates = [r[0] for r in rows]
            _conn.close()
        except Exception as exc:
            log.warning("Could not enumerate pick dates (%s) — falling back to single date", exc)
            dates = [target_date]

        log.info("Resolving outcomes for %d date(s) from %s → %s …",
                 len(dates), start.isoformat(), end.isoformat())

        totals = {"fetched": 0, "final": 0, "written": 0}
        for d in dates:
            try:
                result = self._feedback.resolve_date(d)
                log.info("  %s", result)
                # Best-effort tally if attributes exist
                for k in totals:
                    totals[k] += getattr(result, f"games_{k}", 0) if k != "written" \
                                 else getattr(result, "outcomes_written", 0)
            except Exception as exc:
                log.warning("  resolve_date(%s) failed: %s", d, exc)

        log.info("Range totals  fetched=%(fetched)d  final=%(final)d  written=%(written)d",
                 totals)

        # Weekly report
        log.info("Building %d-week performance report …", weeks)
        report = self._feedback.weekly_report(weeks=weeks)
        self._feedback.print_report(report)
        return report

    # ── Summary table ─────────────────────────────────────────────────────────

    def _print_summary_table(
        self,
        pick_cards:   list,
        n_in_script:  int,
        target_date:  str,
        dry_run:      bool = False,
    ) -> None:
        """Print formatted pick summary to stdout."""
        tag = "  [DRY-RUN]" if dry_run else ""
        sep = "─" * 74

        print(f"\n{'═' * 74}")
        print(f"  NHL MAS — Pick Summary  [{target_date}]{tag}")
        print(f"{'═' * 74}")
        print(f"  {'Matchup':<18s}  {'Tier':<8s}  {'Edge%':>6s}  "
              f"{'Pick':<5s}  {'Dir':<5s}  {'Agree':>6s}  {'Agents':>6s}  Script")
        print(f"  {sep}")

        n_playable = 0
        n_high     = 0

        for card in pick_cards:
            tier      = getattr(card, "confidence_tier",    "?")
            edge      = getattr(card, "edge_pct",           0.0)
            pick_abbr = getattr(card, "pick",               "?")
            direction = getattr(card, "pick_direction",     "?")
            agree     = getattr(card, "agent_agreement_score", 0.0)
            above     = getattr(card, "agents_above_floor", 0)
            matchup   = getattr(card, "matchup",            "? @ ?")

            is_play   = getattr(card, "is_playable", tier != "SKIP")
            if is_play:
                n_playable += 1
            if tier == "HIGH":
                n_high += 1

            tier_display = {
                "HIGH": "HIGH   ", "MEDIUM": "MEDIUM ", "LOW": "LOW    ",
                "SKIP": "SKIP   ",
            }.get(tier, tier)

            edge_str  = f"{edge:+.1f}%" if is_play else " -----"
            dir_str   = direction.upper()[:4] if direction not in ("skip",) else "---"
            pick_disp = pick_abbr[:5] if is_play else "SKIP "
            n_agents  = len(self._agents) if self._agents else 5

            in_script_mark = "✓" if (
                 tier in ("HIGH", "LOW") and edge >= 5.0 and
                direction in ("home", "away")
                    ) else "✗"

            print(
                f"  {matchup:<18s}  {tier_display}  {edge_str:>6s}  "
                f"{pick_disp:<5s}  {dir_str:<5s}  {agree:>5.0%}  "
                f"{above:>3d}/{n_agents:<3d}  {in_script_mark}"
            )

        print(f"  {sep}")
        print(f"  Games: {len(pick_cards)}   Playable: {n_playable}   "
              f"High-conf: {n_high}   In script: {n_in_script}")
        print(f"{'═' * 74}\n")

    # ── Dry-run signal dump ───────────────────────────────────────────────────

    def _print_dry_run_signals(
        self,
        game_label: str,
        signals:    list,
        card,
    ) -> None:
        """Print per-agent signals for a game in dry-run mode."""
        sep = "─" * 60
        print(f"\n  [DRY-RUN] {game_label}")
        print(f"  {sep}")

        weights = getattr(self._synthesizer, "weights_registry", {})

        for sig in signals:
            aid   = getattr(sig, "agent_id",       "?")
            raw   = getattr(sig, "raw_score",      0.5)
            conf  = getattr(sig, "confidence",     0.0)
            dirn  = getattr(sig, "pick_direction", "neutral")
            w     = weights.get(aid, 1.0)
            arrow = "↑" if raw > 0.55 else ("↓" if raw < 0.45 else "→")
            print(
                f"  {arrow} {aid:<18s}  raw={raw:.3f}  "
                f"conf={conf:.3f}  → {dirn.upper():<8s}  w={w:.2f}"
            )

        tier  = getattr(card, "confidence_tier", "?")
        raw_s = getattr(card, "raw_score",       0.5)
        edge  = getattr(card, "edge_pct",        0.0)
        dirn  = getattr(card, "pick_direction",  "?")
        mode  = getattr(card, "mode_used",       "?")
        print(f"  {sep}")
        print(f"  ENSEMBLE: {raw_s:.3f} → {dirn.upper():<8s}  "
              f"edge={edge:+.1f}%  [{tier}]  [{mode}]")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_picks.py",
        description="NHL MAS Orchestrator — tie all modules together",
    )
    p.add_argument(
        "--mode", default="full",
        choices=["full", "script-only", "report"],
        help="Run mode (default: full)",
    )
    p.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Target game date (default: today)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print signals/outputs without writing to DB",
    )
    p.add_argument(
        "--tone", default=None,
        choices=["hype", "analytical"],
        help="Alexis script tone (default: from config)",
    )
    p.add_argument(
        "--weeks", type=int, default=1,
        help="Weeks of history for report mode (default: 1)",
    )
    p.add_argument(
        "--no-advanced", action="store_true",
        help="Skip NHL play-by-play in full mode (faster)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Set log level to DEBUG",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Set log level to WARNING",
    )
    p.add_argument(
        "--env", default=None, metavar="PATH",
        help="Path to .env config file (default: Boxscores/.env)",
    )
    return p.parse_args()


def main() -> int:
    args   = _parse_args()

    # Config
    env_path = Path(args.env) if args.env else None
    config   = load_config(env_path)

    # Log level: CLI flags override config
    if args.verbose:
        config.log_level = "DEBUG"
    elif args.quiet:
        config.log_level = "WARNING"
    if args.no_advanced:
        config.fetch_advanced = False

    setup_logging(config.log_level)
    log.debug("Config loaded: %s", config)

    # Resolve date
    target_date = args.date or date.today().isoformat()

    # Tone: CLI flag > config
    tone = args.tone or config.script_tone

    # Build orchestrator with real modules
    orch = MASOrchestrator(config=config)

    if args.mode == "full":
        result = orch.run_full(target_date, dry_run=args.dry_run, tone=tone)
        return 0 if not result.errors else 1

    elif args.mode == "script-only":
        script = orch.run_script_only(target_date, tone=tone)
        return 0 if script else 1

    elif args.mode == "report":
        orch.run_report(target_date, weeks=args.weeks)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
