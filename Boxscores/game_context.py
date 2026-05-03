"""
game_context.py
---------------
Module 1 output contract for the NHL Prediction MAS.

Every downstream module (2-6) receives a list[GameContext].
All fields are Optional so partial data never blocks the pipeline —
check the data quality flags (has_advanced_stats, goalie_confirmed, etc.)
before consuming a field.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Goalie ────────────────────────────────────────────────────────────────────

@dataclass
class GoalieContext:
    # Identity
    player_id:    Optional[int]   = None
    name:         str             = "Unknown"
    status:       str             = "unknown"   # confirmed / likely / expected / fallback / unknown

    # Live season stats (from DailyFaceoff scrape)
    gaa_live:     Optional[float] = None
    svpct_live:   Optional[float] = None
    wins_live:    Optional[int]   = None
    losses_live:  Optional[int]   = None

    # Rolling from goalie_features.parquet (already built)
    rolling_savePct_avg3:   Optional[float] = None
    rolling_savePct_avg5:   Optional[float] = None
    rolling_savePct_avg10:  Optional[float] = None
    rolling_gaa_avg3:       Optional[float] = None
    rolling_gaa_avg5:       Optional[float] = None
    rolling_win_rate_3:     Optional[float] = None
    rolling_win_rate_5:     Optional[float] = None
    rolling_win_rate_10:    Optional[float] = None
    rolling_shotsAgainst_avg5: Optional[float] = None

    # GSAA proxy derived from play-by-play (Module 1 new)
    # Goals Saved Above Average = saves - (shots_against * league_avg_save_pct)
    gsaa_season:  Optional[float] = None
    gsaa_last10:  Optional[float] = None

    @property
    def is_trusted(self) -> bool:
        s = self.status.lower()
        # Directly trusted statuses
        if s in ("confirmed", "likely", "expected", "probable", "manual",
                 "roster-confirmed"):
            return True
        # Inference engine result — rotation-pattern based (~78% accurate)
        if s == "inferred":
            return True
        # Conflict resolved in favour of a high-confidence candidate
        if s.startswith("conflict-"):
            inner = s[len("conflict-"):]
            return inner in ("confirmed", "likely", "expected", "probable")
        return False


# ── Team ──────────────────────────────────────────────────────────────────────

@dataclass
class TeamContext:
    team:     str  = ""
    is_home:  bool = False

    # Situational (from existing pipeline)
    rest_days:  Optional[int]  = None
    is_b2b:     bool           = False

    # Skater rolling aggregates — mean across roster (from skater_features.parquet)
    sk_goals_avg3_mean:   Optional[float] = None
    sk_goals_avg5_mean:   Optional[float] = None
    sk_goals_avg10_mean:  Optional[float] = None
    sk_goals_avg20_mean:  Optional[float] = None

    sk_shots_avg3_mean:   Optional[float] = None
    sk_shots_avg5_mean:   Optional[float] = None
    sk_shots_avg10_mean:  Optional[float] = None

    sk_points_avg5_mean:  Optional[float] = None
    sk_points_avg10_mean: Optional[float] = None

    sk_pp_goals_avg5_mean:  Optional[float] = None
    sk_pp_goals_avg10_mean: Optional[float] = None

    sk_hits_avg5_mean:      Optional[float] = None
    sk_takeaways_avg5_mean: Optional[float] = None
    sk_giveaways_avg5_mean: Optional[float] = None
    sk_plus_minus_avg5_mean: Optional[float] = None

    # Skater rolling sums — total team firepower
    sk_goals_avg5_sum:    Optional[float] = None
    sk_shots_avg5_sum:    Optional[float] = None
    sk_points_avg5_sum:   Optional[float] = None
    sk_pp_goals_avg5_sum: Optional[float] = None

    # Advanced stats — derived from NHL play-by-play API (Module 1 new)
    # All at 5v5 even-strength, rolling last 5 team games
    cf_pct_last5:     Optional[float] = None   # Corsi For % (all shot attempts)
    ff_pct_last5:     Optional[float] = None   # Fenwick For % (unblocked shots)
    xg_for_last5:     Optional[float] = None   # Expected goals for
    xg_against_last5: Optional[float] = None   # Expected goals against
    xg_pct_last5:     Optional[float] = None   # xGF / (xGF + xGA)
    sh_attempts_last5: Optional[float] = None  # Raw shot attempts per game

    # Goalie
    goalie: Optional[GoalieContext] = None

    # News / injury headlines (from DailyFaceoff — now persisted, not dropped)
    news_headlines: list = field(default_factory=list)


# ── Odds ──────────────────────────────────────────────────────────────────────

@dataclass
class OddsContext:
    # Current lines (from DailyFaceoff scrape)
    home_ml:       Optional[float] = None
    away_ml:       Optional[float] = None
    point_spread:  Optional[float] = None
    home_implied:  Optional[float] = None   # implied prob from moneyline
    away_implied:  Optional[float] = None

    # Line movement — requires persisted odds_history.csv (Phase 2)
    # Populated once we have ≥2 scraped snapshots for a game
    opening_home_ml:  Optional[float] = None
    opening_away_ml:  Optional[float] = None
    line_movement:    Optional[float] = None   # current_home_ml - opening_home_ml

    scraped_at: Optional[datetime] = None

    @property
    def has_movement(self) -> bool:
        return self.opening_home_ml is not None and self.home_ml is not None


# ── Game (top-level output contract) ─────────────────────────────────────────

@dataclass
class GameContext:
    # Identity
    game_id:    int   = 0
    game_date:  str   = ""
    start_time: str   = ""
    season:     str   = ""
    game_type:  int   = 2   # 2 = regular season, 3 = playoffs

    # Teams
    home: TeamContext = field(default_factory=TeamContext)
    away: TeamContext = field(default_factory=TeamContext)

    # Odds
    odds: OddsContext = field(default_factory=OddsContext)

    # Metadata
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    # ── Data quality flags — checked by downstream modules ────────────
    goalie_confirmed:    bool = False   # both starters trusted
    has_advanced_stats:  bool = False   # xG/CF% populated
    has_odds:            bool = False   # moneyline populated
    has_news:            bool = False   # at least one news headline

    def __str__(self):
        g_flag  = "✅" if self.goalie_confirmed   else "❓"
        adv_flag = "✅" if self.has_advanced_stats else "⚠"
        odds_flag = "✅" if self.has_odds          else "⚠"
        return (
            f"GameContext [{self.game_date}] "
            f"{self.away.team} @ {self.home.team}  "
            f"goalies:{g_flag} adv:{adv_flag} odds:{odds_flag}"
        )

    def __repr__(self):
        return self.__str__()
