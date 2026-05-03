"""
verify_fix.py — quick diagnostic to confirm agent confidence values post-fix.
Run from Boxscores/ directory: python verify_fix.py
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from agents.schedule_agent import ScheduleAgent
from agents.goalie_form_agent import GoalieFormAgent
from agents.player_form_agent import PlayerFormAgent
from agents.team_form_agent import TeamFormAgent
from agents.sentiment_agent import SentimentAgent
from game_context import TeamContext, GoalieContext
from datetime import date

FLOOR = 0.52


class FakeCtx:
    def __init__(self, home, away):
        self.home = home
        self.away = away


def make_goalie(status, has_rolling=True):
    return GoalieContext(
        name='Test Goalie', status=status,
        rolling_savePct_avg5 = 0.910 if has_rolling else None,
        rolling_savePct_avg3 = 0.912 if has_rolling else None,
        svpct_live           = 0.908 if has_rolling else None,
        gsaa_season          = 5.0   if has_rolling else None,
    )


def flag(conf):
    return f"{'ABOVE' if conf >= FLOOR else 'BELOW'} floor"


sa  = ScheduleAgent()
ga  = GoalieFormAgent(gl=pd.DataFrame())
pfa = PlayerFormAgent(sk=pd.read_parquet('skater_features.parquet'))
tfa = TeamFormAgent(target_date=date(2026, 4, 11))
tfa._elo_ratings = {'BOS': 1550.0, 'TOR': 1500.0}
snt = SentimentAgent()

print()
print("=" * 62)
print("  SCHEDULE AGENT  (fixed base_conf = 0.50)")
print("=" * 62)
schedule_scenarios = [
    ("Normal night (no B2B, rest diff=1)",  2, False, 1, False),
    ("Both equal rest, no B2B",             2, False, 2, False),
    ("B2B mismatch (away tired)",           2, False, 1, True),
    ("Big rest diff (3 vs 1)",              3, False, 1, False),
]
for label, hr, hb, ar, ab in schedule_scenarios:
    home = TeamContext(team='BOS', is_home=True,  news_headlines=[], rest_days=hr, is_b2b=hb)
    away = TeamContext(team='TOR', is_home=False, news_headlines=[], rest_days=ar, is_b2b=ab)
    sig = sa.analyze(FakeCtx(home, away))
    print(f"  {label:<42s}  conf={sig.confidence:.3f}  {flag(sig.confidence)}")

print()
print("=" * 62)
print("  GOALIE AGENT  (fixed base_conf=0.62, two-tier penalties)")
print("=" * 62)
goalie_scenarios = [
    ("Both CONFIRMED, rolling data",        "confirmed", "confirmed", True,  True),
    ("Both UNKNOWN, rolling data exists",   "unknown",   "unknown",   True,  True),
    ("One UNKNOWN (data), one CONFIRMED",   "unknown",   "confirmed", True,  True),
    ("Both UNKNOWN, NO data at all",        "unknown",   "unknown",   False, False),
]
for label, h_status, a_status, h_data, a_data in goalie_scenarios:
    home = TeamContext(team='BOS', is_home=True, news_headlines=[],
                       rest_days=2, is_b2b=False,
                       goalie=make_goalie(h_status, h_data))
    away = TeamContext(team='TOR', is_home=False, news_headlines=[],
                       rest_days=1, is_b2b=False,
                       goalie=make_goalie(a_status, a_data))
    sig = ga.analyze(FakeCtx(home, away))
    print(f"  {label:<46s}  conf={sig.confidence:.3f}  {flag(sig.confidence)}")

print()
print("=" * 62)
print("  FULL 5-AGENT SIMULATION")
print("  BOS @ TOR — both goalies unknown, rolling parquet data present")
print("  (representative of a typical game at run-time)")
print("=" * 62)
home = TeamContext(
    team='BOS', is_home=True, news_headlines=[], rest_days=2, is_b2b=False,
    sk_points_avg5_mean=0.55, sk_pp_goals_avg5_sum=2.3, sk_goals_avg10_mean=0.27,
    cf_pct_last5=0.525, xg_for_last5=2.85, xg_against_last5=2.40,
    goalie=make_goalie('unknown', True),
)
away = TeamContext(
    team='TOR', is_home=False, news_headlines=[], rest_days=1, is_b2b=False,
    sk_points_avg5_mean=0.50, sk_pp_goals_avg5_sum=1.9, sk_goals_avg10_mean=0.25,
    cf_pct_last5=0.490, xg_for_last5=2.50, xg_against_last5=2.70,
    goalie=make_goalie('unknown', True),
)
ctx = FakeCtx(home, away)

print(f"  {'Agent':<20s} {'RawScore':>10s} {'Confidence':>12s} {'Direction':<10s} {'AboveFloor?'}")
print(f"  {'-' * 66}")
above_dir = 0
above_all = 0
for agent in [tfa, pfa, ga, sa, snt]:
    sig = agent.analyze(ctx)
    is_dir = agent.agent_id not in ('sentiment',)
    above = sig.confidence >= FLOOR
    if above:
        above_all += 1
    if above and is_dir:
        above_dir += 1
    marker = "YES" if above else " no"
    print(f"  {agent.agent_id:<20s} {sig.raw_score:>10.4f} {sig.confidence:>12.4f} {sig.pick_direction:<10s} {marker}")

print(f"  {'-' * 66}")
print(f"  Directional above floor: {above_dir}/4  (need 3 to avoid SKIP)")
print(f"  All agents above floor:  {above_all}/5  (displayed in Agents column)")
print(f"  Result: {'NO SKIP  ✅' if above_dir >= 3 else 'SKIP  ❌'}")
print()
