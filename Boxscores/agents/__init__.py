"""
agents/__init__.py
NHL MAS — Module 2: Agent Framework
"""

from .agent_base        import NHLAgent, AgentSignal, logistic, clamp
from .team_form_agent   import TeamFormAgent
from .player_form_agent import PlayerFormAgent
from .goalie_form_agent import GoalieFormAgent
from .schedule_agent    import ScheduleAgent
from .sentiment_agent   import SentimentAgent

__all__ = [
    "NHLAgent",
    "AgentSignal",
    "TeamFormAgent",
    "PlayerFormAgent",
    "GoalieFormAgent",
    "ScheduleAgent",
    "SentimentAgent",
]

# Default agent ensemble with initial weights
# Weights are updated by Module 3 meta-learner over time
DEFAULT_AGENTS = [
    TeamFormAgent(weight=1.0),
    PlayerFormAgent(weight=0.9),
    GoalieFormAgent(weight=1.2),    # highest — goalies drive outcomes
    ScheduleAgent(weight=0.8),
    SentimentAgent(weight=0.5),     # multiplier role, lower direct weight
]
