"""
agent_base.py
-------------
Base class and signal contract for all NHL MAS agents.

Every agent:
  - Inherits NHLAgent
  - Implements analyze(game_context) -> AgentSignal
  - Has a weight (float) updated by the Module 3 meta-learner
  - Has a confidence_floor — signals below this are suppressed

AgentSignal fields:
  agent_id        str           which agent produced this
  pick_direction  str           "home" | "away" | "neutral"
  raw_score       float 0-1     >0.5 = home lean, <0.5 = away lean
  confidence      float 0-1     how certain the agent is
  reasoning       str           human-readable explanation for logging
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime

PickDirection = Literal["home", "away", "neutral"]


# ── Signal contract ───────────────────────────────────────────────────────────

@dataclass
class AgentSignal:
    agent_id:       str
    pick_direction: PickDirection   # "home" | "away" | "neutral"
    raw_score:      float           # 0.0–1.0   (0.5 = toss-up)
    confidence:     float           # 0.0–1.0   (how certain the agent is)
    reasoning:      str             # log-friendly explanation

    # Optional metadata (populated by some agents)
    factors:        dict  = field(default_factory=dict)   # sub-factor breakdown
    produced_at:    Optional[datetime] = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self.raw_score  = max(0.0, min(1.0, float(self.raw_score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.pick_direction not in ("home", "away", "neutral"):
            raise ValueError(f"pick_direction must be home/away/neutral, got {self.pick_direction!r}")

    @property
    def home_edge(self) -> float:
        """Signed edge: positive = home lean, negative = away lean."""
        return self.raw_score - 0.5

    @property
    def weighted_score(self) -> float:
        """raw_score weighted by confidence — used by meta-learner."""
        return 0.5 + (self.raw_score - 0.5) * self.confidence

    def __str__(self) -> str:
        direction = "→HOME" if self.pick_direction == "home" else (
                    "→AWAY" if self.pick_direction == "away" else "  ~  ")
        return (f"[{self.agent_id:<18s}] {direction}  "
                f"score={self.raw_score:.3f}  conf={self.confidence:.3f}  "
                f"| {self.reasoning[:80]}")


# ── Base agent ────────────────────────────────────────────────────────────────

class NHLAgent(ABC):
    """
    Abstract base class for all NHL prediction agents.

    Subclass this and implement analyze().
    The meta-learner (Module 3) adjusts self.weight over time based on
    each agent's historical accuracy per game context type.
    """

    def __init__(
        self,
        agent_id:         str,
        weight:           float = 1.0,
        confidence_floor: float = 0.52,
    ):
        """
        Args:
            agent_id:         Unique identifier string (e.g. "team_form")
            weight:           Initial ensemble weight. Updated by meta-learner.
            confidence_floor: Signals below this confidence are flagged as low-quality.
        """
        self.agent_id         = agent_id
        self.weight           = weight
        self.confidence_floor = confidence_floor

    @abstractmethod
    def analyze(self, game_context) -> AgentSignal:
        """
        Analyze a GameContext and return an AgentSignal.

        Must never raise — return a neutral signal with low confidence
        if data is unavailable rather than throwing.
        """

    def is_signal_valid(self, signal: AgentSignal) -> bool:
        """True if signal clears the confidence floor."""
        return signal.confidence >= self.confidence_floor

    def neutral_signal(self, reason: str = "Insufficient data") -> AgentSignal:
        """Convenience: return a neutral no-opinion signal."""
        return AgentSignal(
            agent_id       = self.agent_id,
            pick_direction = "neutral",
            raw_score      = 0.5,
            confidence     = 0.0,
            reasoning      = reason,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id!r}, weight={self.weight:.2f})"


# ── Utility: logistic helper ──────────────────────────────────────────────────

def logistic(x: float, scale: float = 1.0) -> float:
    """Map any real number to (0, 1) using the logistic function."""
    import math
    return 1.0 / (1.0 + math.exp(-x * scale))


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))
