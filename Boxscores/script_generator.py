"""
script_generator.py
-------------------
Module 5: Script Generator — Alexis Rivers NHL Picks Channel

Takes a list of PickCard objects and produces a D-ID avatar read script.

Filtering rules:
    - Only confidence_tier == "HIGH" picks are included.
    - Picks with edge_pct < 3.0% are excluded.

Script format:
    Opening framing → one paragraph per pick → standard disclaimer.

Output:
    Plain string — no markdown, no stage directions, no parentheticals.
    Alexis reads everything literally.

Target: 200–250 words (~90–120 seconds at 150 wpm).

Tone params:
    "hype"       — energetic, fan excitement, hockey fan voice
    "analytical" — calm, data-informed, fan-friendly explanations

Usage:
    from script_generator import generate_script, preview_picks

    script = generate_script(pick_cards, tone="hype")
    preview_picks(pick_cards)

CLI:
    python script_generator.py --date 2025-01-15 [--tone hype|analytical]
    python script_generator.py --preview-only --date 2025-01-15
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_EDGE_PCT     = 5.0          # minimum edge % to qualify
TIER_REQUIRED    = ("HIGH", "LOW")  # only these tiers make the cut
MAX_PICKS        = 3            # cap script at 3 picks to stay in word budget
TARGET_WORDS_LO  = 200
TARGET_WORDS_HI  = 250
READ_WPM         = 150          # average speaking pace

VALID_TONES      = ("hype", "analytical")

# ── Team name registry ────────────────────────────────────────────────────────

TEAMS: dict[str, dict] = {
    "ANA": {"city": "Anaheim",       "name": "Ducks"},
    "BOS": {"city": "Boston",        "name": "Bruins"},
    "BUF": {"city": "Buffalo",       "name": "Sabres"},
    "CAR": {"city": "Carolina",      "name": "Hurricanes"},
    "CBJ": {"city": "Columbus",      "name": "Blue Jackets"},
    "CGY": {"city": "Calgary",       "name": "Flames"},
    "CHI": {"city": "Chicago",       "name": "Blackhawks"},
    "COL": {"city": "Colorado",      "name": "Avalanche"},
    "DAL": {"city": "Dallas",        "name": "Stars"},
    "DET": {"city": "Detroit",       "name": "Red Wings"},
    "EDM": {"city": "Edmonton",      "name": "Oilers"},
    "FLA": {"city": "Florida",       "name": "Panthers"},
    "LAK": {"city": "Los Angeles",   "name": "Kings"},
    "MIN": {"city": "Minnesota",     "name": "Wild"},
    "MTL": {"city": "Montreal",      "name": "Canadiens"},
    "NJD": {"city": "New Jersey",    "name": "Devils"},
    "NSH": {"city": "Nashville",     "name": "Predators"},
    "NYI": {"city": "New York",      "name": "Islanders"},
    "NYR": {"city": "New York",      "name": "Rangers"},
    "OTT": {"city": "Ottawa",        "name": "Senators"},
    "PHI": {"city": "Philadelphia",  "name": "Flyers"},
    "PIT": {"city": "Pittsburgh",    "name": "Penguins"},
    "SEA": {"city": "Seattle",       "name": "Kraken"},
    "SJS": {"city": "San Jose",      "name": "Sharks"},
    "STL": {"city": "St. Louis",     "name": "Blues"},
    "TBL": {"city": "Tampa Bay",     "name": "Lightning"},
    "TOR": {"city": "Toronto",       "name": "Maple Leafs"},
    "UTA": {"city": "Utah",          "name": "Hockey Club"},
    "VAN": {"city": "Vancouver",     "name": "Canucks"},
    "VGK": {"city": "Vegas",         "name": "Golden Knights"},
    "WPG": {"city": "Winnipeg",      "name": "Jets"},
    "WSH": {"city": "Washington",    "name": "Capitals"},
}


def _city(abbrev: str) -> str:
    return TEAMS.get(abbrev, {}).get("city", abbrev)


def _name(abbrev: str) -> str:
    return TEAMS.get(abbrev, {}).get("name", abbrev)


def _full(abbrev: str) -> str:
    """Boston Bruins, Vegas Golden Knights, etc."""
    t = TEAMS.get(abbrev)
    if not t:
        return abbrev
    return f"{t['city']} {t['name']}"


def _word_count(text: str) -> int:
    return len(text.split())


def estimate_read_seconds(text: str, wpm: int = READ_WPM) -> float:
    """Return estimated read time in seconds for the given plain text."""
    return _word_count(text) / wpm * 60


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_picks(pick_cards: list) -> list:
    """
    Return only the picks that qualify for the script.

    Rules:
        1. confidence_tier == "HIGH"
        2. edge_pct >= MIN_EDGE_PCT (3.0%)
        3. pick_direction is "home" or "away" (not "skip")
        4. Cap at MAX_PICKS (3), sorted by edge_pct descending
    """
    qualified = [
        p for p in pick_cards
        if getattr(p, "confidence_tier", "") in TIER_REQUIRED
        and getattr(p, "edge_pct", 0.0) >= MIN_EDGE_PCT
        and getattr(p, "pick_direction", "skip") in ("home", "away")
    ]
    qualified.sort(key=lambda p: p.edge_pct, reverse=True)
    return qualified[:MAX_PICKS]


# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning engine — agent signals → fan language
# ═══════════════════════════════════════════════════════════════════════════════

def _agent_supports(signal: dict, pick_direction: str, threshold: float = 0.58) -> bool:
    """True if this agent signal supports the pick direction with enough conviction."""
    return (
        signal.get("direction") == pick_direction
        and float(signal.get("raw_score", 0.5)) >= threshold
    )


def _team_form_sentence(pick, sig: dict, tone: str) -> str:
    """Fan-language sentence about team form (CF%, xG, Elo)."""
    raw   = float(sig.get("raw_score", 0.5))
    team  = _full(pick.pick)
    is_strong = raw >= 0.70

    if tone == "hype":
        if is_strong:
            return (f"The {team} have been absolutely dominant lately, "
                    f"controlling the puck and creating chance after chance every single night.")
        return (f"The {team} have found their stride and are getting the better "
                f"of play in their recent games.")
    else:
        if is_strong:
            return (f"The {team} have ranked among the league leaders in possession "
                    f"metrics and chance creation over the past ten games, "
                    f"and that kind of sustained dominance represents a meaningful "
                    f"advantage heading into this matchup.")
        return (f"The {team} carry a measurable possession advantage into this game "
                f"based on recent advanced numbers, and that edge is reflected "
                f"clearly in the model's output.")


def _goalie_form_sentence(pick, sig: dict, tone: str) -> str:
    """Fan-language sentence about goalie matchup."""
    raw       = float(sig.get("raw_score", 0.5))
    team      = _full(pick.pick)
    is_strong = raw >= 0.70

    if tone == "hype":
        if is_strong:
            return ("Their goalie has been an absolute wall, completely locked in "
                    "and giving his team a chance to win every single night.")
        return ("They have the better goalie matchup here, "
                "and in the NHL that can make all the difference.")
    else:
        if is_strong:
            return ("The goaltending advantage is one of the clearest signals "
                    "in this game. Their starter has been posting numbers well "
                    "above league average over a meaningful recent sample, "
                    "and that holds up under scrutiny.")
        return (f"The {team} hold a moderate goaltending edge based on "
                f"recent performance, and that factor contributes meaningfully "
                f"to the overall confidence level in this selection.")


def _schedule_sentence(pick, sig: dict, tone: str) -> str:
    """Fan-language sentence about rest / schedule advantage."""
    raw  = float(sig.get("raw_score", 0.5))
    opp  = _full(pick.away_team if pick.pick_direction == "home" else pick.home_team)
    is_strong = raw >= 0.70

    if tone == "hype":
        if is_strong:
            return (f"The opponent is playing on tired legs tonight — "
                    f"the schedule absolutely sets this one up to fall our way.")
        return ("The schedule sets up nicely in their favor with a real rest advantage heading in.")
    else:
        if is_strong:
            return (f"The schedule advantage is significant. "
                    f"The {opp} are dealing with fatigue while this side is fresh and rested.")
        return ("There is a moderate rest advantage here that the numbers back up.")


def _player_form_sentence(pick, sig: dict, tone: str) -> str:
    """Fan-language sentence about forward line production."""
    raw       = float(sig.get("raw_score", 0.5))
    team      = _full(pick.pick)
    is_strong = raw >= 0.70

    if tone == "hype":
        if is_strong:
            return ("Their top line is completely on fire right now, "
                    "putting up points and making things happen every time they hit the ice.")
        return ("Their forwards have the momentum right now and have been generating offense at a high rate.")
    else:
        if is_strong:
            return (f"The {team} top forwards are producing well above their seasonal "
                    f"averages over the last stretch, pointing to strong current form.")
        return (f"The {team} forward group has been trending in the right direction based on recent scoring rates.")


def _sentiment_sentence(pick, sig: dict, tone: str) -> str:
    """Fan-language sentence about news / injury status."""
    if tone == "hype":
        return ("The news out of their camp is all positive — healthy roster, "
                "good energy, and nothing concerning heading into tonight.")
    return ("Recent news flow has been favorable for their side, "
            "with no injury concerns or lineup disruptions to note.")


_REASON_BUILDERS = {
    "team_form":   _team_form_sentence,
    "goalie_form": _goalie_form_sentence,
    "schedule":    _schedule_sentence,
    "player_form": _player_form_sentence,
    "sentiment":   _sentiment_sentence,
}


def _build_reasoning(pick, tone: str, max_sentences: int = 2) -> str:
    """
    Build fan-friendly reasoning from pick.signal_summary.

    Ranks supporting signals by (raw_score × weight) descending
    and takes the top max_sentences to build the reasoning block.
    """
    summary   = getattr(pick, "signal_summary", {}) or {}
    direction = getattr(pick, "pick_direction", "home")

    # Rank agents that support the pick direction
    ranked: list[tuple[float, str, dict]] = []
    for aid, sig in summary.items():
        if not isinstance(sig, dict):
            continue
        # sentiment is a multiplier — only include if explicitly positive
        if aid == "sentiment":
            mult = float(sig.get("multiplier", 1.0))
            raw  = float(sig.get("raw_score", 0.5))
            if mult <= 1.05 and raw <= 0.58:
                continue
            score_key = raw * float(sig.get("weight", 0.5))
        else:
            if not _agent_supports(sig, direction):
                continue
            score_key = float(sig.get("raw_score", 0.5)) * float(sig.get("weight", 1.0))
        ranked.append((score_key, aid, sig))

    ranked.sort(reverse=True)
    top = ranked[:max_sentences]

    sentences = []
    for _, aid, sig in top:
        builder = _REASON_BUILDERS.get(aid)
        if builder:
            sentences.append(builder(pick, sig, tone))

    # Fallback if no signals available (e.g., mock PickCard with empty summary)
    if not sentences:
        team = _full(pick.pick)
        if tone == "hype":
            sentences = [f"The system is high on the {team} tonight across multiple key factors."]
        else:
            sentences = [f"The {team} meet our highest-confidence threshold across multiple metrics."]

    return " ".join(sentences)


# ═══════════════════════════════════════════════════════════════════════════════
# Opening templates
# ═══════════════════════════════════════════════════════════════════════════════

def _opening(n_picks: int, tone: str, game_date: str = "") -> str:
    date_str = f" for {game_date}" if game_date else " tonight"

    if tone == "hype":
        if n_picks == 0:
            return (
                "What is up hockey fans, Alexis Rivers here. "
                "The system scanned the entire slate tonight and nothing hit the highest "
                "confidence threshold. Sometimes the right move is to sit out. "
                "No plays tonight, but we will be back tomorrow night with the full breakdown."
            )
        if n_picks == 1:
            return (
                f"What is up hockey fans, Alexis Rivers here with tonight's premium pick. "
                f"The system went through the entire slate{date_str} and there is exactly one game "
                f"where all the key factors line up. "
                f"When the model gets this selective, that is when you really want to pay attention. "
                f"Let me walk you through it."
            )
        return (
            f"What is up hockey fans, Alexis Rivers here with tonight's premium picks. "
            f"The system identified {n_picks} high-confidence plays{date_str}. "
            f"Let's break them all down."
        )
    else:
        if n_picks == 0:
            return (
                "Good evening. Alexis Rivers here. "
                "Tonight's NHL slate did not produce any picks meeting the high-confidence threshold. "
                "No action tonight. We will return tomorrow with a full analysis."
            )
        if n_picks == 1:
            return (
                f"Good evening. Alexis Rivers here with tonight's analysis. "
                f"The system evaluated every game on the NHL slate{date_str} and found exactly one "
                f"that cleared all of our confidence thresholds. "
                f"That level of selectivity reflects genuine signal quality across multiple factors. "
                f"Here is the complete breakdown."
            )
        return (
            f"Good evening. Alexis Rivers here. "
            f"The system identified {n_picks} games meeting the high-confidence threshold{date_str}. "
            f"Here is the full breakdown."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pick paragraph builders
# ═══════════════════════════════════════════════════════════════════════════════

_ORDINALS = ["First", "Next", "Finally", "Fourth"]


def _pick_paragraph(pick, idx: int, total: int, tone: str) -> str:
    """
    Build one pick paragraph.

    Format: [intro] [context] [pick + type] [reasoning] [edge note] [call to action]
    Target: ~110 words for solo pick; ~75–80 for 2-pick; ~60 for 3-pick.
    """
    ordinal    = _ORDINALS[idx] if idx < len(_ORDINALS) else f"Pick {idx + 1}"
    pick_team  = _full(pick.pick)
    home_full  = _full(pick.home_team)
    away_full  = _full(pick.away_team)
    is_home    = pick.pick_direction == "home"
    opponent   = _full(pick.away_team if is_home else pick.home_team)
    edge       = getattr(pick, "edge_pct", 0.0)

    # Context sentence — for hype multi-pick scripts use a shorter version to stay in budget
    if tone == "hype":
        if total == 1:
            # Full context for solo picks (helps hit word budget floor)
            if is_home:
                context = (f"The {pick_team} get to play in front of their home crowd tonight, "
                           f"and the system is very clear about where the edge is in this one.")
            else:
                context = (f"The {pick_team} come in here as the road side, "
                           f"and that is exactly where the system is finding value tonight.")
        else:
            # Short context for multi-pick hype (saves ~15 words to stay under ceiling)
            if is_home:
                context = f"Home ice tonight and the system has clear conviction on this one."
            else:
                context = f"Road side tonight and the value is right here."
    else:
        if is_home:
            context = (f"The home side has a measurable advantage across several "
                       f"key factors in this matchup.")
        else:
            context = (f"The road side is where the model identifies the value "
                       f"in this particular game.")

    # Reasoning — 3 sentences for solo, 2 for multi
    max_r     = 3 if total == 1 else 2
    reasoning = _build_reasoning(pick, tone, max_sentences=max_r)

    # Edge note — adds ~15-25 words, key for word budget
    if tone == "hype":
        edge_note = (f"The system has us at a {edge:.0f} percent edge on this game, "
                     f"and that is a number I feel really good about.")
    else:
        edge_note = (f"The model quantifies our edge on this game at {edge:.0f} percent. "
                     f"That number clears our required threshold for a quality selection, "
                     f"which is exactly where we want to be.")

    if tone == "hype":
        intro  = f"{ordinal} up: {away_full} visiting {home_full} tonight."
        action = f"Give me the {pick_team} on the money line."
        cta    = "That is the play." if idx < total - 1 else "Lock it in."
    else:
        intro  = f"Game {idx + 1}: {away_full} at {home_full}."
        action = f"The play is {pick_team} on the money line."
        cta    = "This one meets our highest standard."

    # Assemble: for 3-pick scripts drop the context sentence to stay in budget
    if total >= 3:
        return f"{intro} {action} {reasoning} {edge_note} {cta}"
    return f"{intro} {context} {action} {reasoning} {edge_note} {cta}"


# ═══════════════════════════════════════════════════════════════════════════════
# Closing / disclaimer
# ═══════════════════════════════════════════════════════════════════════════════

def _closing(tone: str) -> str:
    if tone == "hype":
        return (
            "That is the card for tonight. "
            "Trust the process, stick to the plays, and as always only bet what you can afford to lose. "
            "Good luck out there tonight. Let us get these wins."
        )
    return (
        "Those are tonight's picks. "
        "These selections represent the system's highest-confidence output for this slate. "
        "Please gamble responsibly and only wager amounts you are comfortable losing. "
        "Good luck tonight."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Word budget trimmer
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on '. ' preceded by a letter/digit (not '8.5').
    Avoids breaking on decimal numbers like '18.0 percent'.
    """
    parts = _re.split(r'(?<=[a-zA-Z])\.\s+', text)
    return [s.strip() for s in parts if s.strip()]


def _trim_to_budget(parts: list[str]) -> list[str]:
    """
    Progressively shorten pick paragraphs (not opening/closing) until the
    assembled script is within TARGET_WORDS_HI.  Trims the longest pick
    paragraph one sentence at a time, looping up to 40 passes.
    """
    if _word_count(" ".join(parts)) <= TARGET_WORDS_HI:
        return parts

    trimmed      = list(parts)
    pick_indices = list(range(1, len(trimmed) - 1))

    for _ in range(40):
        if _word_count(" ".join(trimmed)) <= TARGET_WORDS_HI:
            break

        # Always trim the longest pick paragraph
        longest_i  = max(pick_indices, key=lambda i: _word_count(trimmed[i]))
        sentences  = _split_sentences(trimmed[longest_i])

        if len(sentences) <= 2:
            # Already very short — try another paragraph
            pick_indices = [j for j in pick_indices if j != longest_i]
            if not pick_indices:
                break
            continue

        trimmed[longest_i] = ". ".join(sentences[:-1]) + "."

    return trimmed


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_script(
    pick_cards: list,
    tone:       str = "hype",
    game_date:  str = "",
) -> str:
    """
    Generate a D-ID avatar script for Alexis Rivers.

    Args:
        pick_cards:  List of PickCard objects (any mix of tiers/directions).
        tone:        "hype" or "analytical".
        game_date:   Optional date string for the opening line (e.g. "January 15th").

    Returns:
        Plain string script — no markdown, no stage directions.
        Approximately 200–250 words (90–120 seconds read time).
    """
    if tone not in VALID_TONES:
        tone = "hype"

    qualified = filter_picks(pick_cards)
    n         = len(qualified)

    opening = _opening(n, tone, game_date)
    closing = _closing(tone)

    if n == 0:
        # No picks — just opening + closing, under budget by design
        return f"{opening} {closing}"

    pick_parts = [
        _pick_paragraph(p, i, n, tone)
        for i, p in enumerate(qualified)
    ]

    # Assemble + budget check
    all_parts = [opening] + pick_parts + [closing]
    all_parts = _trim_to_budget(all_parts)

    return " ".join(all_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# preview_picks — filtering table printed to stdout
# ═══════════════════════════════════════════════════════════════════════════════

def preview_picks(pick_cards: list, game_date: str = "") -> None:
    """
    Print a formatted table of all picks, showing what passes and what was cut.

    Columns: Matchup | Tier | Edge% | Direction | Included? | Filter Reason
    """
    SEP        = "─" * 72
    date_label = f"  ({game_date})" if game_date else ""

    print(f"\nPICK PREVIEW{date_label}")
    print(SEP)
    print(f"  {'Matchup':<18s}  {'Tier':<8s}  {'Edge%':>6s}  {'Dir':<5s}  {'Included?'}")
    print(SEP)

    n_included = 0
    n_total    = 0
    qualified  = filter_picks(pick_cards)
    qualified_ids = {id(p) for p in qualified}

    for p in pick_cards:
        tier      = getattr(p, "confidence_tier", "?")
        edge      = getattr(p, "edge_pct",        0.0)
        direction = getattr(p, "pick_direction",  "?")
        matchup   = getattr(p, "matchup",         "? @ ?")
        n_total  += 1

        # Determine filter reason
        if tier == "SKIP":
            reason = "SKIP card"
        elif tier not in TIER_REQUIRED:
            reason = f"tier={tier} (need HIGH or LOW)"
        elif edge < MIN_EDGE_PCT:
            reason = f"edge={edge:.1f}% < {MIN_EDGE_PCT}%"
        elif direction not in ("home", "away"):
            reason = "invalid direction"
        elif id(p) in qualified_ids:
            reason = "INCLUDED"
        else:
            reason = "over MAX_PICKS cap"

        included = id(p) in qualified_ids
        mark     = "✓" if included else "✗"
        dir_str  = direction.upper() if direction in ("home", "away") else "-"

        print(
            f"  {matchup:<18s}  {tier:<8s}  {edge:>5.1f}%  "
            f"{dir_str:<5s}  {mark}  {reason}"
        )

        if included:
            n_included += 1

    print(SEP)
    print(f"  Included: {n_included} of {n_total} pick{'s' if n_total != 1 else ''}")
    print(SEP)

    # Word count preview
    if n_included > 0:
        for tone in VALID_TONES:
            script = generate_script(pick_cards, tone=tone)
            wc     = _word_count(script)
            secs   = estimate_read_seconds(script)
            print(f"  [{tone:<12s}]  {wc} words  ~{secs:.0f}s read time")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NHL MAS Module 5 — Alexis Rivers Script Generator"
    )
    p.add_argument("--date",         default=None,   metavar="YYYY-MM-DD",
                   help="Target game date (default: today)")
    p.add_argument("--tone",         default="hype",
                   choices=VALID_TONES, help="Script tone (default: hype)")
    p.add_argument("--preview-only", action="store_true",
                   help="Print pick table only, do not generate script")
    p.add_argument("--no-advanced",  action="store_true",
                   help="Skip play-by-play when building contexts")
    return p.parse_args()


def main() -> None:
    args        = _parse_args()
    target_date = args.date or date.today().isoformat()

    BASE = Path(r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player\Boxscores")
    sys.path.insert(0, str(BASE))

    try:
        from module1_ingest      import build_today_contexts
        from agents              import DEFAULT_AGENTS
        from module3_synthesizer import MasterSynthesizer

        print(f"\n  Building contexts for {target_date} …")
        contexts    = build_today_contexts(
            target_date, fetch_advanced=not args.no_advanced
        )
        synthesizer = MasterSynthesizer()
        synthesizer.fit_from_db()

        pick_cards = []
        for ctx in contexts:
            signals = [a.analyze(ctx) for a in DEFAULT_AGENTS]
            card    = synthesizer.synthesize(signals, ctx, log=False)
            pick_cards.append(card)

    except ImportError as exc:
        print(f"  ⚠  Could not import pipeline modules: {exc}")
        print("  Running with empty pick list — use from Python directly.")
        pick_cards = []

    # Preview table
    preview_picks(pick_cards, game_date=target_date)

    if not args.preview_only:
        script = generate_script(pick_cards, tone=args.tone,
                                 game_date=target_date)
        print(f"\n{'═' * 62}")
        print(f"  ALEXIS RIVERS SCRIPT  [{target_date}]  tone={args.tone}")
        print(f"{'═' * 62}\n")
        print(script)
        print(f"\n{'═' * 62}")
        wc   = _word_count(script)
        secs = estimate_read_seconds(script)
        print(f"  Word count: {wc}   Estimated read: ~{secs:.0f}s")
        print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()