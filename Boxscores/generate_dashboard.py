"""
generate_dashboard.py - NHL MAS Performance Dashboard Renderer
===============================================================
Produces public-facing PNG dashboards for YouTube, Reddit, IG, and channel banners.
Reads the same prediction_log.csv used by mas_effectiveness.py and renders
visual cards that show Season / HIGH-tier / Last-N-days performance plus
a model calibration line and brand signature.

Usage:
    python generate_dashboard.py                 # all 3 sizes, last-7-day window
    python generate_dashboard.py --size youtube  # one size only (youtube|square|wide)
    python generate_dashboard.py --days 14       # override the "last N days" window
    python generate_dashboard.py --today 2026-04-18  # inject a date (for backfill/testing)

Outputs (to ./dashboards/):
    dashboard_YYYY-MM-DD_youtube.png   1200x900   (YouTube community post)
    dashboard_YYYY-MM-DD_square.png    1080x1080  (Reddit, Instagram)
    dashboard_YYYY-MM-DD_wide.png      1920x1080  (YT end screen, banner)

NOTE: The helper functions tier_label(), roi_pct(), and BREAKEVEN are duplicated
from mas_effectiveness.py to guarantee the visual dashboard and the console
report never drift. If you change tier cutoffs or vig math in one, change both.
A future refactor could lift these into a shared metrics_core.py module.
"""

import argparse
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image


# -- Config --------------------------------------------------------------------
BASE        = Path(__file__).resolve().parent
LOG_FILE    = BASE / "prediction_log.csv"
LOGO_FILE   = BASE / "smart_thinking_logo.png"
OUTPUT_DIR  = BASE / "dashboards"

# Vig / unit math - MUST match mas_effectiveness.py
VIG_STAKE   = 110     # stake 110 to win 100 at -110
WIN_PAYOUT  = 100
BREAKEVEN   = 0.5238  # break-even win rate at -110

# Tier cutoffs - MUST match mas_effectiveness.py
TIER_HIGH   = 0.65
TIER_MED    = 0.58
TIER_LOW    = 0.52

# Regular-season scope. Dashboard excludes playoff picks by default so the
# public stats stay apples-to-apples. BUMP THIS ANNUALLY.
# NHL 2025-26 regular season final day. Override via --reg-season-end YYYY-MM-DD
# or bypass entirely with --include-playoffs.
REGULAR_SEASON_END = "2026-04-17"

# Brand palette - clean minimalist, white bg, black text, subtle green
WHITE       = "#FFFFFF"
BLACK       = "#0A0A0A"
GREEN       = "#16A34A"   # positive / above breakeven
GRAY        = "#64748B"   # neutral / breakeven
RED         = "#DC2626"   # below 50%
MUTED       = "#94A3B8"   # captions, subheads
CARD_EDGE   = "#E2E8F0"
STRIP_BG    = "#F8FAFC"

# Canvas sizes
SIZES = {
    "youtube": (1200, 900),    # YT community post (4:3)
    "square":  (1080, 1080),   # Reddit / IG
    "wide":    (1920, 1080),   # YT end screen / channel banner
}


# -- Shared metric helpers (keep in sync with mas_effectiveness.py) ------------
def tier_label(conf: float) -> str:
    if conf >= TIER_HIGH: return "HIGH"
    if conf >= TIER_MED:  return "MED"
    if conf >= TIER_LOW:  return "LOW"
    return "SKIP"


def roi_pct(wins: int, losses: int):
    """ROI% at -110 vig. Returns None if no settled picks."""
    total = wins + losses
    if total == 0:
        return None
    net = wins * WIN_PAYOUT - losses * VIG_STAKE
    return net / (total * VIG_STAKE) * 100


def units(wins: int, losses: int) -> float:
    """Net units where 1 unit = $100 win target (matches WIN_PAYOUT=100)."""
    return (wins * WIN_PAYOUT - losses * VIG_STAKE) / WIN_PAYOUT


def win_color(win_pct):
    if win_pct is None:
        return GRAY
    if win_pct >= BREAKEVEN: return GREEN
    if win_pct >= 0.50:      return GRAY
    return RED


def fit_text(ax, x, y, text, base_pt, max_width_px, weight="normal",
             color=BLACK, ha="center", va="center", min_pt=10):
    """Render text at base_pt, measure, shrink if wider than max_width_px."""
    t = ax.text(x, y, text, fontsize=base_pt, weight=weight,
                color=color, ha=ha, va=va)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox_px = t.get_window_extent(renderer=renderer)
    if bbox_px.width > max_width_px and bbox_px.width > 0:
        new_pt = max(min_pt, base_pt * max_width_px / bbox_px.width * 0.97)
        t.set_fontsize(new_pt)
    return t


# -- Data loading --------------------------------------------------------------
def load_data(log_path: Path, reg_season_end: str = None, include_playoffs: bool = False):
    """Load the pick log. By default filters to regular-season picks only
    (pred_date <= reg_season_end). Pass include_playoffs=True to keep all rows.
    """
    df = pd.read_csv(log_path)
    df["confidence"] = df["confidence"].astype(float)
    df["edge"]       = df["edge"].astype(float)
    df["tier"]       = df["confidence"].apply(tier_label)

    if not include_playoffs and reg_season_end:
        df = df[df["pred_date"] <= reg_season_end].copy()

    resolved = df[df["result"].isin(["W", "L"])].copy()
    resolved["win"] = (resolved["result"] == "W").astype(int)

    pending = df[~df["result"].isin(["W", "L"])]
    return df, resolved, pending


def compute_metrics(resolved: pd.DataFrame, pending_count: int, days_window: int,
                    today_override: date = None, reg_season_end: str = None) -> dict:
    # Effective "today" for the recent-window. Use the earliest of:
    #   - calendar today (or override)
    #   - regular-season end date (if set; keeps the window in-season)
    #   - latest resolved pred_date (avoids an empty window on sparse days
    #     or once the pipeline stops logging reg-season picks)
    # This is Jeff's "last 7 days of the hockey season" intent.
    today_ref = today_override or date.today()
    from datetime import datetime as _dt
    if reg_season_end:
        rse = _dt.strptime(reg_season_end, "%Y-%m-%d").date()
        if today_ref > rse:
            today_ref = rse
    if len(resolved) > 0:
        latest_resolved = _dt.strptime(resolved["pred_date"].max(), "%Y-%m-%d").date()
        if today_ref > latest_resolved:
            today_ref = latest_resolved
    # Overall (season)
    ow = int(resolved["win"].sum())
    ol = len(resolved) - ow
    overall = dict(
        wins=ow, losses=ol, n=len(resolved),
        win_pct=(ow / len(resolved)) if len(resolved) else 0.0,
        roi=roi_pct(ow, ol),
        units=units(ow, ol),
    )

    # HIGH-tier
    high = resolved[resolved["tier"] == "HIGH"]
    hw = int(high["win"].sum()); hl = len(high) - hw
    tier_high = dict(
        wins=hw, losses=hl, n=len(high),
        win_pct=(hw / len(high)) if len(high) else 0.0,
        roi=roi_pct(hw, hl),
        units=units(hw, hl),
    )

    # Recent window (days-based, using today like mas_effectiveness.py does)
    cutoff = str(today_ref - timedelta(days=days_window))
    recent = resolved[resolved["pred_date"] >= cutoff]
    rw = int(recent["win"].sum()); rl = len(recent) - rw
    recent_m = dict(
        wins=rw, losses=rl, n=len(recent),
        win_pct=(rw / len(recent)) if len(recent) else 0.0,
        roi=roi_pct(rw, rl),
        units=units(rw, rl),
        days=days_window,
    )

    # Calibration over playable (non-SKIP) resolved picks - same cohort as the
    # CALIBRATION section in mas_effectiveness.py
    playable = resolved[resolved["tier"] != "SKIP"]
    if len(playable) > 0:
        pred   = float(playable["confidence"].mean())
        actual = float(playable["win"].mean())
    else:
        pred = actual = 0.0
    calibration = dict(predicted=pred, actual=actual, delta=actual - pred, n=len(playable))

    return dict(
        overall=overall,
        tier_high=tier_high,
        recent=recent_m,
        calibration=calibration,
        total_resolved=len(resolved),
        pending=pending_count,
        date_range=(resolved["pred_date"].min(), resolved["pred_date"].max()),
    )


# -- Logo handling: auto-key dark background, tint grays to black -------------
def load_logo_rgba(path: Path, dark_thresh: int = 60, gray_sat_thresh: int = 35):
    """Load logo PNG and prep it for white backgrounds.

    1) Key near-black background pixels to fully transparent.
    2) Apply a soft alpha ramp through the transition band so edges aren't hard.
    3) Tint surviving grayscale pixels (silver wordmark) to black so the
       'Smart Thinking' text reads on a white dashboard.
    4) Preserve saturated (colored) pixels so the cyan/blue brain mark keeps
       its brand identity.
    """
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].astype(np.int16)
    lum = rgb.max(axis=2)

    # 1) Fully transparent background
    dark_mask = lum < dark_thresh
    arr[dark_mask, 3] = 0

    # 2) Soft transition ramp
    ramp_mask = (lum >= dark_thresh) & (lum < dark_thresh + 40)
    scale = ((lum[ramp_mask] - dark_thresh) / 40 * 255).astype(np.uint8)
    arr[ramp_mask, 3] = np.minimum(arr[ramp_mask, 3], scale)

    # 3/4) Tint near-grayscale survivors to black, keep saturated pixels
    sat = rgb.max(axis=2) - rgb.min(axis=2)   # cheap saturation proxy
    survivor = arr[:, :, 3] > 0
    gray_survivor = survivor & (sat < gray_sat_thresh)
    # Map grayscale RGB->black (so silver text becomes black ink on white bg)
    arr[gray_survivor, 0] = 0
    arr[gray_survivor, 1] = 0
    arr[gray_survivor, 2] = 0

    return Image.fromarray(arr)


# -- Drawing -------------------------------------------------------------------
def draw_dashboard(metrics: dict, size_key: str, out_path: Path,
                   today_ref: date = None) -> Path:
    w_px, h_px = SIZES[size_key]
    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_px)
    ax.set_ylim(0, h_px)
    ax.axis("off")

    # Font scale - uses the smaller dimension so narrow canvases (square)
    # don't blow up fonts beyond what card width can accommodate.
    s = min(w_px, h_px) / 900.0

    # Layout in pixels
    pad          = int(w_px * 0.035)
    header_h     = int(h_px * 0.14)
    bottom_h     = int(h_px * 0.05)
    footer_h     = int(h_px * 0.13)
    footer_pad   = int(h_px * 0.02)

    # Card band
    card_top    = h_px - header_h
    card_bottom = bottom_h + footer_h + footer_pad
    card_h      = card_top - card_bottom - pad
    card_gap    = int(w_px * 0.02)
    card_w      = (w_px - 2 * pad - 2 * card_gap) // 3

    # -- Header: logo + title row
    draw_header(ax, w_px, h_px, pad, header_h, s, today_ref)

    # -- Three cards
    cards = [
        ("SEASON",            metrics["overall"],   None),
        ("HIGH CONFIDENCE",   metrics["tier_high"], "Top tier only"),
        (f"LAST {metrics['recent']['days']} DAYS", metrics["recent"], "Rolling window"),
    ]
    for i, (title, m, sub) in enumerate(cards):
        x = pad + i * (card_w + card_gap)
        y = card_bottom
        draw_card(ax, x, y, card_w, card_h, title, m, sub, s)

    # -- Calibration strip
    draw_calibration(ax, pad, bottom_h, w_px - 2 * pad, footer_h,
                     metrics["calibration"], s)

    # -- Bottom signature line
    draw_bottom(ax, w_px, bottom_h, metrics, s)

    fig.savefig(out_path, dpi=dpi, facecolor=WHITE)
    plt.close(fig)
    return out_path


def draw_header(ax, w_px, h_px, pad, header_h, s, today_ref=None):
    # Logo top-left, or text fallback
    logo = load_logo_rgba(LOGO_FILE)
    logo_box_h = int(header_h * 0.75)
    logo_box_top = h_px - pad
    logo_box_bottom = logo_box_top - logo_box_h

    if logo is not None:
        aspect = logo.width / logo.height
        # Cap width so a very wide logo doesn't crowd the title
        max_logo_w = int(w_px * 0.28)
        logo_w = min(int(logo_box_h * aspect), max_logo_w)
        logo_h = int(logo_w / aspect)
        logo_top = logo_box_top
        logo_bottom = logo_top - logo_h
        logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
        ax.imshow(np.array(logo_resized),
                  extent=[pad, pad + logo_w, logo_bottom, logo_top],
                  zorder=5, interpolation="lanczos")
    else:
        ax.text(pad, (logo_box_top + logo_box_bottom) / 2,
                "SMART THINKING",
                fontsize=28 * s, weight="bold", color=BLACK,
                va="center", ha="left")

    # Title top-right
    today_str = (today_ref or date.today()).strftime("%B %d, %Y")
    title_y = logo_box_top - logo_box_h * 0.30
    sub_y   = logo_box_top - logo_box_h * 0.80
    ax.text(w_px - pad, title_y, "NHL MODEL PERFORMANCE",
            fontsize=20 * s, weight="bold", color=BLACK, va="center", ha="right")
    ax.text(w_px - pad, sub_y, f"Smart Thinking Media  \u00b7  {today_str}",
            fontsize=13 * s, color=MUTED, va="center", ha="right")


def draw_card(ax, x, y, w, h, title, m, sub, s):
    # Card container
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=14",
        linewidth=1.5, edgecolor=CARD_EDGE, facecolor=WHITE, zorder=1,
    )
    ax.add_patch(card)

    # Title
    ax.text(x + w / 2, y + h - int(22 * s), title,
            fontsize=15 * s, weight="bold", color=MUTED,
            ha="center", va="top")

    rec_color = win_color(m["win_pct"])
    has_data  = (m["wins"] + m["losses"]) > 0

    if not has_data:
        ax.text(x + w / 2, y + h * 0.55, "\u2014",
                fontsize=64 * s, weight="bold", color=GRAY,
                ha="center", va="center")
        ax.text(x + w / 2, y + h * 0.30, "No picks yet",
                fontsize=14 * s, color=MUTED, ha="center", va="center")
        if sub:
            ax.text(x + w / 2, y + int(18 * s), sub,
                    fontsize=11 * s, color=MUTED, ha="center",
                    va="bottom", style="italic")
        return

    # Big W-L - dynamically sized so the longest record still fits the card.
    rec = f"{m['wins']}-{m['losses']}"
    fit_text(ax, x + w / 2, y + h * 0.62, rec,
             base_pt=64 * s, max_width_px=w * 0.82,
             weight="bold", color=BLACK, ha="center", va="center")

    # Win %
    ax.text(x + w / 2, y + h * 0.36, f"{m['win_pct']:.1%}",
            fontsize=30 * s, weight="bold", color=rec_color,
            ha="center", va="center")

    # Units + ROI
    u   = m["units"]
    r   = m["roi"] if m["roi"] is not None else 0.0
    u_s = f"{'+' if u >= 0 else ''}{u:.1f}u"
    r_s = f"{'+' if r >= 0 else ''}{r:.1f}% ROI"
    ax.text(x + w / 2, y + h * 0.21, f"{u_s}   \u00b7   {r_s}",
            fontsize=13 * s, weight="bold", color=rec_color,
            ha="center", va="center")

    if sub:
        ax.text(x + w / 2, y + int(18 * s), sub,
                fontsize=11 * s, color=MUTED,
                ha="center", va="bottom", style="italic")


def draw_calibration(ax, x, y, w, h, cal, s):
    # Background strip
    strip = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=10",
        linewidth=0, facecolor=STRIP_BG, zorder=0,
    )
    ax.add_patch(strip)

    pred   = cal["predicted"]
    actual = cal["actual"]
    delta  = cal["delta"]
    abs_d  = abs(delta)
    n      = cal["n"]

    if n == 0:
        trust_text, trust_color = "CALIBRATION UNAVAILABLE", GRAY
    elif abs_d < 0.03:
        trust_text, trust_color = "EXCELLENT CALIBRATION", GREEN
    elif abs_d < 0.05:
        trust_text, trust_color = "STRONG CALIBRATION", GREEN
    elif abs_d < 0.08:
        trust_text, trust_color = "REASONABLE CALIBRATION", GRAY
    else:
        trust_text, trust_color = "CALIBRATION NEEDS TUNING", RED

    ax.text(x + w / 2, y + h * 0.78, "MODEL CALIBRATION",
            fontsize=11 * s, weight="bold", color=MUTED,
            ha="center", va="center")

    if n == 0:
        ax.text(x + w / 2, y + h * 0.42, "Awaiting playable picks",
                fontsize=18 * s, weight="bold", color=BLACK,
                ha="center", va="center")
    else:
        line = (f"Predicted {pred:.1%}     \u00b7     Actual {actual:.1%}"
                f"     \u00b7     Delta {delta:+.1%}")
        fit_text(ax, x + w / 2, y + h * 0.42, line,
                 base_pt=20 * s, max_width_px=w * 0.92,
                 weight="bold", color=BLACK, ha="center", va="center")

    ax.text(x + w / 2, y + h * 0.14, trust_text,
            fontsize=12 * s, weight="bold", color=trust_color,
            ha="center", va="center")


def draw_bottom(ax, w_px, h, metrics, s):
    y = h * 0.50
    left  = (f"{metrics['total_resolved']} picks tracked"
             f"   \u00b7   {metrics['pending']} pending")
    right = "Smart Thinking Media   \u00b7   Entertainment purposes only"
    ax.text(w_px * 0.035, y, left,
            fontsize=11 * s, color=MUTED, ha="left", va="center")
    ax.text(w_px * 0.965, y, right,
            fontsize=11 * s, color=MUTED, ha="right", va="center")


# -- Main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Render NHL MAS performance dashboards.")
    ap.add_argument("--size", choices=list(SIZES.keys()),
                    help="Generate only one size (default: all 3)")
    ap.add_argument("--days", type=int, default=7,
                    help="Recent window for Card 3 (default: 7)")
    ap.add_argument("--today", type=str, default=None,
                    help="Override today's date (YYYY-MM-DD) -- useful for backfill/testing")
    ap.add_argument("--reg-season-end", type=str, default=REGULAR_SEASON_END,
                    help=f"Last day of the NHL regular season (YYYY-MM-DD). "
                         f"Default: {REGULAR_SEASON_END}")
    ap.add_argument("--include-playoffs", action="store_true",
                    help="Include playoff picks instead of regular-season only")
    args = ap.parse_args()

    today_ref = None
    if args.today:
        from datetime import datetime as _dt
        today_ref = _dt.strptime(args.today, "%Y-%m-%d").date()

    if not LOG_FILE.exists():
        print(f"ERROR: prediction_log.csv not found at {LOG_FILE}")
        sys.exit(1)

    _, resolved, pending = load_data(
        LOG_FILE,
        reg_season_end=args.reg_season_end,
        include_playoffs=args.include_playoffs,
    )
    if len(resolved) == 0:
        print("No resolved picks yet -- dashboard skipped.")
        sys.exit(0)

    metrics = compute_metrics(
        resolved, len(pending), args.days,
        today_override=today_ref,
        reg_season_end=None if args.include_playoffs else args.reg_season_end,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = (today_ref or date.today()).isoformat()
    sizes_to_run = [args.size] if args.size else list(SIZES.keys())

    scope = "regular season only" if not args.include_playoffs else "including playoffs"
    print(f"\n  Generating dashboards for {today}")
    print(f"  Scope: {scope}   (reg-season-end: {args.reg_season_end})")
    print(f"  Data: {metrics['total_resolved']} resolved  .  {metrics['pending']} pending")
    print(f"  Window: {metrics['date_range'][0]} -> {metrics['date_range'][1]}\n")

    out_paths = []
    for size_key in sizes_to_run:
        fname = f"dashboard_{today}_{size_key}.png"
        out = OUTPUT_DIR / fname
        draw_dashboard(metrics, size_key, out, today_ref=today_ref)
        out_paths.append(out)
        print(f"  OK {size_key:<8}  {out}")

    print(f"\n  Done. {len(out_paths)} file(s) in {OUTPUT_DIR}\n")
    return out_paths


if __name__ == "__main__":
    main()
