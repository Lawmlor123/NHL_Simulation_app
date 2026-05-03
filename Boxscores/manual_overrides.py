# manual_overrides.py — 2026-04-13
# Update this file daily before predict_today.py runs
# Status key: ✅ confirmed | ⚠ expected/likely | ❓ unconfirmed
#
# ⚠️  STALE FILE GUARD: If today's date doesn't match the date above,
#     this file is from a previous day — update before rerunning!
#
# CHANGES FROM 2026-04-13:
#   Cleared April 7 overrides — multi-source scraper now handles conflicts automatically
#   Add entries here ONLY for goalies the scraper still gets wrong after running

GOALIE_OVERRIDES = {
    # ── April 13 games ────────────────────────────────────────────
    # Example: "NSH": "Juuse Saros",   # ✅ if scraper still shows Annunen
}
