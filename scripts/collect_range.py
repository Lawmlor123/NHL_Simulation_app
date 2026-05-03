#!/usr/bin/env python
from datetime import date, timedelta
from subprocess import run

def collect_range(start: date, end: date) -> None:
    cur = start
    while cur <= end:
        iso = cur.isoformat()
        print(f"Collecting {iso} ...")
        run([
            "python",
            "nhl_daily_collector.py",
            "--date", iso,
            "--include-stats",
            "--save", f"nhl_{iso}.json",
        ], check=True)
        cur += timedelta(days=1)

if __name__ == "__main__":
    collect_range(date(2023, 10, 10), date(2024, 4, 18))   # 2023-24 season
    collect_range(date(2024, 10, 9), date(2025, 4, 15))    # 2024-25 season