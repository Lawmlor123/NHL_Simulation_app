"""
daily_update.py
Quick daily refresh: pull latest box scores, rebuild features, predict.
"""
import subprocess
import sys
from datetime import date

scripts = [
    ("Pulling skater box scores", "collect_and_flatten.py"),
    ("Building skater features", "build_features.py"),
    ("Pulling goalie box scores", "pull_goalies.py"),
    ("Building goalie features", "build_goalie_features.py"),
]

print(f"{'='*60}")
print(f"  NHL Daily Update — {date.today()}")
print(f"{'='*60}\n")

for desc, script in scripts:
    print(f"\n{'─'*40}")
    print(f"  {desc} ...")
    print(f"{'─'*40}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n  ❌ ERROR in {script}! Stopping.")
        sys.exit(1)

# Now predict
print(f"\n{'─'*40}")
print(f"  Making predictions ...")
print(f"{'─'*40}")
subprocess.run([sys.executable, "predict_today.py"])

print(f"\n{'='*60}")
print(f"  DONE — {date.today()}")
print(f"{'='*60}")