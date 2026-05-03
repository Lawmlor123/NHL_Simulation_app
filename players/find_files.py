import os
import pandas as pd

# ── Search for parquet files starting from a few likely roots ──
search_roots = [
    os.path.expanduser("~"),                          # home dir
    r"C:\Users\shell\OneDrive\Documents",             # Documents
    r"C:\Users\shell\OneDrive\Documents\Code Projects\NHL & Sports\NHL_Player",  # project root
    os.getcwd(),                                       # current dir
]

targets = ["skater_features.parquet", "goalie_features.parquet"]
found = {}

print("=" * 60)
print("SEARCHING FOR PARQUET FILES...")
print("=" * 60)

for root in search_roots:
    if not os.path.isdir(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for t in targets:
            if t in filenames and t not in found:
                full_path = os.path.join(dirpath, t)
                found[t] = full_path
                print(f"\n✓ Found: {full_path}")
    # Stop early if both found
    if len(found) == len(targets):
        break

# ── Report what we found ──
print("\n" + "=" * 60)
if not found:
    print("✗ No parquet files found. Check that they exist.")
else:
    for name, path in found.items():
        print(f"\n{'=' * 60}")
        print(f"FILE: {path}")
        print(f"{'=' * 60}")
        df = pd.read_parquet(path)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            nulls = df[col].isna().sum()
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else "ALL NULL"
            print(f"  {i:3d}. {col:<35s} {str(dtype):<15s} nulls={nulls:<8d} sample={sample}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        print()

# ── Print paths ready to copy into the real script ──
print("\n" + "=" * 60)
print("COPY THESE PATHS INTO player.py:")
print("=" * 60)
for name, path in found.items():
    varname = "SKATER_PATH" if "skater" in name else "GOALIE_PATH"
    print(f'{varname} = r"{path}"')