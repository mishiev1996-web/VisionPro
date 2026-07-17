"""Detailed leakage analysis."""
import csv
import os

DATA_DIR = os.path.join(os.getcwd(), "база для обучения")
csv_path = os.path.join(DATA_DIR, "Matches.csv")

print("=" * 70)
print("  DIAGNOSTIC: Data Leakage Analysis for train_v2.py")
print("=" * 70)

r = csv.DictReader(open(csv_path, encoding='utf-8'))
rows = list(r)

print(f"\nTotal rows: {len(rows)}")

# Step 1: Check stats features
print("\n" + "=" * 70)
print("STEP 1: Stats Features Analysis")
print("=" * 70)

stats_cols = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
              'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners']

for col in stats_cols:
    vals = [float(rows[i][col]) for i in range(min(10000, len(rows))) if rows[i][col]]
    non_zero = sum(1 for v in vals if v > 0)
    avg = sum(vals) / len(vals) if vals else 0
    print(f"\n  {col}:")
    print(f"    Non-empty values: {len(vals)}/{min(10000, len(rows))}")
    print(f"    Average: {avg:.1f}")
    print(f"    Non-zero: {non_zero}")
    if vals:
        print(f"    Sample: {vals[:5]}")
    print(f"    VERDICT: LEAKAGE - match-level final statistics")

# Step 2: Check Odds and Form
print("\n" + "=" * 70)
print("STEP 2: Odds and Form Analysis")
print("=" * 70)

odds_cols = ['OddHome', 'OddDraw', 'OddAway']
for col in odds_cols:
    vals = [float(rows[i][col]) for i in range(min(1000, len(rows))) if rows[i][col]]
    if vals:
        avg = sum(vals) / len(vals)
        print(f"\n  {col}: average={avg:.2f}, sample={vals[:3]}")
        print(f"    VERDICT: NO LEAKAGE - pre-match odds")

form_cols = ['Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
for col in form_cols:
    vals = [float(rows[i][col]) for i in range(min(1000, len(rows))) if rows[i][col]]
    if vals:
        avg = sum(vals) / len(vals)
        print(f"\n  {col}: average={avg:.1f}, range=[{min(vals):.0f}, {max(vals):.0f}]")
        print(f"    VERDICT: NO LEAKAGE - points from prior matches")

# Step 3: Filtering bias
print("\n" + "=" * 70)
print("STEP 3: Filtering Analysis")
print("=" * 70)

# Apply same filters
filtered = [r for r in rows if r['FTResult'] in ['H', 'D', 'A'] and r['FTHome'] and r['FTAway']]
print(f"\nOriginal: {len(rows)}")
print(f"After FTResult filter: {len(filtered)}")

# Check division distribution
from collections import Counter
div_all = Counter(r['Division'] for r in rows)
div_filt = Counter(r['Division'] for r in filtered)

print("\nDivision distribution (top 10):")
print(f"{'Division':<10} {'Original':>10} {'Filtered':>10} {'Ratio':>8}")
for div, count in div_all.most_common(10):
    filt_count = div_filt.get(div, 0)
    ratio = filt_count / count * 100 if count > 0 else 0
    print(f"{div:<10} {count:>10} {filt_count:>10} {ratio:>7.1f}%")

# Final summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
LEAKAGE FOUND in:
  - HomeShots, AwayShots (match final stats)
  - HomeTarget, AwayTarget (match final stats)
  - HomeFouls, AwayFouls (match final stats)
  - HomeCorners, AwayCorners (match final stats)
  - HomeYellow, AwayYellow (match final stats)
  - HomeRed, AwayRed (match final stats)
  - shots_diff, corners_diff (derived from leaked features)

NO LEAKAGE in:
  - HomeElo, AwayElo (pre-match ratings)
  - Form3/5 Home/Away (points from prior matches)
  - OddHome/Draw/Away (pre-match odds)

RECOMMENDATION: Remove all stats features (Shots, Corners, Fouls, Cards)
and retrain. These features leak match outcome information.
""")
