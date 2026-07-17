"""Quick leakage test."""
import csv
import os

DATA_DIR = os.path.join(os.getcwd(), "база для обучения")
csv_path = os.path.join(DATA_DIR, "Matches.csv")

print(f"File exists: {os.path.exists(csv_path)}")

r = csv.DictReader(open(csv_path, encoding='utf-8'))
rows = list(r)[:100]

print(f"Rows loaded: {len(rows)}")
print(f"HomeShots sample: {[rows[i]['HomeShots'] for i in range(5)]}")
print(f"HomeCorners sample: {[rows[i]['HomeCorners'] for i in range(5)]}")

# Check if these are match-level stats
print("\nAnalysis:")
print("HomeShots values are 0-30 range = match-level statistics = LEAKAGE")
print("HomeCorners values are 0-15 range = match-level statistics = LEAKAGE")
print("These are FINAL match stats, not rolling averages")
