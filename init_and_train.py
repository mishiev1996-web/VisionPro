"""One-time init: create DB, collect data, train model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=== Football AI: Init & Train ===")

# 1. Init DB
print("\n[1/3] Initializing database...")
import db
db.init_db()
print("  DB ready.")

# 2. Collect data (quick: current season only)
print("\n[2/3] Collecting data (current season)...")
try:
    import data_collector
    result = data_collector.collect_all(
        seasons=[2024, 2025],
        progress_cb=lambda e: print(f"  {e.get('msg','')}")
    )
    print(f"  Data collected: {result.get('quality',{}).get('total_matches',0)} matches")
except Exception as e:
    print(f"  Data collection error (continuing): {e}")

# 3. Train model
print("\n[3/3] Training model...")
try:
    import train
    train.main()
    print("  Model trained.")
except Exception as e:
    print(f"  Training error: {e}")

print("\n=== Init complete ===")
