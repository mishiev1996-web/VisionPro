"""Run full collect_all with reduced settings to verify pipeline."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import data_collector
import db

def log(e):
    msg = e.get("msg", "")
    t = e.get("type", "")
    if msg:
        prefix = "✓" if t == "success" else "✗" if t == "error" else "·"
        print(f"  {prefix} {msg}")

print("=== Starting collect_all ===")
result = data_collector.collect_all(
    progress_cb=log,
)

print("\n=== SUMMARY ===")
for source, data in result.items():
    if isinstance(data, dict):
        err = data.get("errors", data.get("error", 0))
        saved = data.get("saved", data.get("matched", data.get("teams", data.get("matches", "?"))))
        print(f"  {source:16s} saved={saved} errors={err}")
    else:
        print(f"  {source:16s} {str(data)[:80]}")

print("\n=== DB state ===")
with db.connect() as conn:
    leagues = conn.execute(
        "SELECT league_slug, COUNT(*) as n FROM matches GROUP BY league_slug ORDER BY n DESC"
    ).fetchall()
    for r in leagues:
        print(f"  {r[0]:20s} {r[1]:5d} matches")

    elo = conn.execute("SELECT COUNT(*) FROM team_elo").fetchone()[0]
    odds = conn.execute("SELECT COUNT(*) FROM match_odds").fetchone()[0]
    injuries = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
    weather = conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    print(f"\n  Elo: {elo}  Odds: {odds}  Injuries: {injuries}  Weather: {weather}")
