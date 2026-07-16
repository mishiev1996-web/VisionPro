"""Test prematch analysis with a real game_id."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests

BASE = "http://127.0.0.1:8000"

# Step 1: Get today's prematch matches
r = requests.get(f"{BASE}/api/prematch/today")
data = r.json()
matches = data.get("matches", [])
print(f"Prematch matches: {len(matches)}")
for m in matches[:10]:
    gid = m.get("game_id")
    h = m.get("home", "?")
    a = m.get("away", "?")
    lg = m.get("league", "?")
    print(f"  game_id={gid}, {h} vs {a} ({lg})")

if not matches:
    print("No matches found!")
    sys.exit(1)

# Pick the first match
test_match = matches[0]
game_id = test_match["game_id"]
home = test_match["home"]
away = test_match["away"]
print(f"\n=== Testing AI analysis for game_id={game_id}: {home} vs {away} ===")

# Step 2: Start prematch analysis
t0 = time.time()
r = requests.post(f"{BASE}/api/prematch/{game_id}/analyze")
print(f"POST /api/prematch/{game_id}/analyze: {r.status_code} {r.json()}")
job_data = r.json()

if not job_data.get("ok"):
    print("Failed to start analysis!")
    sys.exit(1)

# Step 3: Poll for result via collect/status
print("Waiting for result...")
for i in range(60):
    time.sleep(2)
    elapsed = time.time() - t0
    r = requests.get(f"{BASE}/api/collect/status")
    status = r.json()
    if status.get("result"):
        print(f"\n=== RESULT after {elapsed:.1f}s ===")
        result = status["result"]
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000])
        break
    if status.get("error"):
        print(f"\n=== ERROR after {elapsed:.1f}s ===")
        print(status["error"])
        break
    if i % 5 == 0:
        print(f"  [{elapsed:.0f}s] still running...")
else:
    print(f"\n=== TIMEOUT after {time.time()-t0:.1f}s ===")
