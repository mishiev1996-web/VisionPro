"""BSD recon part 3 — ML predictions + historical data check."""
import json
import urllib.request

TOKEN = "25b059faf7f87e66955d0b1f554e40622c7a79ba"
BASE = "https://sports.bzzoiro.com/api"

def api_get(path, params=None):
    url = BASE + path
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{qs}"
    headers = {"Authorization": f"Token {TOKEN}", "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as e:
        return 0, str(e)

def p(obj, max_len=3000):
    s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    print(s[:max_len])
    if len(s) > max_len:
        print("  [... truncated ...]")

# ── 1. ML PREDICTIONS for event 46392 ──
print("=" * 60)
print("1. ML PREDICTIONS (event 46392)")
print("=" * 60)
status, data = api_get("/predictions/", {"event_id": 46392})
if status == 200:
    results = data.get("results", [])
    print(f"Total predictions: {data.get('count', len(results))}")
    # Show first prediction details
    if results:
        pred = results[0]
        print(f"\nPrediction id={pred.get('id')}")
        # Look for probability fields
        for k, v in pred.items():
            if k not in ("event",):
                if isinstance(v, (int, float, str, bool)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict) and len(str(v)) < 500:
                    print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
        # Full prediction object
        print("\nFull prediction object (excluding event):")
        p({k: v for k, v in pred.items() if k != "event"})

# ── 2. HISTORICAL DATA — check if scores are available ──
print("\n" + "=" * 60)
print("2. HISTORICAL DATA (scores)")
print("=" * 60)
for date in ["2026-05-24", "2026-04-13", "2025-12-01", "2025-06-01"]:
    status, data = api_get("/events/", {"date": date})
    events = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(events, list) and events:
        with_score = [e for e in events if e.get("home_score") is not None]
        print(f"  {date}: {len(events)} events, {len(with_score)} with scores")
        if with_score:
            ev = with_score[0]
            print(f"    Example: {ev['home_team']} {ev['home_score']}:{ev['away_score']} {ev['away_team']}")
    else:
        print(f"  {date}: 0 events (status={status})")

# ── 3. COMPARE BSD ODDS vs SSTATS for same match ──
print("\n" + "=" * 60)
print("3. ODDS COMPARISON: BSD vs SSTATS")
print("=" * 60)
# Find an EPL match in BSD
status, data = api_get("/events/", {"date": "2026-07-13", "league_id": 1})
events = data.get("results", data) if isinstance(data, dict) else data
if isinstance(events, list):
    epl = [e for e in events if "Premier" in (e.get("league") or {}).get("name", "")]
    if epl:
        ev = epl[0]
        print(f"BSD: {ev['home_team']} vs {ev['away_team']}")
        print(f"  odds_home={ev.get('odds_home')} draw={ev.get('odds_draw')} away={ev.get('odds_away')}")
    else:
        print("No EPL matches today in BSD")
        # Show any match with odds
        for ev in events[:3]:
            league = (ev.get("league") or {}).get("name", "?")
            print(f"  {league}: {ev['home_team']} vs {ev['away_team']} odds={ev.get('odds_home')}/{ev.get('odds_draw')}/{ev.get('odds_away')}")
else:
    print(f"  Status: {status}")

# ── 4. sr_stats field ──
print("\n" + "=" * 60)
print("4. SR_STATS (SportsRadar stats?)")
print("=" * 60)
status, data = api_get("/events/", {"date": "2026-07-13"})
events = data.get("results", data) if isinstance(data, dict) else data
if isinstance(events, list) and events:
    ev = events[0]
    sr = ev.get("sr_stats")
    if sr:
        print(f"  sr_stats type: {type(sr).__name__}")
        p(sr)
    else:
        print("  sr_stats: None/empty")
    # Check live_stats
    ls = ev.get("live_stats")
    if ls:
        print(f"\n  live_stats type: {type(ls).__name__}")
        p(ls)
    else:
        print("  live_stats: None/empty")
    # Check funfacts
    ff = ev.get("funfacts")
    if ff:
        print(f"\n  funfacts type: {type(ff).__name__}")
        p(ff)
