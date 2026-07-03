"""Test collect_elo on 10 teams only."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import db
from data_collector import collect_elo

class Counter:
    def __init__(self):
        self.count = 0
    def __call__(self, e):
        self.count += 1
        if self.count <= 5 or "success" in str(e.get("type","")):
            print("  ", e.get("msg", ""))

result = collect_elo(progress_cb=Counter())
print("RESULT:", result)
