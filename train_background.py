#!/usr/bin/env python3
"""Background training script for VisionPro model.
Runs train.py, saves model, and restarts visionpro service.

Usage: python3 /opt/VisionPro/train_background.py
Logs to: /opt/VisionPro/logs/train.log
"""
import sys
import os
import time
import subprocess
import datetime

LOG_DIR = "/opt/VisionPro/logs"
LOG_FILE = os.path.join(LOG_DIR, "train.log")
MODEL_PATH = "/opt/VisionPro/model.pkl"
BACKUP_PATH = "/opt/VisionPro/model.pkl.backup"

os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("TRAINING STARTED")

# Backup current model
if os.path.exists(MODEL_PATH):
    import shutil
    shutil.copy2(MODEL_PATH, BACKUP_PATH)
    log(f"Backed up current model to {BACKUP_PATH}")

# Run training
t0 = time.time()
try:
    result = subprocess.run(
        [sys.executable, "/opt/VisionPro/train.py"],
        cwd="/opt/VisionPro",
        capture_output=True,
        text=True,
        timeout=7200  # 2 hour timeout
    )
    
    elapsed = (time.time() - t0) / 60
    log(f"Training finished in {elapsed:.1f} min")
    
    if result.returncode == 0:
        log("Training SUCCESS")
        # Log last 30 lines of output
        lines = result.stdout.strip().split("\n")
        for line in lines[-30:]:
            log(f"  {line}")
        
        # Restart visionpro to load new model
        log("Restarting visionpro service...")
        r = subprocess.run(["systemctl", "restart", "visionpro"], capture_output=True, text=True)
        if r.returncode == 0:
            log("Service restarted OK")
        else:
            log(f"Restart failed: {r.stderr}")
    else:
        log(f"Training FAILED (exit code {result.returncode})")
        log(f"STDERR: {result.stderr[-1000:]}")
        
except subprocess.TimeoutExpired:
    log("Training TIMED OUT (2h)")
except Exception as e:
    log(f"ERROR: {e}")

log("TRAINING SESSION ENDED")
