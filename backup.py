"""
backup.py — Create a snapshot of the current project state.

Run: python backup.py
Creates: backup/snapshot_YYYY-MM-DD_HH-MM.zip

Includes:
  - All .py files (core logic)
  - requirements.txt
  - best_params.json
  - config files

Excludes:
  - data/football.db (too large, rebuild with data_collector.py)
  - __pycache__/
  - .git/
  - frontend/ (you'll redo this)
  - model.pkl (rebuild with train.py)
"""
import os
import zipfile
import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
BACKUP_DIR = PROJECT_DIR / "backup"
SNAPSHOT_NAME = f"snapshot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.zip"

INCLUDE_EXTENSIONS = {".py", ".txt", ".json", ".bat", ".md"}
INCLUDE_FILES = {"requirements.txt", "best_params.json", ".gitignore"}

EXCLUDE_DIRS = {"__pycache__", ".git", ".mimocode", "backup", "data", "frontend"}
EXCLUDE_FILES = {"model.pkl", "football.db", "football_export.json"}


def should_include(path: Path) -> bool:
    rel = path.relative_to(PROJECT_DIR)
    parts = rel.parts

    # Skip excluded directories
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False

    # Skip excluded files
    if path.name in EXCLUDE_FILES:
        return False

    # Include by extension or exact name
    if path.name in INCLUDE_FILES:
        return True
    if path.suffix in INCLUDE_EXTENSIONS:
        return True

    return False


def create_snapshot():
    BACKUP_DIR.mkdir(exist_ok=True)
    snapshot_path = BACKUP_DIR / SNAPSHOT_NAME

    files_included = 0
    with zipfile.ZipFile(snapshot_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Skip excluded dirs in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for fname in sorted(files):
                fpath = Path(root) / fname
                if should_include(fpath):
                    arcname = fpath.relative_to(PROJECT_DIR)
                    zf.write(fpath, arcname)
                    files_included += 1

    size_kb = snapshot_path.stat().st_size / 1024
    print(f"Snapshot created: {snapshot_path}")
    print(f"  Files: {files_included}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"\nTo restore: unzip {snapshot_path.name} -d <target_dir>")
    print(f"To see contents: python -c \"import zipfile; [print(z) for z in zipfile.ZipFile('{snapshot_path}').namelist()]\"")


if __name__ == "__main__":
    create_snapshot()
