"""Restore the known-good demo state. Usage: python scripts/reset_demo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lookroom import db, seed  # noqa: E402

if __name__ == "__main__":
    stats = seed.reset_demo()
    print(f"Reset complete ({db.db_path()}): {stats}")
