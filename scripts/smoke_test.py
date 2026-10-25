"""End-to-end smoke test on a throwaway data dir. Usage: python scripts/smoke_test.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with tempfile.TemporaryDirectory() as tmp:
    os.environ["LOOKROOM_DATA_DIR"] = tmp
    from lookroom import db, service  # noqa: E402
    from lookroom.models import validate_context  # noqa: E402

    service.ensure_ready()
    items = db.list_items()
    assert len(items) >= 14, items
    print(f"[ok] database initialized at {db.db_path()} with {len(items)} seed items")

    ctx = validate_context("work", 10, "smart")
    res = service.get_recommendations(ctx)
    assert len(res.outfits) == 3
    for i, o in enumerate(res.outfits, 1):
        print(f"[ok] #{i} {o.score:5.1f}  " + " + ".join(it.name for it in o.items))

    fb = service.record_feedback(list(res.outfits[0].key), "like", ctx)
    assert fb["score_after"] > fb["score_before"]
    print(f"[ok] like changed score {fb['score_before']} -> {fb['score_after']}")
    print("SMOKE TEST PASSED")
