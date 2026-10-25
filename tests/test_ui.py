from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lookroom import db, seed

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_and_like_updates_feedback(isolated_data):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert "Your wardrobe. Fewer decisions." in _md(at)
    uid = at.query_params["u"][0] if isinstance(at.query_params["u"], list) else at.query_params["u"]
    assert uid and uid != db.DEFAULT_USER and db.count_items(uid) == len(seed.load_demo_items())  # private workspace provisioned
    likes = [b for b in at.button if b.key and b.key.startswith("like-")]
    assert len(likes) == 3
    likes[0].click().run()
    assert not at.exception
    assert len(db.list_feedback(user_id=uid)) == 1 and db.list_feedback() == []  # isolated from other users
    assert "last_feedback" in at.session_state


@pytest.mark.parametrize("lang,headline", [("hy", "Ձեր զգեստապահարանը"), ("ru", "Ваш гардероб")])
def test_app_languages(isolated_data, lang, headline):
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["lang"] = lang
    at.run()
    assert not at.exception
    assert headline in _md(at)


def test_app_recovers_from_stale_modules_after_git_pull(isolated_data, monkeypatch):
    """Streamlit Cloud keeps old modules in memory after a push; the app must re-import them, not crash."""
    import sys
    import types

    import lookroom.models as real
    old = types.ModuleType("lookroom.models")  # the pre-pull models.py: no DISPLAY_ORDER yet
    old.__file__, old._wil_mtime = real.__file__, 0.0
    saved = {n: m for n, m in sys.modules.items() if n == "lookroom" or n.startswith("lookroom.")}
    monkeypatch.setitem(sys.modules, "lookroom.models", old)
    try:
        at = AppTest.from_file(APP, default_timeout=90).run()
        assert not at.exception
        assert sys.modules["lookroom.models"] is not old and hasattr(sys.modules["lookroom.models"], "DISPLAY_ORDER")
    finally:
        sys.modules.update(saved)
