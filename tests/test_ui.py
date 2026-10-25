from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lookroom import db, seed, service

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _qp(at, key):
    v = at.query_params.get(key)
    return v[0] if isinstance(v, list) else v


def _guest(uid: str = "tstguest01", **params) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["u"] = uid
    for k, v in params.items():
        at.query_params[k] = v
    return at.run()


def test_first_visit_shows_sign_in_then_guest_mode(isolated_data):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert any(b.key == "si_go" for b in at.button) and any(b.key == "su_go" for b in at.button)
    assert not [b for b in at.button if b.key and b.key.startswith("like-")]  # nothing private before choosing
    at.button(key="guest").click().run()
    assert not at.exception and _qp(at, "u")
    assert [b for b in at.button if b.key and b.key.startswith("like-")]


def test_sign_up_sign_out_sign_in_keeps_wardrobe(isolated_data):
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.text_input(key="su_user").input("Anna.Test")
    at.text_input(key="su_pw").input("secret123")
    at.button(key="su_go").click().run()
    assert not at.exception
    token = _qp(at, "s")
    assert token and not _qp(at, "u")
    uid = service.session_user(token)
    assert uid and service.account_name(uid) == "anna.test"
    new = service.add_item("My Red Scarf", "outerwear", "red", "casual", 3, 2, user_id=uid)
    at.segmented_control(key="nav").set_value("learn").run()
    at.toggle(key="tech").set_value(True).run()
    at.button(key="signout").click().run()
    assert not _qp(at, "s") and any(b.key == "si_go" for b in at.button)
    assert service.session_user(token) is None  # session revoked server-side
    at.text_input(key="si_user").input("anna.test")
    at.text_input(key="si_pw").input("wrong-password")
    at.button(key="si_go").click().run()
    assert any("Wrong username or password" in e.value for e in at.error)
    at.text_input(key="si_pw").input("secret123")
    at.button(key="si_go").click().run()
    assert not at.exception
    assert service.session_user(_qp(at, "s")) == uid
    assert new.id in {i.id for i in db.list_items(uid)}  # the account's own item is still there
    assert at.segmented_control(key="nav").value == "today" and not at.toggle(key="tech").value  # fresh start
    assert [b for b in at.button if b.key and b.key.startswith("like-hero-")]  # Today's page is what renders


def test_duplicate_username_is_rejected(isolated_data):
    service.register("taken", "secret123")
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.text_input(key="su_user").input("TAKEN")
    at.text_input(key="su_pw").input("secret123")
    at.button(key="su_go").click().run()
    assert any("already taken" in e.value for e in at.error) and not _qp(at, "s")


def test_dislike_brings_a_clearly_different_look_each_time(isolated_data):
    at = _guest()
    seen = []
    for _ in range(4):
        hero = next(b for b in at.button if b.key and b.key.startswith("dislike-hero-"))
        key = tuple(hero.key.removeprefix("dislike-hero-").split("|"))
        seen.append(key)
        hero.click().run()
        assert not at.exception
    new_hero = next(b for b in at.button if b.key and b.key.startswith("dislike-hero-"))
    new_key = tuple(new_hero.key.removeprefix("dislike-hero-").split("|"))
    assert new_key not in seen
    assert len(set(new_key) & set(seen[-1])) <= max(len(new_key), len(seen[-1])) - 2  # 2+ pieces changed


def test_saved_looks_and_technical_details_toggle(isolated_data):
    at = _guest()
    like = next(b for b in at.button if b.key and b.key.startswith("like-hero-"))
    liked = sorted(like.key.removeprefix("like-hero-").split("|"))
    like.click().run()
    assert [sorted(i.id for i in look) for look in service.saved_looks("tstguest01")] == [liked]
    at.segmented_control(key="nav").set_value("saved").run()
    assert not at.exception and "Saved looks: 1" in " ".join(c.value for c in at.caption)
    page = _md(at) + " ".join(c.value for c in at.caption)
    assert "Qwen" not in page and "FashionSigLIP" not in page  # no model names for normal users
    at.toggle(key="tech").set_value(True).run()
    assert "FashionSigLIP" in _md(at)  # interview mode shows what runs under the hood


def test_app_renders_and_like_updates_feedback(isolated_data):
    at = _guest("tstguest02")
    assert not at.exception
    assert "Your wardrobe. Fewer decisions." in _md(at)
    uid = _qp(at, "u")
    assert uid and uid != db.DEFAULT_USER and db.count_items(uid) == len(seed.load_demo_items())  # private workspace provisioned
    likes = [b for b in at.button if b.key and b.key.startswith("like-")]
    assert len(likes) == 3
    likes[0].click().run()
    assert not at.exception
    assert len(db.list_feedback(user_id=uid)) == 1 and db.list_feedback() == []  # isolated from other users
    assert "last_feedback" in at.session_state


@pytest.mark.parametrize("lang,headline", [("hy", "Ձեր զգեստապահարանը"), ("ru", "Ваш гардероб")])
def test_app_languages(isolated_data, lang, headline):
    at = _guest(lang=lang)
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
        at = _guest()
        assert not at.exception
        assert sys.modules["lookroom.models"] is not old and hasattr(sys.modules["lookroom.models"], "DISPLAY_ORDER")
    finally:
        sys.modules.update(saved)


def test_chat_dislike_answers_with_a_different_look(isolated_data):
    at = _guest("tstguest03")
    at.segmented_control(key="nav").set_value("stylist").run()
    at.button[[b.label for b in at.button].index("Dinner date, 8°C, classic")].click().run()
    assert not at.exception
    first = at.session_state["chat"][-1]["item_ids"]
    at.button(key="chat-dislike-1").click().run()
    assert not at.exception
    reply = at.session_state["chat"][-1]
    assert reply["role"] == "assistant" and reply["item_ids"] and reply["item_ids"] != first
    assert len(set(reply["item_ids"]) & set(first)) <= max(len(first), len(reply["item_ids"])) - 2
