import pytest
from fastapi.testclient import TestClient

from api import app, RecommendationsOut


@pytest.fixture
def client(isolated_data):
    with TestClient(app) as c:  # lifespan seeds the temp DB
        yield c


def test_health_and_wardrobe(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["items"] == 14 and isinstance(h["vision_available"], bool)
    items = client.get("/wardrobe").json()
    assert len(items) == 14 and {"id", "category", "warmth"} <= set(items[0])


def test_recommendations_schema(client):
    r = client.get("/recommendations", params={"occasion": "work", "temperature": 8, "style": "classic"})
    assert r.status_code == 200
    body = RecommendationsOut.model_validate(r.json())
    assert len(body.outfits) == 3 and [o.rank for o in body.outfits] == [1, 2, 3]
    assert all(any(i.category == "outerwear" for i in o.items) for o in body.outfits)


@pytest.mark.parametrize("params", [{"occasion": "party"}, {"temperature": 99}, {"style": "goth"}])
def test_recommendations_validation(client, params):
    assert client.get("/recommendations", params=params).status_code == 422


def test_feedback_flow_changes_score(client):
    params = {"occasion": "date", "temperature": 15, "style": "minimal"}
    first = client.get("/recommendations", params=params).json()["outfits"][0]
    r = client.post("/feedback", json={"item_ids": first["item_ids"], "vote": "like", **params})
    assert r.status_code == 201
    assert r.json()["score_after"] > r.json()["score_before"]
    summary = client.get("/feedback-summary").json()
    assert summary["likes"] == 1 and summary["dislikes"] == 0 and len(summary["items"]) == len(first["item_ids"])


def test_vision_tag_endpoint(client):
    from pathlib import Path
    img = (Path(__file__).resolve().parent.parent / "demo_assets" / "sneakers_public_domain.jpg").read_bytes()
    r = client.post("/vision/tag", files={"file": ("x.jpg", img, "image/jpeg")})
    assert r.status_code == 200 and r.json()["color"] == "white"
    bad = client.post("/vision/tag", files={"file": ("x.jpg", b"nope", "image/jpeg")})
    assert bad.status_code == 422


def test_stylist_chat_endpoint_offline(client):
    r = client.post("/stylist/chat", json={"message": "office meeting, 10 degrees", "style": "smart"})
    body = r.json()
    assert r.status_code == 200 and body["source"] == "rules" and body["occasion"] == "work"
    assert body["outfit"]["score"] > 0
    assert client.post("/stylist/chat", json={"message": ""}).status_code == 422


def test_tenant_isolation_and_planning(client):
    a = {"X-User-Id": "alice"}
    params = {"occasion": "work", "temperature": 15, "style": "smart"}
    first = client.get("/recommendations", params=params, headers=a).json()["outfits"][0]
    client.post("/feedback", json={"item_ids": first["item_ids"], "vote": "like", **params}, headers=a)
    assert client.get("/feedback-summary", headers=a).json()["likes"] == 1
    assert client.get("/feedback-summary", headers={"X-User-Id": "bob"}).json()["likes"] == 0
    assert client.get("/feedback-summary", headers={"X-User-Id": "../etc"}).json()["likes"] == 0  # unsafe id -> demo
    week = client.post("/plan/week", json={"days": [{"occasion": "work", "temperature": t} for t in (10, 12, 14)],
                                           "style": "smart", "lang": "ru"}).json()
    assert len(week) == 3 and all(d["score"] > 0 for d in week)
    r = client.get("/recommendations", params={**params, "lang": "hy"}).json()
    assert any("Ա" <= ch <= "֏" for ch in r["outfits"][0]["explanation"])  # Armenian script


def test_feedback_rejects_bad_input(client):
    base = {"occasion": "casual", "temperature": 20, "style": "casual"}
    ok_ids = ["top-black-tee", "bottom-black-jeans", "shoes-white-sneakers"]
    assert client.post("/feedback", json={**base, "item_ids": ok_ids, "vote": "love"}).status_code == 422
    assert client.post("/feedback", json={**base, "item_ids": ok_ids[:2], "vote": "like"}).status_code == 422
    bad = ["ghost-item", *ok_ids[1:]]
    assert client.post("/feedback", json={**base, "item_ids": bad, "vote": "like"}).status_code == 404
