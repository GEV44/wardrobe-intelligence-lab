import io
import json

from lookroom import weather


class FakeResp(io.BytesIO):
    headers = {"Expires": "Mon, 01 Jan 2035 00:00:00 GMT"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _payload(temp=7.4, symbol="lightrain"):
    return json.dumps({"properties": {"timeseries": [{"data": {
        "instant": {"details": {"air_temperature": temp}},
        "next_1_hours": {"summary": {"symbol_code": symbol}, "details": {"precipitation_amount": 0.6}}}}]}}).encode()


def test_parses_and_caches(monkeypatch):
    monkeypatch.setattr(weather, "_cache", {})
    calls = []

    def fetch(req, timeout):
        calls.append(req)
        assert "WardrobeIntelligenceLab" in req.get_header("User-agent") and timeout <= 5
        assert "lat=51.5072&lon=-0.1276" in req.full_url  # <= 4 decimals per MET terms
        return FakeResp(_payload())

    w = weather.current("London", fetch=fetch)
    assert w == {"city": "London", "temperature": 7.4, "condition": "light rain", "precipitation": 0.6,
                 "source": weather.ATTRIBUTION}
    assert weather.current("London", fetch=fetch) == w and len(calls) == 1  # served from cache


def test_failures_return_none(monkeypatch):
    monkeypatch.setattr(weather, "_cache", {})

    def boom(req, timeout):
        raise TimeoutError

    assert weather.current("Paris", fetch=boom) is None
    assert weather.current("Paris", fetch=lambda r, timeout: FakeResp(b"{not json")) is None
    assert weather.current("Atlantis") is None
