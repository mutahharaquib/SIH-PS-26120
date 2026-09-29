"""M10 — API smoke test over a 2-well live runner (TestClient)."""

import os

import pytest

os.environ.setdefault("TWIN_N_WELLS", "2")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        r = c.post("/sim", json={"step_hours": 24 * 35})
        assert r.status_code == 200
        yield c


def test_wells_and_state(client):
    w = client.get("/wells").json()
    assert len(w["wells"]) == 2
    row = w["wells"][0]
    assert {"risk_rank", "rfi", "q_oil_m3d", "tiers"} <= set(row)
    s = client.get("/wells/W01/state").json()
    assert s["well_id"] == "W01" and s["reservoir"]["T_avg_heated"]["prov"]["tier"] in "ABC"
    f = client.get("/wells/W01/state?units=field").json()
    assert f["reservoir"]["T_avg_heated"]["unit"] == "degF"
    assert client.get("/wells/NOPE/state").status_code == 404


def test_history_cards_cutoff_recommendation(client):
    h = client.get("/wells/W01/history?fields=rfi,q_oil").json()
    assert h["rows"] and set(h["rows"][-1]) <= {"t", "rfi", "q_oil"}
    assert "cards" in client.get("/wells/W01/cards").json()
    assert "g_star" in client.get("/wells/W01/cutoff").json()
    rec = client.get("/wells/W01/controller/recommendation").json()
    assert rec["mode"] == "advisory"
    assert "risk" in client.get("/wells/W01/diagnostics").json()
    assert "cycles" in client.get("/wells/W01/reservoir").json()


def test_mode_and_supervised_approval(client):
    r = client.post("/wells/W01/mode", json={"mode": "supervised", "band_spm": 0.4}).json()
    assert r["mode"] == "supervised" and r["bands"]["spm"] == 0.4
    client.post("/sim", json={"step_hours": 24})
    pend = client.get("/wells/W01/controller/recommendation").json()["pending"]
    if pend:
        out = client.post("/wells/W01/controller/approve", json={"action_id": pend[-1]["id"]}).json()
        assert set(out["written"]) == {"set_spm", "set_profile"}
    audit = client.get("/wells/W01/controller/recommendation").json()["audit"]
    assert any(a["event"] == "mode_change" for a in audit)


def test_whatif_has_no_side_effects(client):
    before = client.get("/wells/W02/state").json()["t"]
    out = client.post("/whatif", json={"well_id": "W02", "steam_t": 4000, "spm": 3.0, "downstroke": 0.6}).json()
    assert {"plan", "scenario", "delta"} <= set(out)
    assert client.get("/wells/W02/state").json()["t"] == before


def test_config_about_eval(client):
    c = client.get("/config").json()
    assert "economics.oil_price" in c["placeholders"]
    assert client.get("/about").json()["limitations"]
    assert "runs" in client.get("/evaluation").json()


def test_websocket_stream(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["wells"]
