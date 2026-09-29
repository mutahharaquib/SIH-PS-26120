"""M1 — core: config (placeholder flags), units, steam, state, bus, persistence."""

from datetime import datetime, timezone

import numpy as np
import pytest

from core import steam, units
from core.bus import Bus
from core.config import get_config
from core.state import Card, Tier, iter_quantities, q
from core.store import Store


def test_every_config_leaf_has_source_and_unit():
    cfg = get_config()
    leaves = cfg.leaves()
    assert len(leaves) > 50
    for path, node in leaves:
        assert "source" in node and node["source"], path
        assert "unit" in node, path


def test_placeholder_flags():
    cfg = get_config()
    assert cfg.is_placeholder("field.reservoir.J_cold")
    assert not cfg.is_placeholder("field.crude.api_gravity")
    assert cfg.meta("field.crude.api_gravity")["tier"] == "A"
    assert cfg.v("field.reservoir.T_R") == [46.0, 48.0]
    assert cfg.v("field.crude.api_gravity") == [17.0, 19.0]
    assert "economics.oil_price" in cfg.placeholders()


def test_overrides_do_not_mutate_base():
    cfg = get_config()
    c2 = cfg.with_overrides({"economics.oil_price": 50.0})
    assert c2.v("economics.oil_price") == 50.0
    assert cfg.v("economics.oil_price") == 70.0


def test_units_roundtrip():
    assert units.k_to_c(units.c_to_k(47.0)) == pytest.approx(47.0)
    assert units.f_to_k(units.k_to_f(320.0)) == pytest.approx(320.0)
    assert units.psi_to_pa(units.pa_to_psi(1e6)) == pytest.approx(1e6)
    assert units.m3d_to_bbld(1.0) == pytest.approx(6.2898, rel=1e-4)
    assert units.api_to_sg(10.0) == pytest.approx(1.0, rel=1e-3)
    v, u = units.to_field(1.0, "Pa.s")
    assert (v, u) == (1000.0, "cP")


def test_steam_properties_match_iapws_reference():
    # IAPWS-IF97 saturation at 1 MPa: T = 453.03 K, h_fg = 2014.6 kJ/kg
    assert steam.T_sat(1.0e6) == pytest.approx(453.03, abs=0.1)
    assert steam.h_fg(1.0e6) == pytest.approx(2014.6e3, rel=2e-3)
    ps = np.geomspace(2e5, 1.5e7, 20)
    assert np.all(np.diff(steam.T_sat(ps)) > 0)
    assert np.all(np.diff(steam.h_fg(ps)) < 0)
    assert steam.p_sat(float(steam.T_sat(5e6))) == pytest.approx(5e6, rel=1e-3)


def test_quantity_carries_provenance():
    x = q(1.0, "m", Tier.B, "unit-test", placeholder=True, inputs={"a": 1})
    assert x.prov.tier == Tier.B and x.prov.placeholder and len(x.prov.inputs_hash) == 12


def test_bus_pubsub():
    b = Bus()
    got = []
    unsub = b.subscribe("w1", lambda t, p: got.append(p))
    b.publish("w1", 1)
    unsub()
    b.publish("w1", 2)
    assert got == [1] and b.latest("w1") == 2


def test_store_cards_and_states(sample_state):
    s = Store()
    s.save_state(sample_state)
    card = Card(position=[0, 1, 0], load=[1, 2, 1], kind="surface", t=datetime.now(timezone.utc), spm=4.0)
    s.save_card("W01", card)
    hist = s.history("W01")
    assert len(hist) == 1 and hist[0].well_id == "W01"
    assert s.cards("W01")[0].load == [1, 2, 1]
    # every quantity in a persisted state keeps its tier
    assert all(qq.prov.tier in Tier for _, qq in iter_quantities(hist[0]))
