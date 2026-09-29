"""M5 — RFI, card features, classifier (with report), efficiency."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from diagnostics import card_features as cf
from diagnostics import efficiency, rfi
from diagnostics.classifier import CLASSES, CardClassifier, report
from physics.pump import DownholePump, PumpCondition
from simulation.card_dataset import generate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_rfi_value_and_levels():
    assert rfi.rfi(0.5, 0.5) == 1.0
    assert rfi.level(0.85, 0.8, 1.0) == "warn" and rfi.level(1.2, 0.8, 1.0) == "alarm"
    assert rfi.level(0.3, 0.8, 1.0) == "ok"


def test_rfi_tracker_slope_per_stroke():
    tr = rfi.RFITracker()
    for i in range(20):
        tr.update(T0 + timedelta(minutes=i), 0.1 + 0.04 * i, spm=4.0)     # +0.04 per minute
    assert tr.slope_per_stroke() == pytest.approx(0.01, rel=1e-6)          # 4 strokes per minute


def test_lead_times():
    times = [T0 + timedelta(hours=h) for h in range(48)]
    values = [0.2 + 0.03 * h for h in range(48)]                          # crosses 0.8 at h=20
    onsets = [T0 + timedelta(hours=30), T0 + timedelta(hours=5)]
    lt = rfi.lead_times(times, values, onsets, warn=0.8)
    assert lt[0] == pytest.approx(10.0) and lt[1] is None


def _pump_card(cond):
    t = np.linspace(0, 2 * np.pi, 400)
    u = -1.0 * (1 - np.cos(t))
    v = np.gradient(u)
    p = DownholePump(1.55e-3, 10e3, friction=0.0, cond=cond, S_p=2.0)
    return -u, np.array([p.force(a, b) for a, b in zip(u, v)])


def test_features_and_fillage():
    pos, load = _pump_card(PumpCondition(1.0))
    x, y = cf.normalize(pos, load)
    assert x.min() == 0 and x.max() == 1 and y.min() == 0 and y.max() == 1
    assert cf.fillage_estimate(pos, load) > 0.9
    for fill in (0.5, 0.7):
        p2, l2 = _pump_card(PumpCondition(fill))
        assert efficiency.fillage_from_card(p2, l2) == pytest.approx(fill, abs=0.06)
    f = cf.features(pos, load)
    assert f["fd"].shape == (16,) and f["grid"].shape == (32, 32) and f["angles"].shape == (4,)
    assert cf.vectorize(f).ndim == 1


def test_volumetric_efficiency():
    d = efficiency.theoretical_displacement_m3d(1.55e-3, 2.2, 4.0)
    assert d == pytest.approx(1.55e-3 * 2.2 * 4 * 1440)
    assert efficiency.volumetric_efficiency(0.8 * d, 1.55e-3, 2.2, 4.0) == pytest.approx(0.8)


@pytest.fixture(scope="module")
def small_dataset():
    return generate(n_wells=10, per_class=5, seed=42)


def test_classifier_by_well_split_and_report(small_dataset):
    data = small_dataset
    wells = np.array([s.well for s in data])
    test_w = {0, 5}
    tr = np.array([w not in test_w for w in wells])
    X = np.stack([s.x for s in data])
    y = [s.label for s in data]
    assert set(y) == set(CLASSES)
    clf = CardClassifier(0.5).fit(X[tr], [y[i] for i in np.where(tr)[0]])
    rep = report(clf, X[~tr], [y[i] for i in np.where(~tr)[0]])
    assert set(rep["per_class"]) == set(CLASSES)
    assert rep["accuracy"] > 0.8
    assert rep["per_class"]["pump_unseated"]["recall"] > 0.9
    assert len(rep["confusion"]["matrix"]) == len(CLASSES) + 1
    pred = clf.predict(*data[0].dh, *data[0].surf)
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-6


def test_unknown_when_low_confidence(small_dataset):
    data = small_dataset
    X = np.stack([s.x for s in data])
    clf = CardClassifier(min_conf=0.999999).fit(X, [s.label for s in data])
    noisy = np.random.default_rng(0).normal(0, 1, (5, X.shape[1]))
    P = clf.predict_proba(noisy)
    labels = [CLASSES[int(np.argmax(p))] if p.max() >= clf.min_conf else "unknown" for p in P]
    assert "unknown" in labels
