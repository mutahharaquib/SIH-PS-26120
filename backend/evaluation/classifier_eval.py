"""Train and evaluate the card classifier: split by well, plus a noise-stressed test set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostics.classifier import MODEL_PATH, CardClassifier, report
from simulation.card_dataset import generate


def train_and_evaluate(n_wells: int = 30, per_class: int = 8, seed: int = 0, min_conf: float = 0.55,
                       save: bool = True) -> tuple[CardClassifier, dict]:
    data = generate(n_wells=n_wells, per_class=per_class, seed=seed)
    wells = np.array([s.well for s in data])
    test_wells = set(range(0, n_wells, 4))                       # 25 % of wells held out
    tr = np.array([w not in test_wells for w in wells])
    X = np.stack([s.x for s in data])
    y = [s.label for s in data]
    clf = CardClassifier(min_conf, seed).fit(X[tr], [y[i] for i in np.where(tr)[0]])
    held = report(clf, X[~tr], [y[i] for i in np.where(~tr)[0]])
    stress = generate(n_wells=max(6, n_wells // 4), per_class=per_class, seed=seed + 999, noise_scale=3.0)
    rep_stress = report(clf, np.stack([s.x for s in stress]), [s.label for s in stress])
    rep = {"held_out_by_well": held, "noise_stressed_x3": rep_stress, "n_train": int(tr.sum()),
           "n_test": int((~tr).sum()), "split": "by well", "model": "HistGradientBoosting v1", "tier": "C"}
    if save:
        clf.save()
        (MODEL_PATH.parent / "card_clf_report.json").write_text(json.dumps(rep, indent=2))
    return clf, rep


def ensure_model(path: Path = MODEL_PATH, quick: bool = False) -> CardClassifier:
    """Load the classifier, training it first if no saved model exists."""
    if path.exists():
        return CardClassifier.load(path)
    clf, _ = train_and_evaluate(n_wells=12 if quick else 30, per_class=6 if quick else 8)
    return clf


if __name__ == "__main__":
    _, r = train_and_evaluate()
    print(json.dumps({k: (v["per_class"] if isinstance(v, dict) and "per_class" in v else v) for k, v in r.items()},
                     indent=1))
