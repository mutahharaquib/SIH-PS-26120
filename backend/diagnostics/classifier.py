"""Card fault classifier (BUILD_SPEC §6.3). v1: HistGradientBoosting on card features.

Classes: normal, fluid_pound, gas_interference, rod_float, pump_unseated.
Outputs probabilities; `unknown` when max probability < clf_min_conf. Trained on the
synthetic simulator (Tier C), split by well, with a separate noise-stressed test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from diagnostics.card_features import features, vectorize

CLASSES = ["normal", "fluid_pound", "gas_interference", "rod_float", "pump_unseated"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "card_clf.joblib"


@dataclass
class Prediction:
    label: str
    probs: dict[str, float]
    confident: bool


class CardClassifier:
    def __init__(self, min_conf: float = 0.55, seed: int = 0):
        self.min_conf = min_conf
        self.model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                                    l2_regularization=1e-3, random_state=seed)
        self.fitted = False

    def fit(self, X: np.ndarray, y: list[str]) -> "CardClassifier":
        self.model.fit(X, np.array([CLASSES.index(c) for c in y]))
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((len(X), len(CLASSES)))
        P[:, self.model.classes_] = self.model.predict_proba(X)
        return P

    def predict(self, dh_pos, dh_load, surf_pos=None, surf_load=None, F_ref=None) -> Prediction:
        x = vectorize(features(dh_pos, dh_load, surf_pos, surf_load, F_ref))[None, :]
        p = self.predict_proba(x)[0]
        k = int(np.argmax(p))
        conf = bool(p[k] >= self.min_conf)
        return Prediction(CLASSES[k] if conf else "unknown", {c: float(v) for c, v in zip(CLASSES, p)}, conf)

    def save(self, path: Path = MODEL_PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "min_conf": self.min_conf}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CardClassifier":
        import joblib

        d = joblib.load(path)
        c = cls(d["min_conf"])
        c.model = d["model"]
        c.fitted = True
        return c


def report(clf: CardClassifier, X: np.ndarray, y: list[str]) -> dict:
    """Per-class precision/recall/F1 and confusion matrix (unknowns counted as errors)."""
    P = clf.predict_proba(X)
    pred = [CLASSES[int(np.argmax(p))] if p.max() >= clf.min_conf else "unknown" for p in P]
    labels = CLASSES + ["unknown"]
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "classes": CLASSES,
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "accuracy": float(np.mean(np.array(pred) == np.array(y))),
        "unknown_rate": float(np.mean(np.array(pred) == "unknown")),
        "confusion": {"labels": labels, "matrix": cm.tolist()},
        "tier": "C",
    }
