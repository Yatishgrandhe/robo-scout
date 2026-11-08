"""Features and inference for the local review-trained launch classifier."""

import json

import numpy as np
from scipy.special import expit

FEATURE_COUNT = 10


def shot_features(samples):
    trace = np.asarray(samples, dtype=float)
    if trace.ndim != 2 or trace.shape[1] != 3 or len(trace) < 3 or not np.isfinite(trace).all():
        raise ValueError("A launch needs at least three finite time and position observations.")
    time, x, y = trace.T
    time = time - time[0]
    duration = time[-1]
    if not np.all(np.diff(time) > 0):
        raise ValueError("A launch needs at least three observations with increasing timestamps.")
    progress = time / duration
    rise = y[0] - np.min(y)
    x_line = np.polyval(np.polyfit(progress, x, 1), progress)
    y_curve = np.polyval(np.polyfit(progress, y, 2), progress)
    gaps = np.diff(time)
    return np.array([
        len(time), duration, rise, rise / duration, abs(x[-1] - x[0]),
        np.sqrt(np.mean((x - x_line) ** 2)), np.sqrt(np.mean((y - y_curve) ** 2)),
        np.mean(np.diff(y) < 0), np.max(gaps), np.mean(np.diff(y, n=2) > 0),
    ])


class ShotModel:
    def __init__(self, checkpoint):
        self.mean, self.scale, self.weights = (np.asarray(checkpoint[key], dtype=float) for key in ("mean", "scale", "weights"))
        self.bias = float(checkpoint["bias"])
        if (any(values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all()
                for values in (self.mean, self.scale, self.weights))
                or np.any(self.scale <= 0) or not np.isfinite(self.bias)):
            raise ValueError("Invalid shot model parameters.")

    def score(self, samples):
        features = np.clip((shot_features(samples) - self.mean) / self.scale, -6, 6)
        return float(expit(features @ self.weights + self.bias))

    @classmethod
    def load(cls, path):
        try:
            return cls(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            return None


def prioritize_shots(rows, path):
    candidates = [row for row in rows if row["status"] == "suggested" and row["kind"].endswith("_launch")
                  and row.get("trajectory")]
    model = ShotModel.load(path) if candidates else None
    if model is not None:
        for row in candidates:
            try:
                score = model.score(row["trajectory"])
            except (ValueError, TypeError, IndexError):
                continue
            if np.isfinite(score):
                row["shot_priority"] = round(score, 4)
