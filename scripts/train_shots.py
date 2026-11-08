"""Train a small launch classifier on reviewed traces grouped by original recording."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from backend.shot_model import ShotModel, shot_features


def fit(features, labels):
    mean, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-6)
    design = np.column_stack((np.clip((features - mean) / scale, -6, 6), np.ones(len(labels))))
    positive_rate = labels.mean()
    if not 0 < positive_rate < 1:
        raise ValueError("Training needs both confirmed and dismissed launches.")
    balance = np.where(labels, .5 / positive_rate, .5 / (1 - positive_rate))

    def loss(weights):
        logits = design @ weights
        error = (expit(logits) - labels) * balance
        regularization = np.r_[weights[:-1], 0] * .10
        cost = np.mean(balance * (np.logaddexp(0, logits) - labels * logits)) + .05 * np.sum(weights[:-1] ** 2)
        return cost, design.T @ error / len(labels) + regularization

    weights = minimize(loss, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B").x
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights[:-1].tolist(), "bias": float(weights[-1])}


def average_precision(labels, scores):
    ordered = labels[np.argsort(-scores)]
    return float(np.sum(ordered * np.cumsum(ordered) / np.arange(1, len(ordered) + 1)) / np.sum(ordered))


def train(rows):
    features = np.stack([shot_features(row["samples"]) for row in rows])
    labels = np.array([row["label"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    if len(set(groups)) < 2 or not set(labels) <= {0, 1}:
        raise ValueError("Provide at least two distinct recordings with binary review labels.")
    validation = []
    for group in sorted(set(groups)):
        test = groups == group
        train_labels, test_labels = labels[~test], labels[test]
        if len(set(train_labels)) != 2 or len(set(test_labels)) != 2:
            raise ValueError("Each held-out recording and its training set need both review outcomes.")
        model = ShotModel(fit(features[~test], train_labels))
        scores = np.array([model.score(rows[index]["samples"]) for index in np.flatnonzero(test)])
        validation.append({"group": str(group), "examples": int(test.sum()), "positive": int(test_labels.sum()),
                           "baseline_precision": float(test_labels.mean()), "average_precision": average_precision(test_labels, scores)})
    return {**fit(features, labels), "examples": len(rows), "validation": validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSON list of group, label (0/1) and samples ([time, x, y])")
    parser.add_argument("--output", type=Path, default=Path("models/shot-motion.json"))
    args = parser.parse_args()
    checkpoint = train(json.loads(args.dataset.read_text()))
    args.output.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(json.dumps(checkpoint["validation"], indent=2))


if __name__ == "__main__":
    main()
