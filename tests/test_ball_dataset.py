import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.build_ball_dataset import apply_review, write_labels
from scripts.evaluate_balls import match
from scripts.mine_labels import apply as apply_mined, triage, unlabeled


class BallDatasetTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(self.enterContext(TemporaryDirectory()))
        for split in ("train", "val"):
            (self.out / "images" / split).mkdir(parents=True)
            (self.out / "labels" / split).mkdir(parents=True)
        (self.out / "review").mkdir()
        (self.out / "images/train/a-0001.00.jpg").write_bytes(b"")
        (self.out / "images/val/b-0002.00.jpg").write_bytes(b"")
        self.proposals = [
            {"id": 0, "key": "a-0001.00", "split": "train", "kind": "pollen", "box": [.1, .1, .2, .2]},
            {"id": 1, "key": "a-0001.00", "split": "train", "kind": "red_nectar", "box": [.5, .5, .6, .6]},
            {"id": 2, "key": "b-0002.00", "split": "val", "kind": "blue_nectar", "box": [.3, .3, .4, .4]},
        ]
        (self.out / "proposals.json").write_text(json.dumps(self.proposals))

    def test_review_drops_rejected_boxes_and_relabels(self):
        (self.out / "review/decisions-0000.json").write_text(json.dumps({"reject": [1], "relabel": {"2": "pollen"}}))
        apply_review(Namespace(output=self.out))
        train = (self.out / "labels/train/a-0001.00.txt").read_text().splitlines()
        val = (self.out / "labels/val/b-0002.00.txt").read_text().splitlines()
        self.assertEqual([line.split()[0] for line in train], ["0"])
        self.assertEqual(val[0].split()[:3], ["0", "0.350000", "0.350000"])
        self.assertIn("names:", (self.out / "dataset.yaml").read_text())

    def test_frames_without_kept_boxes_become_empty_negatives(self):
        write_labels(self.out, self.proposals, {0, 1})
        self.assertEqual((self.out / "labels/train/a-0001.00.txt").read_text(), "")

    def test_matching_requires_same_class_and_nearby_center(self):
        truth = [("pollen", np.array((.10, .10, .12, .12))), ("red_nectar", np.array((.5, .5, .52, .52)))]
        predicted = [("pollen", np.array((.101, .101, .121, .121)), .9),
                     ("blue_nectar", np.array((.5, .5, .52, .52)), .8),
                     ("pollen", np.array((.7, .7, .72, .72)), .5)]
        found, unmatched = match(predicted, truth, .6)
        self.assertEqual(found, 1)
        self.assertEqual([row[0] for row in unmatched], ["blue_nectar", "pollen"])


    def test_mined_balls_skip_labeled_spots_and_apply_after_review(self):
        truth = [("pollen", np.array((.10, .10, .12, .12)))]
        predicted = [("red_nectar", np.array((.101, .101, .121, .121)), .9),
                     ("pollen", np.array((.7, .7, .72, .72)), .5)]
        self.assertEqual([row[0] for row in unlabeled(predicted, truth, .6)], ["pollen"])
        (self.out / "mined").mkdir()
        (self.out / "mined/candidates.json").write_text(json.dumps([
            {"id": 0, "split": "train", "key": "a-0001.00", "kind": "blue_nectar", "box": [.7, .7, .72, .72]},
            {"id": 1, "split": "train", "key": "a-0001.00", "kind": "pollen", "box": [.3, .3, .32, .32]}]))
        (self.out / "mined/verdict-a.json").write_text(json.dumps({"ok": [0]}))
        for _ in range(2):
            apply_mined(Namespace(dataset=self.out))
        train = (self.out / "labels/train/a-0001.00.txt").read_text().splitlines()
        self.assertEqual([line.split()[0] for line in train], ["0", "1", "2"])

    def test_triage_auto_accepts_only_confirmed_training_proposals(self):
        proposals = [("pollen", np.array((.1, .1, .12, .12))), ("red_nectar", np.array((.5, .5, .52, .52))),
                     ("blue_nectar", np.array((.3, .3, .32, .32)))]
        predicted = [("pollen", np.array((.101, .101, .121, .121)), .8),
                     ("blue_nectar", np.array((.5, .5, .52, .52)), .9),
                     ("pollen", np.array((.8, .8, .82, .82)), .6)]
        auto, review, dropped = triage(proposals, predicted, .6, .5, review_all=False)
        self.assertEqual([row[1] for row in auto], ["pollen"])
        self.assertEqual([(row[0], row[1]) for row in review],
                         [("color", "red_nectar"), ("color", "blue_nectar"), ("model", "pollen")])
        auto, review, dropped = triage(proposals, predicted, .6, .5, review_all=False, drop_unseen=True)
        self.assertEqual((len(review), dropped), (2, 1))
        auto, review, dropped = triage(proposals, predicted, .6, .5, review_all=True, drop_unseen=True)
        self.assertEqual((len(auto), len(review), dropped), (0, 4, 0))


if __name__ == "__main__":
    unittest.main()
