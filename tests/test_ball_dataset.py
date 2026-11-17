import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.build_ball_dataset import apply_review, write_labels
from scripts.evaluate_balls import match


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


if __name__ == "__main__":
    unittest.main()
