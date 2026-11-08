import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.shot_model import FEATURE_COUNT, prioritize_shots


class ShotModelTest(unittest.TestCase):
    def test_model_prioritizes_only_pending_launches_with_valid_traces(self):
        track = [[0, .4, .6], [.04, .4, .55], [.08, .41, .52]]
        rows = [
            {"kind": "pollen_launch", "status": "suggested", "trajectory": track},
            {"kind": "pollen_launch", "status": "confirmed", "trajectory": track},
            {"kind": "cell_entry", "status": "suggested", "trajectory": track},
            {"kind": "pollen_launch", "status": "suggested", "trajectory": [track[0]] * 3},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shot-model.json"
            path.write_text(json.dumps({"mean": [0] * FEATURE_COUNT, "scale": [1] * FEATURE_COUNT,
                                       "weights": [0] * FEATURE_COUNT, "bias": 0}))
            prioritize_shots(rows, path)
        self.assertEqual(rows[0]["shot_priority"], .5)
        self.assertTrue(all("shot_priority" not in row for row in rows[1:]))
        self.assertEqual(rows[0]["status"], "suggested")
