import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import main
from backend.defense import add_lane_defense, pressure_rating


class DefenseRatingTest(unittest.TestCase):
    def test_short_pressure_gets_credit_and_full_credit_is_capped(self):
        self.assertEqual(pressure_rating(2.6, 114.5, .953), 23)
        self.assertEqual(pressure_rating(1, 110, .90), 9)
        self.assertEqual(pressure_rating(12, 120, 1), 100)
        self.assertEqual(pressure_rating(24, 120, 1), 100)

    def test_zero_pressure_gets_a_rating_when_shared_tracking_is_sufficient(self):
        self.assertEqual(pressure_rating(0, 108.1, .9), 0)
        self.assertIsNone(pressure_rating(0, 0, 0))
        self.assertIsNone(pressure_rating(1, 9.9, .9))
        self.assertIsNone(pressure_rating(1, 40, .34))

    def test_saved_evidence_is_rerated_without_mutation(self):
        directory = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.object(main, "MATCHES", Path(directory)))
        match = "a" * 32
        folder = main.MATCHES / match
        folder.mkdir()
        main.write(folder / "meta.json", {"id": match, "state": "complete"})
        result = folder / "result.json"
        main.write(result, {"defense": {"13684": {"rating": 2, "pressure_seconds": 2.6,
                                                  "observed_seconds": 114.5, "coverage": .953}}})
        self.assertEqual(main.frames(match)["defense"]["13684"]["rating"], 23)
        self.assertEqual(main.read(result)["defense"]["13684"]["rating"], 2)

    def test_held_lane_counts_when_an_opponent_turns(self):
        frames = []
        for index in range(28):
            x = .30 + .015 * min(index, 10)
            y = .65 - .015 * max(index - 10, 0)
            frames.append({"time": index / 10, "robots": {
                "10": [.53, .55, .63, .65], "20": [x - .05, y - .10, x + .05, y],
            }})
        base = {"pressure_seconds": 0, "observed_seconds": 100, "coverage": 1, "episodes": []}
        result = {"10": base.copy()}
        add_lane_defense(result, frames, {"10": "red", "20": "blue"}, (100, 100), None)
        self.assertGreater(result["10"]["rating"], 0)
        self.assertEqual(result["10"]["lane_episodes"][0]["kind"], "lane_hold")
        self.assertEqual(result["10"]["lane_episodes"][0]["opponent"], "20")

        for frame in frames[8:14]:
            frame["robots"]["20"] = None
        add_lane_defense(result, frames, {"10": "red", "20": "blue"}, (100, 100), None)
        self.assertEqual(result["10"]["rating"], 0)
        self.assertEqual(result["10"]["lane_episodes"], [])
