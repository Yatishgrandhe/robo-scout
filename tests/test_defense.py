import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend import main
from backend.defense import pressure_rating


class DefenseRatingTest(unittest.TestCase):
    def test_short_pressure_gets_credit_and_full_credit_is_capped(self):
        self.assertEqual(pressure_rating(2.6, 114.5, .953), 23)
        self.assertEqual(pressure_rating(1, 110, .90), 9)
        self.assertEqual(pressure_rating(12, 120, 1), 100)
        self.assertEqual(pressure_rating(24, 120, 1), 100)

    def test_rating_requires_visible_pressure_and_enough_shared_tracking(self):
        self.assertIsNone(pressure_rating(0, 120, 1))
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
