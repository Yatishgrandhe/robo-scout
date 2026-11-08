import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import UploadFile

from backend import main
from backend.rules import event


class SavedReviewsTest(unittest.TestCase):
    recording = b"exact source bytes"
    digest = sha256(recording).hexdigest()
    red = {"team": "13684", "alliance": "red"}
    blue = {"team": "15465", "alliance": "blue"}

    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.object(main, "MATCHES", Path(directory)))
        self.reviewed = self.save("a", config={**self.red, "robots": [self.blue]}, events=[
            event("pollen_launch", 1, "Shot", "", status="confirmed"),
            event("cell_entry", 2, "Entered", "", status="confirmed", piece="pollen"),
        ])

    def save(self, key, *, config, events=(), **changes):
        match = key * 32
        folder = main.MATCHES / match
        folder.mkdir()
        main.write(folder / "meta.json", {"id": match, "state": "complete", "config": config,
                                         "name": "Same filename", "video_digests": [self.digest], **changes})
        main.write(folder / "events.json", list(events))
        return match

    def test_exact_duplicate_opens_saved_review_with_requested_team(self):
        duplicate = self.save("b", config=self.blue, events=[
            event("pollen_launch", 1, "Possible launch", "", status="suggested"),
        ])
        self.assertEqual(main.get_match(duplicate)["id"], duplicate)
        result = main.get_match(duplicate, prefer_reviewed=True)
        self.assertEqual(result["id"], self.reviewed)
        self.assertEqual(result["opened_team"], self.blue["team"])
        self.assertTrue(result["reused_review"])
        self.assertEqual(result["team_summaries"][self.red["team"]]["confirmed"]["pollen"]["entered"], 1)
        self.assertEqual(main.rows_for(duplicate)[0]["status"], "suggested")

    def test_other_footage_rosters_and_review_decisions_keep_their_own_id(self):
        cases = [
            {"config": self.red, "video_digests": ["different-digest"]},
            {"config": {**self.red, "alliance": "blue"}},
            {"config": self.red, "media_id": self.reviewed},
            {"config": self.red, "events": [event("pollen_launch", 1, "Dismissed", "", status="dismissed")]},
        ]
        for key, changes in zip("bcde", cases):
            with self.subTest(changes=changes):
                match = self.save(key, **changes)
                self.assertEqual(main.get_match(match, prefer_reviewed=True)["id"], match)

    def test_exact_reimport_returns_saved_review_without_conversion(self):
        with patch.object(main.subprocess, "run") as run:
            result = main.upload(UploadFile(BytesIO(self.recording), filename="Different filename.mov"))
        self.assertEqual(result["id"], self.reviewed)
        self.assertTrue(result["reused_review"])
        self.assertEqual([path.name for path in main.MATCHES.iterdir()], [self.reviewed])
        run.assert_not_called()

    def test_reimport_prefers_broader_team_coverage_over_more_events(self):
        main.write(main.MATCHES / self.reviewed / "events.json", [
            {**event("pollen_launch", index, "Shot", "", status="confirmed"), "team": seed["team"]}
            for index, seed in enumerate((self.red, self.blue))
        ])
        self.save("b", config=self.red, events=[
            event("pollen_launch", index, "Shot", "", status="confirmed") for index in range(3)
        ])
        self.assertEqual(main.saved_review({self.digest}), self.reviewed)
