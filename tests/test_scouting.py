import unittest

from backend.rules import event, event_piece, summarize
from backend.scouting import scouting_report


class ScoutingReportTest(unittest.TestCase):
    roster = {"10": "red", "11": "red", "20": "blue", "21": "blue"}

    def entry(self, team, time, count=1, status="confirmed"):
        row = event("cell_entry", time, "Reviewed Hive entry", "", count=count, status=status, piece="pollen")
        return {**row, "team": team}

    def review(self, rows):
        rows = sorted(({**row, "piece": event_piece(row)} for row in rows), key=lambda row: row["time"])
        summaries = {team: summarize([row for row in rows if row.get("team", "10") == team], alliance)
                     for team, alliance in self.roster.items()}
        return scouting_report(self.roster, "10", rows, summaries)

    def test_fluidity_uses_only_confirmed_entries_from_each_alliance(self):
        red = [self.entry("10", 1, 2), self.entry("11", 8, 2), self.entry("10", 19)]
        blue = [self.entry("20", 2), self.entry("21", 3, status="suggested")]
        result = self.review([*red, *blue, self.entry("11", 9, 5, status="dismissed")])
        self.assertEqual(result["alliances"]["red"]["rating"], 90)
        self.assertEqual(result["alliances"]["red"]["entries"], {"10": 3, "11": 2})
        self.assertEqual(len(result["alliances"]["red"]["switches"]), 2)
        self.assertIsNone(result["alliances"]["blue"]["rating"])
        self.assertIn("3 Pollen Hive entries", result["robots"]["10"]["comment"])
        self.assertEqual(result["robots"]["10"]["clip"]["time"], 1)
        self.assertIsNone(result["robots"]["21"]["clip"])

    def test_ambiguous_simultaneous_entries_do_not_create_switches(self):
        entries = [self.entry("10", 1), self.entry("10", 2), self.entry("11", 2), self.entry("11", 3)]
        result = self.review(entries)["alliances"]["red"]
        self.assertEqual(result["entries"], {"10": 2, "11": 2})
        self.assertEqual(result["transitions"], 0)
        self.assertEqual(result["switches"], [])
        self.assertIsNone(result["rating"])

    def test_scoring_roles_get_different_performance_clips(self):
        release = event("red_nectar_launch", 5, "Nectar release", "", status="confirmed")
        entry = self.entry("10", 1, 2)
        result = self.review([{**release, "team": "11"}, entry])["robots"]
        self.assertIn("highlighted 2-piece volley", result["10"]["comment"])
        self.assertIn("check its landing", result["11"]["comment"])
        self.assertEqual(result["11"]["clip"]["time"], 5)
