import unittest

from backend.hive_entries import arrival, verified_candidates
from backend.main import linked_count
from backend.rules import event, summarize


class HiveEntriesTest(unittest.TestCase):
    box = (.4, .1, .6, .3)
    path = [(0, .5, .14), (.08, .5, .18), (.16, .5, .22),
            (.24, .5, .255), (.32, .5, .253), (.4, .5, .252)]

    def test_descent_and_settlement_make_a_candidate(self):
        self.assertEqual(arrival(self.path, self.box).time, .16)

    def test_visible_drop_below_cell_is_not_an_entry(self):
        self.assertIsNone(arrival([*self.path, (.48, .5, .35)], self.box))

    def test_rising_ball_is_not_an_entry(self):
        path = [(i * .08, x, y) for i, (_, x, y) in enumerate(reversed(self.path))]
        self.assertIsNone(arrival(path, self.box))

    def test_flight_must_match_the_marked_teams_alliance(self):
        launches = [{"team": "1", "kind": "pollen", "time": 0, "flight": self.path}]
        self.assertIn(0, verified_candidates(launches, {"red": self.box}, {"1": "red"}))
        self.assertFalse(verified_candidates(launches, {"red": self.box}, {"1": "blue"}))

    def test_entry_and_final_piece_use_separate_tallies(self):
        launch = event("pollen_launch", 1, "Shot", "", status="confirmed")
        entered = event("cell_entry", 2, "Entered", "", status="confirmed", piece="pollen")
        retained = event("cell_ball", 3, "Final", "", status="confirmed", piece="pollen")
        for row in (entered, retained):
            row["launch_id"] = launch["id"]
        events = [launch, entered, retained]
        self.assertEqual(linked_count(events, launch["id"], "cell_entry"), 1)
        self.assertEqual(linked_count(events, launch["id"], "cell_ball"), 1)
        self.assertEqual(summarize(events)["confirmed"]["pollen"], {"shot": 1, "entered": 1, "scored": 1})


if __name__ == "__main__":
    unittest.main()
