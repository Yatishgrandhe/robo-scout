import unittest

from backend.hive_entries import arrival, resting_entries, settled_entry, verified_candidates
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

    def test_short_visibility_gap_can_link_only_unambiguous_team(self):
        ascent = [(0, .51, .50), (.08, .51, .43), (.16, .51, .36)]
        launch = {"team": "1", "kind": "pollen", "time": 0, "samples": ascent, "flight": ascent}
        cells = {"red": self.box}
        observations = {"red": [(time + .5, [(x, y)]) for time, x, y in self.path]}
        self.assertIn(0, verified_candidates([launch], cells, {"1": "red"}, observations))
        ambiguous = verified_candidates([launch, {**launch, "team": "2"}], cells,
                                        {"1": "red", "2": "red"}, observations)
        # The entry still counts, but neither robot is credited.
        self.assertEqual([key for key in ambiguous if isinstance(key, int)], [])
        self.assertEqual(len(ambiguous), 1)

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


class SettledEntryTest(unittest.TestCase):
    box = (.38, .07, .49, .21)

    def test_ball_rising_into_cell_and_resting_is_an_entry(self):
        path = [(9.0, .43, .255), (9.03, .43, .237), (9.07, .43, .212), (9.1, .431, .189), (9.13, .431, .171),
                (9.17, .431, .154), (9.2, .43, .143), (9.23, .43, .136), (9.27, .43, .133), (9.3, .431, .133)]
        self.assertEqual(settled_entry(path, self.box).time, 9.1)

    def test_ball_flying_through_cell_is_not_an_entry(self):
        path = [(t / 30, .40 + t * .012, .25 - t * .02) for t in range(12)]
        self.assertIsNone(settled_entry(path, self.box))

    def test_ball_already_in_cell_is_not_a_new_entry(self):
        path = [(t / 30, .44, .14) for t in range(12)]
        self.assertIsNone(settled_entry(path, self.box))


class RestingEntryTest(unittest.TestCase):
    box = (.38, .07, .49, .21)

    def test_each_new_resting_ball_counts_once_until_the_cell_empties(self):
        rows = [(t / 30, [(.41, .19)] + ([(.45, .19)] if t >= 30 else [])) for t in range(60)]
        rows += [(2 + t / 30, []) for t in range(60)]          # Cell empty for 2 s: a TIP
        rows += [(4 + t / 30, [(.41, .19)]) for t in range(30)]  # same spot after the TIP is a new entry
        self.assertEqual([round(hit.time, 2) for hit in resting_entries(rows, self.box)], [0, 1, 4])

    def test_flicker_of_a_resting_ball_is_not_a_new_entry(self):
        rows = [(t / 30, [(.41, .19)] if t % 7 else []) for t in range(90)]
        self.assertEqual(len(resting_entries(rows, self.box)), 1)


if __name__ == "__main__":
    unittest.main()
