import unittest
from types import SimpleNamespace

import cv2 as cv
import numpy as np

from backend.flowers import FlowerVisit, early_flower_nectar, final_flowers, flower_detections, flower_overlap, from_top
from backend.main import Zone
from backend.vision import match_tracks


class FlowerTrackingTest(unittest.TestCase):
    zone = Zone(kind="flower", alliance="red", box=(.4, .3, .6, .65))

    def test_partial_ball_overlap_counts_but_projected_box_corner_does_not(self):
        self.assertTrue(flower_overlap((.45, .25, .51, .31), self.zone.box))
        self.assertFalse(flower_overlap((.34, .24, .405, .305), self.zone.box))
        self.assertFalse(flower_overlap((.45, .67, .51, .73), self.zone.box))

    def test_diagonal_motion_across_the_ring_from_the_side_is_not_a_top_entry(self):
        self.assertFalse(from_top((.31, .25, .36, .30), self.zone.box, (.42, .28, .47, .33), self.zone.box))
        self.assertTrue(from_top((.42, .25, .47, .30), self.zone.box, (.43, .28, .48, .33), self.zone.box))

    def test_owner_and_bottom_bonus_follow_nectar_order_not_zone_or_placing_robot(self):
        def piece(kind, y, actor):
            return SimpleNamespace(kind=kind, samples=[(99, (.47, y - .02, .51, y + .02), False)],
                                   flower_visit=FlowerVisit(0, 98, 9, True, actor))

        tracks = [piece("pollen", .34, "11"), piece("blue_nectar", .43, "22"), piece("red_nectar", .56, "11")]
        snapshot, events = final_flowers(tracks, [self.zone], 99, .28, {"11": "red", "22": "blue"})
        self.assertEqual(snapshot[0]["pieces"], {"pollen": 1, "blue_nectar": 1, "red_nectar": 1})
        self.assertEqual((snapshot[0]["owner"], snapshot[0]["bottom_nectar"]), ("blue", "red"))
        self.assertEqual((events[0]["team"], events[0]["flower_owner"], events[0]["status"]), ("11", "blue", "suggested"))
        tracks[1].samples[-1] = (99, (.47, .535, .51, .575), False)
        ambiguous, _ = final_flowers(tracks, [self.zone], 99, .28, {"11": "red", "22": "blue"})
        self.assertIsNone(ambiguous[0]["owner"])

    def test_early_nectar_crossing_is_flagged_separately_and_removal_clears_contents(self):
        def tracks_for(kind, start):
            tracks = []
            for i, y in enumerate((.22, .24, .26, .289, .32, .36, .38, .38, .38)):
                box = np.array((.477, y - .025, .527, y + .025))
                match_tracks(tracks, [(kind, box, .9)], start + i / 10, {"11": (.43, .18, .56, .27)}, [self.zone], "11")
            return tracks

        early = tracks_for("red_nectar", 95)
        flagged = set()
        penalty = early_flower_nectar(early[0], 95.8, 0, flagged)
        self.assertEqual((penalty["rule"], penalty["severity"], penalty["team"]), ("G410", "major", "11"))
        self.assertIsNone(early_flower_nectar(early[0], 95.8, 0, flagged))
        self.assertIsNone(early_flower_nectar(tracks_for("red_nectar", 98)[0], 98.8, 0, set()))
        self.assertIsNone(early_flower_nectar(tracks_for("pollen", 95)[0], 95.8, 0, set()))
        self.assertEqual(final_flowers(early, [self.zone], 95.8, .28, {"11": "red"})[0][0]["owner"], "red")

        track = early[0]
        track.update(95.9, np.array((.48, .66, .53, .71)), None, [self.zone], {0: self.zone.box}, "11")
        snapshot, events = final_flowers(early, [self.zone], 95.9, .28, {"11": "red"})
        self.assertEqual(snapshot[0]["pieces"]["red_nectar"], 0)
        self.assertEqual(events, [])

    def test_crop_separates_two_touching_pollen(self):
        image = np.zeros((300, 300, 3), np.uint8)
        zone = Zone(kind="flower", alliance="red", box=(.45, .25, .55, .65))
        for y in (150, 170):
            cv.circle(image, (150, y), 11, (0, 223, 255), -1)
        detections = flower_detections(image, [zone])
        self.assertEqual([kind for kind, _, _ in detections], ["pollen", "pollen"])
