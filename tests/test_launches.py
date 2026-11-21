import unittest

import numpy as np

from backend.launches import Blob, RobotBoxes, Trail, _candidate, _deduplicate

WIDTH, HEIGHT, FPS = 960, 540, 60
ROBOT = [365 / WIDTH, 319 / HEIGHT, 470 / WIDTH, 410 / HEIGHT]


def robot_boxes():
    frames = [{"time": round(8 + i / 10, 3), "robot": ROBOT} for i in range(20)]
    return RobotBoxes(frames, WIDTH, HEIGHT)


def trail(points, start=8.6):
    observations = [(start + i / FPS, Blob("pollen", np.array(point, float), 14., False))
                    for i, point in enumerate(points)]
    return Trail("pollen", observations)


def rising(x, y, frames, step=20):
    return [(x, y - step * (i + 1)) for i in range(frames)]


class LaunchCandidateTest(unittest.TestCase):
    def test_ball_launched_from_rest_on_robot_is_found(self):
        held = [(422, 333)] * 6
        found = _candidate(trail(held + rising(422, 333, 8)), robot_boxes(), WIDTH, HEIGHT)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["time"], 8.6 + 5 / FPS, delta=.04)

    def test_ball_launched_without_a_rest_is_still_found(self):
        self.assertIsNotNone(_candidate(trail(rising(422, 353, 9)), robot_boxes(), WIDTH, HEIGHT))

    def test_rise_after_rolling_is_not_a_launch(self):
        rolling = [(300 + 12 * i, 333) for i in range(6)]
        self.assertIsNone(_candidate(trail(rolling + rising(372, 333, 8)), robot_boxes(), WIDTH, HEIGHT))

    def test_volley_balls_a_tenth_of_a_second_apart_stay_separate(self):
        boxes = robot_boxes()
        first = _candidate(trail([(422, 333)] * 6 + rising(422, 333, 8)), boxes, WIDTH, HEIGHT)
        second = _candidate(trail(rising(417, 335, 9), start=8.8), boxes, WIDTH, HEIGHT)
        self.assertEqual(len(_deduplicate([first, second], WIDTH, HEIGHT)), 2)


if __name__ == "__main__":
    unittest.main()
