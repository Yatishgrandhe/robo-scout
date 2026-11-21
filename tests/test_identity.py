import unittest

import numpy as np

from backend.vision import Identity, TeamTrack


def robot(frame, x, top, bottom):
    frame[100:130, x:x + 60] = top
    frame[130:160, x:x + 60] = bottom
    return np.array((x, 100, x + 60, 160)) / (frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0])


def field():
    return np.full((300, 600, 3), 128, np.uint8)


GREEN, GRAY, RED, BLUE = (40, 200, 60), (110, 110, 110), (40, 40, 220), (220, 80, 30)


class IdentityTest(unittest.TestCase):
    def test_choose_finds_the_seeded_robot_among_others(self):
        frame = field()
        seed = robot(frame, 40, GREEN, GRAY)
        identity = Identity(frame, seed)
        later = field()
        boxes = [robot(later, 400, RED, GRAY), robot(later, 220, GREEN, GRAY), robot(later, 500, BLUE, BLUE)]
        chosen, score = identity.choose(later, boxes)
        self.assertTrue(np.allclose(chosen, boxes[1]))
        self.assertGreater(score, .9)

    def test_choose_refuses_two_lookalikes(self):
        frame = field()
        identity = Identity(frame, robot(frame, 40, GREEN, GRAY))
        later = field()
        boxes = [robot(later, 100, GREEN, GRAY), robot(later, 400, GREEN, GRAY)]
        self.assertIsNone(identity.choose(later, boxes)[0])

    def test_lost_track_is_reacquired_from_detections(self):
        frame = field()
        seed = robot(frame, 40, GREEN, GRAY)
        gray = frame[..., 0].copy()
        track = TeamTrack(frame, gray, seed, None)
        blank = np.full_like(frame, 60)
        for _ in range(20):  # A signal dropout: no robot, no detections.
            track.advance(blank, blank[..., 0].copy(), [], [], False, None)
        later = field()
        boxes = [robot(later, 400, RED, GRAY), robot(later, 250, GREEN, GRAY)]
        found = track.advance(later, later[..., 0].copy(), boxes, [], False, None)
        self.assertTrue(np.allclose(found, boxes[1]))


class ClassifierTrackTest(unittest.TestCase):
    def test_confident_distant_team_robot_takes_over_after_three_samples(self):
        frame = field()
        seed = robot(frame, 40, GREEN, GRAY)
        track = TeamTrack(frame, frame[..., 0].copy(), seed, None)
        decoy = robot(frame, 60, RED, GRAY)
        team = robot(frame, 450, GREEN, GRAY)
        track.classify = lambda image, boxes: [.02 if np.allclose(box, decoy) else .95 for box in boxes]
        results = [track.advance(frame, frame[..., 0].copy(), [decoy, team], [], False, None) for _ in range(3)]
        self.assertFalse(any(result is not None and np.allclose(result, decoy) for result in results))
        self.assertTrue(np.allclose(results[-1], team))


if __name__ == "__main__":
    unittest.main()
