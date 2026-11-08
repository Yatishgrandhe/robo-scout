"""Align scoring zones to a reference view using background image features."""

import cv2 as cv
import numpy as np


class CameraRegistration:
    def __init__(self, reference, field):
        self.height, self.width = reference.shape[:2]
        scale = min(1, 700 / max(self.width, self.height))
        self.shape = round(self.width * scale), round(self.height * scale)
        self.scale = scale
        self.orb = cv.ORB_create(nfeatures=650, fastThreshold=9)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        self.mask = np.zeros(self.shape[::-1], np.uint8)
        top = max(.25, (field or (0, .34, 1, 1))[1])
        self.mask[:round(self.shape[1] * top)] = 255
        self.points, self.descriptors = self.features(reference)
        self.transform = np.eye(2, 3, dtype=np.float32)

    def features(self, frame):
        small = cv.resize(frame, self.shape, interpolation=cv.INTER_AREA)
        return self.orb.detectAndCompute(cv.cvtColor(small, cv.COLOR_BGR2GRAY), self.mask)

    def estimate(self, frame):
        points, descriptors = self.features(frame)
        if descriptors is None or self.descriptors is None or len(self.descriptors) < 2:
            return None
        pairs = self.matcher.knnMatch(descriptors, self.descriptors, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .76 * pair[1].distance]
        if len(good) < 14:
            return None
        current = np.float32([points[match.queryIdx].pt for match in good])
        reference = np.float32([self.points[match.trainIdx].pt for match in good])
        transform, inliers = cv.estimateAffinePartial2D(current, reference, method=cv.RANSAC, ransacReprojThreshold=2.5)
        if transform is None or inliers.sum() < max(10, len(good) * .47):
            return None
        scale = np.hypot(transform[0, 0], transform[1, 0])
        angle = abs(np.arctan2(transform[1, 0], transform[0, 0]))
        if not .73 <= scale <= 1.35 or angle > .25 or np.linalg.norm(transform[:, 2]) > .32 * max(self.shape):
            return None
        transform[:, 2] /= self.scale
        return transform

    def update(self, frame):
        transform = self.estimate(frame)
        if transform is not None:
            self.transform = transform

    def stabilize(self, frame, *, reference_sample=False):
        transform = self.estimate(frame) if reference_sample else self.transform
        if transform is None:
            return None
        return cv.warpAffine(frame, transform, (self.width, self.height))

    def boxes(self, zones):
        inverse = cv.invertAffineTransform(self.transform)
        scale = np.array((self.width, self.height))
        result = []
        for zone in zones:
            x0, y0, x1, y1 = zone.box
            corners = np.array(((x0, y0), (x0, y1), (x1, y0), (x1, y1))) * scale
            current = (corners @ inverse[:, :2].T + inverse[:, 2]) / scale
            box = np.clip(np.r_[current.min(axis=0), current.max(axis=0)], 0, 1)
            result.append(zone.model_copy(update={"box": tuple(box)}))
        return result
