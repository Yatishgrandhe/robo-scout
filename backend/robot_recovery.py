"""Appearance and motion proposals for recovering a selected robot after a gap."""

import cv2 as cv
import numpy as np


def pixels(box, shape):
    height, width = shape[:2]
    return np.rint(np.clip(box, 0, 1) * (width, height, width, height)).astype(int)


def histogram(hsv):
    hist = cv.calcHist([hsv], [0, 1], None, [18, 6], [0, 180, 0, 256])
    return cv.normalize(hist, None, alpha=1, norm_type=cv.NORM_L1)


def texture(gray):
    patch = cv.resize(gray, (40, 40), interpolation=cv.INTER_AREA)
    return float(np.mean(cv.Canny(patch, 45, 115) > 0))


class AppearanceRecovery:
    """Find a robot-sized patch with the selected robot's tint and appearance.

    Local candidates need support from the last trusted location. Distant
    candidates also need foreground motion, so a matching static object cannot
    normally take over the track after an occlusion.
    """

    def __init__(self, seed_frame, normalized_seed_box, field_box=None):
        self.field = (0, 0, 1, 1) if field_box is None else field_box
        self.previous_gray = cv.cvtColor(seed_frame, cv.COLOR_BGR2GRAY)
        self.previous_small = self._small(self.previous_gray)
        self.hue = None
        x0, y0, x1, y1 = pixels(normalized_seed_box, seed_frame.shape)
        self.width, self.height = x1 - x0, y1 - y0
        if min(self.width, self.height) < 8:
            return
        hsv = cv.cvtColor(seed_frame[y0:y1, x0:x1], cv.COLOR_BGR2HSV)
        saturated = (hsv[..., 1] >= 96) & (hsv[..., 2] >= 63)
        if np.count_nonzero(saturated) < max(18, saturated.size * .055):
            return
        hues = np.bincount(hsv[..., 0][saturated], minlength=180)
        smooth = sum(np.roll(hues, shift) for shift in range(-5, 6))
        dominant = int(np.argmax(smooth))
        if smooth[dominant] < max(16, saturated.size * .045):
            return
        self.hue = dominant
        self.tint_fraction = np.count_nonzero(self._tint(hsv)) / saturated.size
        self.seed_hist = histogram(hsv)
        self.seed_texture = texture(self.previous_gray[y0:y1, x0:x1])
        self.seed_small = self.previous_small.copy()
        self.fixed_tint = self._tint(cv.cvtColor(seed_frame, cv.COLOR_BGR2HSV))
        margin = max(3, min(self.width, self.height) // 6)
        self.fixed_tint[max(0, y0-margin):y1+margin, max(0, x0-margin):x1+margin] = 0
        # Broad support tolerates small perspective changes around the camera
        # translation. This is a prior on background clutter, not a hard mask.
        span = max(7, round(min(seed_frame.shape[:2]) * .036) | 1)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (span, span))
        self.fixed_tint = cv.dilate(self.fixed_tint, kernel)
        # Expand tint proposals to include wheels and structural parts beyond
        # the colored bodywork. Relative padding adapts to perspective.
        tx, ty, tw, th = cv.boundingRect(self._tint(hsv))
        self.padding = np.clip(np.array((tx / tw, ty / th,
                                         (self.width - tx - tw) / tw,
                                         (self.height - ty - th) / th)), 0, .45)

    @staticmethod
    def _small(gray):
        height, width = gray.shape
        scale = min(1, 160 / max(height, width))
        return cv.resize(gray, (max(16, round(width * scale)), max(16, round(height * scale))),
                         interpolation=cv.INTER_AREA).astype(np.float32)

    def _tint(self, hsv):
        hue, saturation, value = cv.split(hsv)
        distance = np.abs(hue.astype(np.int16) - self.hue)
        return ((np.minimum(distance, 180 - distance) <= 11)
                & (saturation >= 89) & (value >= 60)).astype(np.uint8) * 255

    def _motion(self, gray):
        small = self._small(gray)
        previous = self.previous_gray
        if small.shape == self.previous_small.shape and previous.shape == gray.shape:
            (dx, dy), reliability = cv.phaseCorrelate(self.previous_small, small)
            if reliability >= .14 and np.isfinite(dx + dy):
                sx = dx * gray.shape[1] / small.shape[1]
                sy = dy * gray.shape[0] / small.shape[0]
                if 1.4 < np.hypot(sx, sy) < .12 * min(gray.shape):
                    matrix = np.float32(((1, 0, sx), (0, 1, sy)))
                    previous = cv.warpAffine(previous, matrix, gray.shape[::-1], borderMode=cv.BORDER_REPLICATE)
            motion = cv.absdiff(gray, previous)
        else:
            motion = np.zeros_like(gray)
        self.previous_gray, self.previous_small = gray, small
        return motion

    def _fixed_background(self, gray, transform):
        if gray.shape != self.fixed_tint.shape or self.previous_small.shape != self.seed_small.shape:
            return np.zeros_like(gray)
        if transform is not None:
            inverse = cv.invertAffineTransform(transform)
            return cv.warpAffine(self.fixed_tint, inverse, gray.shape[::-1], flags=cv.INTER_NEAREST)
        (dx, dy), reliability = cv.phaseCorrelate(self.seed_small, self.previous_small)
        if reliability >= .17 and np.isfinite(dx + dy):
            sx = dx * gray.shape[1] / self.seed_small.shape[1]
            sy = dy * gray.shape[0] / self.seed_small.shape[0]
            if 1.4 < np.hypot(sx, sy) < .22 * min(gray.shape):
                matrix = np.float32(((1, 0, sx), (0, 1, sy)))
                return cv.warpAffine(self.fixed_tint, matrix, gray.shape[::-1], flags=cv.INTER_NEAREST)
        return self.fixed_tint

    def find(self, frame, previous_box=None, *, lost=False, background_transform=None):
        if self.hue is None:
            return None
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        motion = self._motion(gray)
        background = self._fixed_background(gray, background_transform)
        height, width = gray.shape
        left, top, right, bottom = pixels(self.field, frame.shape)
        if min(right - left, bottom - top) < 8:
            return None
        hsv = cv.cvtColor(frame[top:bottom, left:right], cv.COLOR_BGR2HSV)
        tint = self._tint(hsv)
        # A registered reference can separate the moving chassis from a Hive
        # or field barrier even when their colored pixels touch.
        search_tint = tint
        if background_transform is not None:
            search_tint = cv.bitwise_and(tint, cv.bitwise_not(background[top:bottom, left:right]))
        size = max(3, min(17, round(min(self.width, self.height) * .13) | 1))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        joined = cv.morphologyEx(search_tint, cv.MORPH_CLOSE, kernel)
        joined = cv.dilate(joined, kernel)
        contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        prior = pixels(previous_box, frame.shape) if previous_box is not None else None
        best, best_score, runner_up = None, -np.inf, -np.inf
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w < max(7, .20 * self.width) or h < max(6, .19 * self.height):
                continue
            tx, ty, tw, th = cv.boundingRect(search_tint[y:y+h, x:x+w])
            if not tw or not th:
                continue
            x, y = x + tx, y + ty
            padded = np.rint(self.padding * (tw, th, tw, th)).astype(int)
            x0, y0 = max(0, x - padded[0]), max(0, y - padded[1])
            x1, y1 = min(right-left, x+tw+padded[2]), min(bottom-top, y+th+padded[3])
            bw, bh = x1 - x0, y1 - y0
            if not (.27 * self.width <= bw <= 2.9 * self.width and .23 * self.height <= bh <= 2.8 * self.height):
                continue
            aspect = (bw / bh) / (self.width / self.height)
            if not .43 <= aspect <= 2.35:
                continue
            tinted = tint[y0:y1, x0:x1] > 0
            tint_fraction = np.count_nonzero(tinted) / (bw * bh)
            if tint_fraction < max(.065, self.tint_fraction * .37):
                continue
            box = np.array((x0+left, y0+top, x1+left, y1+top))
            distance = np.inf
            overlap = 0
            if prior is not None:
                delta = (box[:2] + box[2:] - prior[:2] - prior[2:]) / 2
                distance = float(np.linalg.norm(delta / np.maximum((self.width, self.height), prior[2:] - prior[:2])))
                intersect = np.maximum(0, np.minimum(box[2:], prior[2:]) - np.maximum(box[:2], prior[:2])).prod()
                overlap = intersect / max(1, bw * bh + (prior[2:] - prior[:2]).prod() - intersect)
            if not lost and prior is not None and distance > 1.35:
                continue
            foreground = motion[box[1]:box[3], box[0]:box[2]]
            moved = float(np.mean(foreground >= 22))
            if overlap < .42 and moved < (.14 if lost else .10):
                continue
            fixed = background[box[1]:box[3], box[0]:box[2]] > 0
            fixed_fraction = float(np.mean(fixed[tinted]))
            if fixed_fraction > .70 and moved < .24:
                continue
            patch = hsv[y0:y1, x0:x1]
            appearance = 1 - cv.compareHist(self.seed_hist, histogram(patch), cv.HISTCMP_BHATTACHARYYA)
            if appearance < (.40 if distance <= 1 else .52):
                continue
            patch_texture = texture(gray[box[1]:box[3], box[0]:box[2]])
            if patch_texture < self.seed_texture * .22:
                continue
            score = (1.8 * appearance + .6 * min(moved, .55) + .15 * min(tint_fraction, .5)
                     - .13 * abs(np.log(aspect)) - .22 * min(distance, 2.0)
                     - .06 * abs(np.log((patch_texture + .02) / (self.seed_texture + .02)))
                     - .34 * fixed_fraction)
            if score > best_score:
                best, best_score, runner_up = box, score, best_score
            else:
                runner_up = max(runner_up, score)
        if best is None or best_score - runner_up < (.12 if lost else .06):
            return None
        return best / (width, height, width, height)
