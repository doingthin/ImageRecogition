#!/usr/bin/env python3
"""Real-time object detection demo using motion or color-based segmentation."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class TrackerState:
    tracker: Optional[cv2.Tracker] = None
    last_box: Optional[Box] = None
    smoothed_box: Optional[Box] = None
    lost_frames: int = 0
    stable_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect moving or color-based objects in a webcam stream."
    )

    # NOTE: We are hardcoding the external webcam (commonly index 1 on Windows),
    # so we DO NOT expose --device as an argument anymore.
    # If you ever need to change it, edit EXTERNAL_WEBCAM_INDEX below.
    parser.add_argument(
        "--method",
        choices=["motion", "color"],
        default="motion",
        help="Detection method",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1500,
        help="Minimum contour area to consider as object",
    )
    parser.add_argument(
        "--quit-key",
        type=str,
        default="q",
        help="Key to press to quit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show FPS and box coordinates",
    )
    parser.add_argument(
        "--hsv-lower",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=[35, 60, 60],
        help="HSV lower bound for color mode",
    )
    parser.add_argument(
        "--hsv-upper",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=[85, 255, 255],
        help="HSV upper bound for color mode",
    )
    parser.add_argument(
        "--tracker",
        choices=["csrt", "kcf", "mosse"],
        default="csrt",
        help="Tracker type for lock-on",
    )
    parser.add_argument(
        "--reinit-frames",
        type=int,
        default=10,
        help="Frames to wait before re-detection when tracker is lost",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="Exponential smoothing factor for box stabilization",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Consecutive frames required before switching targets",
    )
    return parser.parse_args()


def create_tracker(tracker_type: str) -> cv2.Tracker:
    if tracker_type == "kcf":
        return cv2.TrackerKCF_create()
    if tracker_type == "mosse":
        return cv2.TrackerMOSSE_create()
    return cv2.TrackerCSRT_create()


def find_candidates(mask: np.ndarray, min_area: int) -> list[Box]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((x, y, w, h))
    return candidates


def score_candidate(
    box: Box,
    frame_shape: Tuple[int, int],
    last_box: Optional[Box],
) -> float:
    x, y, w, h = box
    area = w * h
    frame_h, frame_w = frame_shape

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    frame_center_x = frame_w / 2.0
    frame_center_y = frame_h / 2.0

    center_dist = np.hypot(center_x - frame_center_x, center_y - frame_center_y)
    center_score = 1.0 - min(center_dist / max(frame_w, frame_h), 1.0)

    aspect_ratio = w / max(h, 1)
    aspect_penalty = 0.0
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        aspect_penalty = 0.3

    continuity_bonus = 0.0
    if last_box:
        last_x, last_y, last_w, last_h = last_box
        last_center_x = last_x + last_w / 2.0
        last_center_y = last_y + last_h / 2.0
        dist = np.hypot(center_x - last_center_x, center_y - last_center_y)
        max_dist = max(frame_w, frame_h)
        continuity_bonus = 1.0 - min(dist / max_dist, 1.0)

    area_score = min(area / (frame_w * frame_h), 0.5) * 2.0

    return area_score + (0.6 * center_score) + (0.8 * continuity_bonus) - aspect_penalty


def pick_best_candidate(
    candidates: list[Box],
    frame_shape: Tuple[int, int],
    last_box: Optional[Box],
    stable_frames: int,
    previous_best: Optional[Box],
) -> Optional[Box]:
    if not candidates:
        return None

    scored = [
        (score_candidate(box, frame_shape, last_box), box) for box in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_box = scored[0]

    if previous_best and stable_frames > 0:
        prev_score = score_candidate(previous_best, frame_shape, last_box)
        if prev_score >= best_score * 0.9:
            return previous_best

    return best_box


def smooth_box(box: Box, previous: Optional[Box], alpha: float) -> Box:
    if previous is None:
        return box
    x, y, w, h = box
    px, py, pw, ph = previous
    smoothed = (
        int(px * alpha + x * (1 - alpha)),
        int(py * alpha + y * (1 - alpha)),
        int(pw * alpha + w * (1 - alpha)),
        int(ph * alpha + h * (1 - alpha)),
    )
    return smoothed


def detect_motion_candidates(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    min_area: int,
) -> list[Box]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    fg_mask = subtractor.apply(gray)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, None, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    return find_candidates(fg_mask, min_area)


def detect_color_candidates(
    frame: np.ndarray, lower: Tuple[int, int, int], upper: Tuple[int, int, int], min_area: int
) -> list[Box]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return find_candidates(mask, min_area)


def annotate_frame(
    frame: np.ndarray,
    box: Optional[Box],
    debug: bool,
    fps: float,
    status: str,
) -> Optional[np.ndarray]:
    if box:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y : y + h, x : x + w]
        if roi.size:
            _ = cv2.resize(roi, (224, 224))
        if debug:
            cv2.putText(
                frame,
                f"Box: x={x} y={y} w={w} h={h}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    else:
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if debug:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return frame


def update_tracker(
    frame: np.ndarray,
    tracker_state: TrackerState,
    smooth_alpha: float,
) -> Optional[Box]:
    if tracker_state.tracker is None:
        return None

    ok, tracked = tracker_state.tracker.update(frame)
    if not ok:
        tracker_state.lost_frames += 1
        return None

    x, y, w, h = [int(v) for v in tracked]
    tracker_state.lost_frames = 0
    tracker_state.last_box = (x, y, w, h)
    tracker_state.smoothed_box = smooth_box(
        tracker_state.last_box, tracker_state.smoothed_box, smooth_alpha
    )
    return tracker_state.smoothed_box


def initialize_tracker(
    frame: np.ndarray,
    box: Box,
    tracker_type: str,
    tracker_state: TrackerState,
) -> None:
    tracker_state.tracker = create_tracker(tracker_type)
    tracker_state.tracker.init(frame, box)
    tracker_state.last_box = box
    tracker_state.smoothed_box = box
    tracker_state.lost_frames = 0


def main() -> None:
    args = parse_args()

    # ===== HARD-CODED EXTERNAL WEBCAM SETTINGS =====
    # On Windows, external USB webcams are very often index 1.
    EXTERNAL_WEBCAM_INDEX = 1

    # Use DirectShow backend on Windows for more reliable camera selection.
    cap = cv2.VideoCapture(EXTERNAL_WEBCAM_INDEX, cv2.CAP_DSHOW)
    # ==============================================

    if not cap.isOpened():
        raise SystemExit(
            f"Unable to open external webcam (index {EXTERNAL_WEBCAM_INDEX}). "
            "If your external cam is a different index, change EXTERNAL_WEBCAM_INDEX "
            "to 0, 2, 3, etc."
        )

    subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=25)
    subtractor.setDetectShadows(False)
    tracker_state = TrackerState()

    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            box = None
            if tracker_state.tracker:
                box = update_tracker(frame, tracker_state, args.smooth_alpha)

            if box is None and (
                tracker_state.tracker is None
                or tracker_state.lost_frames >= args.reinit_frames
            ):
                if args.method == "motion":
                    candidates = detect_motion_candidates(
                        frame, subtractor, args.min_area
                    )
                else:
                    candidates = detect_color_candidates(
                        frame, tuple(args.hsv_lower), tuple(args.hsv_upper), args.min_area
                    )

                candidate_box = pick_best_candidate(
                    candidates,
                    (frame.shape[0], frame.shape[1]),
                    tracker_state.last_box,
                    tracker_state.stable_frames,
                    tracker_state.last_box,
                )

                if candidate_box:
                    tracker_state.stable_frames += 1
                else:
                    tracker_state.stable_frames = 0

                if candidate_box and tracker_state.stable_frames >= args.stable_frames:
                    initialize_tracker(frame, candidate_box, args.tracker, tracker_state)
                    tracker_state.stable_frames = 0

                box = tracker_state.smoothed_box

            if tracker_state.tracker and box:
                status = "Tracking"
            elif tracker_state.tracker and not box:
                status = "Target lost"
            else:
                status = "No object detected"

            now = time.time()
            delta = now - last_time
            if delta > 0:
                fps = 1.0 / delta
            last_time = now

            annotate_frame(frame, box, args.debug, fps, status)
            cv2.putText(
                frame,
                "Press 'r' to select ROI",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Object Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                roi = cv2.selectROI("Object Detection", frame, fromCenter=False)
                if roi and roi[2] > 0 and roi[3] > 0:
                    initialize_tracker(frame, roi, args.tracker, tracker_state)
            elif key == ord(args.quit_key):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
