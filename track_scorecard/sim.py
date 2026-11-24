"""Invented example data: two people walk toward each other and cross.

This is made-up data, not a real video. A very simple greedy IoU tracker is
included so the demo has something to score.
"""
from __future__ import annotations

from .metrics import Box, Frames, iou

W, H = 40.0, 100.0


def make_crossing(n_frames: int = 60) -> Frames:
    """GT: object 1 moves left->right, object 2 right->left, same row."""
    gt: Frames = {}
    for f in range(1, n_frames + 1):
        t = (f - 1) / (n_frames - 1)
        x1 = 20 + t * 360
        x2 = 380 - t * 360
        gt[f] = {1: (x1, 50.0, W, H), 2: (x2, 52.0, W, H)}
    return gt


def greedy_iou_tracker(dets: Frames, min_iou: float = 0.1) -> Frames:
    """Assign each detection to the previous track with the highest IoU.

    Tracks do not use motion prediction, so when two boxes overlap the
    assignment can swap. That weakness is the point of the demo.
    """
    out: Frames = {}
    tracks: dict[int, Box] = {}
    next_id = 1
    for f in sorted(dets):
        boxes = list(dets[f].values())
        pairs = sorted(
            ((iou(tb, b), tid, i)
             for tid, tb in tracks.items() for i, b in enumerate(boxes)),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        assign: dict[int, int] = {}
        for v, tid, i in pairs:
            if v < min_iou:
                break
            if tid in used_t or i in used_b:
                continue
            used_t.add(tid)
            used_b.add(i)
            assign[i] = tid
        for i in range(len(boxes)):
            if i not in assign:
                assign[i] = next_id
                next_id += 1
        tracks = {tid: boxes[i] for i, tid in assign.items()}
        out[f] = {tid: boxes[i] for i, tid in assign.items()}
    return out


def to_mot(frames: Frames) -> str:
    rows = []
    for f in sorted(frames):
        for tid, (x, y, w, h) in sorted(frames[f].items()):
            rows.append(f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    return "\n".join(rows) + "\n"
