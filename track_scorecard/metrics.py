"""CLEAR MOT style scoring for multi-object tracking.

Input boxes are (x, y, w, h). A track file is a dict: frame -> {track_id: box}.
Matching per frame: keep last frame's pairs while IoU >= threshold, then match
the rest greedily by highest IoU. Greedy matching is not optimal (the official
tools use the Hungarian method), so numbers can differ slightly from them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

Box = tuple[float, float, float, float]
Frames = dict[int, dict[int, Box]]


def iou(a: Box, b: Box) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def parse_mot(text: str) -> Frames:
    """Parse MOTChallenge text: frame,id,x,y,w,h[,conf,...]. Blank/# lines skipped."""
    frames: Frames = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"line {n}: need at least 6 fields, got {len(parts)}")
        try:
            f, tid = int(parts[0]), int(parts[1])
            x, y, w, h = (float(p) for p in parts[2:6])
        except ValueError as e:
            raise ValueError(f"line {n}: {e}") from None
        if w <= 0 or h <= 0:
            raise ValueError(f"line {n}: box width and height must be positive")
        frames.setdefault(f, {})[tid] = (x, y, w, h)
    return frames


@dataclass
class Score:
    frames: int
    gt_boxes: int
    false_negatives: int
    false_positives: int
    id_switches: int
    fragmentations: int
    matches: int
    mean_iou: float
    mota: float
    recall: float
    precision: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(gt: Frames, pred: Frames, thr: float = 0.5) -> Score:
    all_frames = sorted(set(gt) | set(pred))
    prev: dict[int, int] = {}          # gt id -> pred id, last frame matched
    last_pred_for_gt: dict[int, int] = {}  # gt id -> pred id, ever
    tracked_last: dict[int, bool] = {}  # gt id -> was matched in its previous frame
    fn = fp = idsw = frag = matches = n_gt = 0
    iou_sum = 0.0
    n_pred = 0
    for f in all_frames:
        g = gt.get(f, {})
        p = pred.get(f, {})
        n_gt += len(g)
        n_pred += len(p)
        cur: dict[int, int] = {}
        used_p: set[int] = set()
        # 1) keep previous pairs that still overlap
        for gid, pid in prev.items():
            if gid in g and pid in p and iou(g[gid], p[pid]) >= thr:
                cur[gid] = pid
                used_p.add(pid)
        # 2) greedy by highest IoU for the rest
        cands = sorted(
            ((iou(gb, pb), gid, pid)
             for gid, gb in g.items() if gid not in cur
             for pid, pb in p.items() if pid not in used_p),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        for v, gid, pid in cands:
            if v < thr:
                break
            if gid in cur or pid in used_p:
                continue
            cur[gid] = pid
            used_p.add(pid)
        for gid, pid in cur.items():
            if gid in last_pred_for_gt and last_pred_for_gt[gid] != pid:
                idsw += 1
            last_pred_for_gt[gid] = pid
            iou_sum += iou(g[gid], p[pid])
        matches += len(cur)
        fn += len(g) - len(cur)
        fp += len(p) - len(cur)
        for gid in g:
            now = gid in cur
            if tracked_last.get(gid) is True and not now:
                frag += 1  # a tracked object lost; counts each break once
            tracked_last[gid] = now
        # an object absent from gt this frame resets its state
        for gid in list(tracked_last):
            if gid not in g:
                tracked_last.pop(gid)
        prev = cur
    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    return Score(
        frames=len(all_frames), gt_boxes=n_gt, false_negatives=fn,
        false_positives=fp, id_switches=idsw, fragmentations=frag,
        matches=matches,
        mean_iou=iou_sum / matches if matches else 0.0,
        mota=mota,
        recall=matches / n_gt if n_gt else 0.0,
        precision=matches / n_pred if n_pred else 0.0,
    )
