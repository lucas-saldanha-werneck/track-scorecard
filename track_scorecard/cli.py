from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import evaluate, parse_mot, Score
from .sim import greedy_iou_tracker, make_crossing, to_mot


def render(s: Score, title: str) -> str:
    return "\n".join([
        f"# {title}",
        "",
        f"- GT boxes: {s.gt_boxes} over {s.frames} frames",
        f"- Matches: {s.matches} (mean IoU {s.mean_iou:.3f})",
        f"- Missed (FN): {s.false_negatives}",
        f"- Extra (FP): {s.false_positives}",
        f"- ID switches: {s.id_switches}",
        f"- Fragmentations: {s.fragmentations}",
        f"- Recall {s.recall:.3f}, precision {s.precision:.3f}",
        f"- MOTA: {s.mota:.3f}",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="track-scorecard",
                                 description="Score tracker output against ground truth (MOTChallenge txt).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score a prediction file against a ground-truth file")
    sc.add_argument("gt")
    sc.add_argument("pred")
    sc.add_argument("--iou", type=float, default=0.5)
    sc.add_argument("--json", action="store_true")
    dm = sub.add_parser("demo", help="run the invented crossing scenario with a naive tracker")
    dm.add_argument("--out", help="folder to write gt.txt, pred.txt, summary.md")
    dm.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "score":
        try:
            gt = parse_mot(Path(a.gt).read_text())
            pred = parse_mot(Path(a.pred).read_text())
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        s = evaluate(gt, pred, a.iou)
        title = f"Score: {a.pred} vs {a.gt}"
    else:
        gt = make_crossing()
        pred = greedy_iou_tracker(gt)
        s = evaluate(gt, pred)
        title = "Demo: naive IoU tracker on an invented crossing scene"
        if a.out:
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "gt.txt").write_text(to_mot(gt))
            (out / "pred.txt").write_text(to_mot(pred))
            (out / "summary.md").write_text(render(s, title))
    print(json.dumps(s.as_dict(), indent=2) if a.json else render(s, title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
