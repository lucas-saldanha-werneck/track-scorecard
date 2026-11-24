import pytest

from track_scorecard.metrics import evaluate, iou, parse_mot
from track_scorecard.sim import greedy_iou_tracker, make_crossing, to_mot
from track_scorecard.cli import main


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_parse_roundtrip_and_comments():
    gt = make_crossing(5)
    assert parse_mot("# c\n\n" + to_mot(gt)).keys() == gt.keys()


@pytest.mark.parametrize("bad", ["1,2,3", "1,a,0,0,5,5", "1,1,0,0,0,5"])
def test_parse_rejects_bad_lines(bad):
    with pytest.raises(ValueError):
        parse_mot(bad)


def test_perfect_prediction():
    gt = make_crossing(20)
    s = evaluate(gt, gt)
    assert (s.false_negatives, s.false_positives, s.id_switches) == (0, 0, 0)
    assert s.mota == 1.0 and s.mean_iou == 1.0


def test_missing_frames_are_false_negatives_and_fragment():
    gt = {f: {1: (0, 0, 10, 10)} for f in range(1, 7)}
    pred = {f: {9: (0, 0, 10, 10)} for f in (1, 2, 5, 6)}
    s = evaluate(gt, pred)
    assert s.false_negatives == 2
    assert s.fragmentations == 1
    assert s.id_switches == 0
    assert s.mota == pytest.approx(1 - 2 / 6)


def test_extra_boxes_are_false_positives():
    gt = {1: {1: (0, 0, 10, 10)}}
    pred = {1: {1: (0, 0, 10, 10), 2: (100, 100, 10, 10)}}
    s = evaluate(gt, pred)
    assert s.false_positives == 1 and s.precision == 0.5


def test_id_switch_counted_once():
    gt = {f: {1: (0, 0, 10, 10)} for f in range(1, 5)}
    pred = {1: {5: (0, 0, 10, 10)}, 2: {5: (0, 0, 10, 10)},
            3: {6: (0, 0, 10, 10)}, 4: {6: (0, 0, 10, 10)}}
    assert evaluate(gt, pred).id_switches == 1


def test_naive_tracker_swaps_ids_when_objects_cross():
    gt = make_crossing()
    s = evaluate(gt, greedy_iou_tracker(gt))
    assert s.false_negatives == 0 and s.false_positives == 0
    assert s.id_switches >= 1
    assert s.mota < 1.0


def test_naive_tracker_ok_when_objects_never_meet():
    gt = {f: {1: (10 + f, 0, 20, 50), 2: (300 + f, 200, 20, 50)} for f in range(1, 30)}
    s = evaluate(gt, greedy_iou_tracker(gt))
    assert s.id_switches == 0 and s.mota == 1.0


def test_empty_input_does_not_crash():
    s = evaluate({}, {})
    assert s.gt_boxes == 0 and s.mota == 0.0


def test_cli_demo_and_score(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").exists()
    capsys.readouterr()
    assert main(["score", str(tmp_path / "gt.txt"), str(tmp_path / "pred.txt"), "--json"]) == 0
    assert '"id_switches"' in capsys.readouterr().out


def test_cli_bad_file(tmp_path, capsys):
    p = tmp_path / "x.txt"
    p.write_text("nope")
    assert main(["score", str(p), str(p)]) == 2
    assert "error" in capsys.readouterr().err
