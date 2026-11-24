# track-scorecard

Independent demo, not affiliated with Marvik.

A small Python tool that scores the output of a multi-object tracker against ground truth. It reads the MOTChallenge text format (`frame,id,x,y,w,h,...`) and prints misses, extra boxes, ID switches, fragmentations and MOTA. It uses only the standard library.

I made it after reading a job post about object trackers on edge hardware. Detection accuracy gets most of the attention, but a tracker can have perfect detections and still swap two people's IDs when they cross. This tool counts that.

## Run it

```
pip install -e '.[dev]'
track-scorecard demo --out out        # invented example scene
track-scorecard score gt.txt pred.txt # your own files
pytest -q
```

Options: `--iou 0.5` sets the match threshold, `--json` prints raw numbers.

## What the demo does

It builds an invented scene: two people walk toward each other along the same row and cross. The "detections" are the exact ground-truth boxes. A deliberately simple tracker (greedy IoU, no motion model) assigns IDs. Then the scorer compares the result with the truth.

Result on the invented scene:

```
GT boxes: 120 over 60 frames, 120 matches, 0 missed, 0 extra
ID switches: 2
MOTA: 0.983
```

The boxes are all correct, yet the tracker swapped the two IDs at the crossing, once for each person. The data is made up and the scene is easy, so this shows that the scorer catches the failure. It says nothing about any real tracker.

## What was tested

- 14 offline tests with pytest, run in CI on every push (`.github/workflows/test.yml`).
- IoU math, file parsing and bad-line errors.
- A perfect prediction scores MOTA 1.0.
- Dropped frames count as misses and as one fragmentation.
- Extra boxes count as false positives.
- One ID change counts as one ID switch.
- The naive tracker swaps IDs in the crossing scene and does not swap them when the objects never meet.
- Empty input, and the two CLI commands including a bad file.

## What was not tested

- No real video, no real tracker output, no public benchmark data.
- I did not compare the numbers with the official MOTChallenge tools (TrackEval). This tool matches boxes greedily, not with the Hungarian method, so numbers can differ a little from the official ones on crowded scenes.
- No IDF1 or HOTA, which many benchmarks now use.
- No performance test on long sequences.
- The MOTChallenge columns after `h` (confidence, class, visibility) are ignored.
