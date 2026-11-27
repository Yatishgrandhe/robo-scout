# Refactor plan (from 22 research agents)

ROBO SCOUT REFACTOR PLAN (priority: missed shots, then wrong-robot credit, then missed Hive entries)

I checked three audit claims against the source and all three hold: the camera-cut reset (backend/launches.py:350-353), the projection time cap (backend/hive_entries.py:223, `min(gap, .27)`), and the pollen-only Hive scan (backend/hive_entries.py:134, 157, 243). I did not run the pipeline.

STEP 0. Evaluation harness (small; do this first, everything else is measured with it)
- Build: backend/eval_events.py. It matches predicted and ground-truth events one-to-one, per event type and per robot, within a time tolerance. Report precision, recall and F1 at 0.25, 0.5 and 1.0 s (SoccerNet protocol, arXiv 1804.04527). Also report per-robot shot and entry precision and recall, robot ID switches, and raw TP/FP/FN per match. Add bootstrap confidence intervals over matches.
- Split by whole match, never by frame or launch window. Keep 4 matches sealed for the final test and run leave-matches-out cross-validation on the other ~16.
- Add an oracle ablation (SoccerNet-GSR, arXiv 2404.11335): substitute ground-truth robot IDs, then ground-truth ball tracks, and see which recovers the most score.
- Verify: on the current held-out match, the harness reproduces the known failures (for example, 1 Hive entry found).

STEP 1. Missed shots
1a. Code fixes (small; source: shots audit). In backend/launches.py:
- Before the camera-cut reset clears `active` (350-353), run attributed() on each active trail.
- Raise the stale limit (320) from 0.075 s to about 0.25 s, and coast trails on a constant-velocity-plus-gravity prediction across the gap.
- Raise the 1.3 s trail cap.
- Fix the duplicate tolerance (253): use at least 1.5 frame periods instead of dt < 0.016.
- Relax the jitter stop in the upward run (159).
- Make the 960/1536 rescale (288-289) one fixed scale.
- In backend/shot_model.py: replace the hard SHOT_FLOOR filter (53-64) with a confidence score attached to the event. The floor was set on one match. Also make the loader fail loudly instead of returning None.
- Verify: shot recall on the cross-validation folds, and count shots recovered through cut frames.
1b. Ball recall (small). Run the YOLO26 ball model with Ultralytics `tile=True` (the docs mark SAHI deprecated in favour of it), then retrain with tile=True on the labeled frames.
- Test yolo26-p2.yaml, initialized from yolo26n.pt (no P2 weights are released).
- Sources: SAHI arXiv 2202.06934; Ultralytics tiling and YOLO26 docs.
- Verify: per-frame in-flight ball recall. The Roboflow shot tracker reports about 98% for basketballs, which is a target, not a guarantee. Also check MPS time per frame.
1c. Ballistic ball tracker (medium). Add a new module, backend/ball_tracks.py:
- ByteTrack-style two-stage association that keeps low-confidence boxes (arXiv 2110.06864).
- Mahalanobis gating instead of IoU.
- Confirm a track after M hits; end it after a set coast time.
- Link fragments when one RANSAC parabola explains both pieces (arXiv 1905.03633).
- This replaces the origin gate (launches.py:116-127), which rejects any trail that starts mid-flight. The launch point and time instead come from back-projecting the fitted parabola.
- Verify: shot recall and ball-track fragmentation per launch.
1d. Optional, only if 1b and 1c leave recall short (large): a TrackNet-style 3-frame heatmap model with frame-difference input (arXiv 1907.03698, 2409.14543). First benchmark against WASB-SBDT (github.com/nttcom/WASB-SBDT) on the 20 videos.

STEP 2. Wrong-robot attribution
2a. Global per-frame assignment (medium; source: tracking audit). In backend/vision.py:862-885:
- Build a team-by-detection cost matrix from classifier probability, IoU with the last trusted box, and appearance.
- Solve it with linear_sum_assignment, which is already imported at line 10 but unused.
- Make the classifier a hard veto on every path, including the named-detection branch (361-371) and the fallback scoring (321-322).
- Add NMS across the merged detector outputs (831-841).
- Roll back the full track state on collision (nano tracker, recovery, gallery, pending counter), not just tracker and supported.
- Stop adding unverified boxes to the identity gallery (286-287).
- Make a missing classifier raise a warning (757-767).
2b. Vote identity over a window (small). Decide each launch's robot by a confidence-weighted majority vote of the classifier over about ±0.5 s around the fitted launch point. Do not trust the long-run track ID. This is tracklet voting (arXiv 2504.06357) applied at the event.
- In backend/launches.py, replace the source_error margin drop (294-303, 180-183): assign the shot to the robot box containing the back-projected launch point. On ties, use the voted identity instead of dropping the shot.
- Dedupe and merge by origin fit, not by confidence (216-242).
2c. Tracker trial (small). Try BoT-SORT or Deep OC-SORT with with_reid on robot boxes via an Ultralytics tracker YAML (arXiv 2206.14651, 2302.11813). Keep whichever gives fewer ID switches.
- Verify 2a-2c: per-robot shot precision and recall, IDF1, and ID switches on held-out matches.
2d. Deferred (medium): PARSeq team-number OCR as an identity anchor (arXiv 2405.13896). First check that team numbers are legible at this resolution.

STEP 3. Missed Hive entries
3a. Code fixes (small). In backend/hive_entries.py:
- Use the full gap in the projection at 223, not min(gap, .27).
- Add red and blue Nectar to the colour fallback (81), the class filter (134, 157) and the launch filter (243). Without this, Nectar entries cannot be found at all.
- Put scan and launch timestamps on the same time base (177 vs pts).
- Surface the silent skips in backend/vision.py (camera_motion, the cell-mark count, the scan fallback) as warnings.
3b. Entry state machine (medium). Replace arrival() (24-36) with per-ball-track states: outside, then candidate, then confirmed.
- Accept an entry from below. Today's rule requires a descent.
- Confirm after N frames inside, with a max-absent grace window (supervision LineZone).
- Count each track at most once.
- Add a second positive signal: a ball that settles inside, detected with a slow background model.
- Credit the entry to the launch it is linked to through the Step 1c trajectory, not to the robot track present at entry time.
3c. Scoreboard check (medium). OCR the overlay ROI at about 0.2 s intervals. Require monotonic, plausible increments: 2 pts per ball, 20 per tip. Any increment with no explaining entry triggers a relaxed re-search of that window. This checks alliance totals, not who scored.
- Verify 3a-3c: entry precision and recall per robot. Also report the alliance-total error against the score overlay.

ONGOING (small): an active-learning review queue ranked by entropy of the shot and entry decisions (arXiv 2304.04220). Ask reviewers binary questions ("did it enter?", "which robot?"). Fine-tune existing weights each round rather than retraining from scratch.

UNSUPPORTED OR WEAK FINDINGS
- TrackNetV4/V5, TOTNet, DAM4SAM, SAMURAI and T-DEED gains are from abstracts or snippets on other sports. None were tested on balls this small (~10 px) at 60 fps.
- The basketball repos (chonyy, anishcd) and the "97%" README claim report no accuracy.
- Patents, the Noah vendor claim, and possession radii (1.5 m, 3 m, 50 px) are snippets only or tied to one setup.
- arXiv 2509.18451 has a unit inconsistency, so its numbers are only usable as relative comparisons.
- SAM 2 has no measured MPS video speed (one blog reports about 1.6 s per frame), and no source shows it on tiny balls. Not recommended.
- Conv3d on MPS is unverified, so do not plan around X3D or SlowFast without a smoke test.
- E2E-Spot and T-DEED expect CUDA, and 20 matches may be too few to train them. Defer.
- Homography on its own is invalid for the elevated Cell rim; that needs PnP.
- The OC-SORT and StrongSORT papers disagree on linear vs GSI interpolation, so test both.
- The robot-tracking research agent noted that it did not do the requested fan-out.
