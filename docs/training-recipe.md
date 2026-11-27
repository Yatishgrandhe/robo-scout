# Shot detector training recipe (from 10 research agents)

TRAINING RECIPE: FTC shot detection on Apple Silicon

STEP 0, ground truth (2-3 h). Pick 3 matches as a held-out set, ideally from a different event or venue, and never train on them. For every shot, a reviewer records the release frame, robot, ball color and whether it entered. This is the scoreboard. Per-frame mAP hides missed shots. In the basketball shot tracker, make/miss accuracy was 94% on shots it detected, and most of the end-to-end error came from shots it missed.

STEP 1, fix the event layer first (day 1, no training).
- Run ftc.pt at 960 px on every frame at conf 0.10-0.15 and keep every box.
- Run a second 1.5-2x upscaled pass on a crop around the basket and launch zones. The ETH shuttle paper found recall drops below about 20 px, and these balls are 10-15 px.
- Link boxes into tracklets with a gated nearest-neighbor tracker (constant velocity, gate grows with frame gap).
- Fit flights with sequential RANSAC. Fit x(t) as linear and y(t) as quadratic using frame timestamps, which handles occlusion gaps better than fitting y(x). Image y points down, so the t² coefficient must be positive.
- Accept a flight with at least 6 inliers over at least 10 frames, residual under about 3 px, and not near-straight.
- Release = earliest frame the fit supports. Robot = the robot box nearest the launch point at that frame, within a distance cutoff; otherwise flag it.
- Entry = the arc crosses the Hive Cell polygon. Send ambiguous cases to review.
- All thresholds are guesses. Tune them on the Step 0 matches, then freeze them.

STEP 2, bootstrap labels per tracklet (days 1-2).
- Show candidate tracklets on image sheets. The reviewer accepts, rejects or splits each one and fixes only bad boxes. Do not review frame by frame.
- Inside accepted flights, fill missed frames from the fitted parabola. This adds boxes exactly in the hard frames (behind the frame, motion blur).
- Frames from rejected tracklets become hard negatives (empty label files): robot parts, the basket frame, blur streaks.
- Target from about 12-15 training matches: 300-500 verified shot tracklets plus 1-2k corrected or hard-negative frames. This is an estimate. SPAM reports 3-20% of full labeling effort for this style of review.
- Do not auto-accept pseudo-labels at 0.85 confidence (the Ultralytics example value). That drops exactly the faint balls you need.

STEP 3, fine-tune YOLO (days 2-4).
- Settings: model=ftc.pt, device=mps, imgsz=960 (try 1280 as an ablation), batch=8, amp=False, workers=4, cache=ram, optimizer=SGD, lr0=0.005, epochs=100, patience=20. Set the optimizer explicitly, because auto ignores lr0. Compare freeze=0 against freeze=10.
- Smoke test with fraction=0.1 and time one epoch on MPS against CPU. Check that MPS inference matches CPU.
- Never set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; it can crash the machine.
- Split train and validation by match.
- Loop: re-run Step 1, re-review, retrain, for 2-3 rounds. Record held-out shot recall every round to catch confirmation bias.
- Optional: offline copy-paste of feathered ball sprites along synthetic parabolas in 20-30% of images. The Ultralytics copy_paste option needs segment labels. The benefit for this domain is unproven, so ablate it against a real-only run.

STEP 4, heatmap tracker (week 2, if recall plateaus).
- Choose a TrackNet/WASB-style heatmap model over YOLO with motion channels. The heatmap route is the documented one for tiny, fast, blurry balls. None of the reports gave evidence for YOLO with motion channels, so treat it as an untested idea.
- Data: turn verified tracklets into per-match CSVs (Frame, Visibility, X, Y), the TrackNetV3 format. Occluded frames get a zero heatmap. 300-500 tracklets give roughly 10-20k point-labeled frames, similar in scale to TrackNet tennis (about 18k + 16k frames) and WASB soccer (12k).
- Input: 3 stacked frames. Do not shrink the full frame to 512x288, which leaves the ball at 4-6 px. Use native-scale 640x360 crops around flight regions instead, with a Gaussian target of d = 2.5 px.
- Model: a small HRNet (about 1.5M parameters, no stem stride), trained from scratch with Adam for 30 epochs at batch 8, using quality focal loss or weighted BCE. Add hard-sample mining from epoch 20 (WASB's method). lr 1e-3 is my choice; the paper does not give one.
- Code: WASB's training code is listed as "TBA", so plan to write the trainer. TrackNetV3 needs MPS porting (device handling, no float64, PYTORCH_ENABLE_MPS_FALLBACK=1).
- Merge heatmap peaks with YOLO boxes and feed both into the same RANSAC linker.

EVALUATION (held-out matches only, thresholds frozen first). Metrics:
- shot recall and precision
- release-frame error (median, and the share of shots within ±3 frames)
- robot attribution accuracy
- entry accuracy
- for the heatmap model, per-frame F1 at 4 px tolerance

Report per ball color. Expect a drop on unseen venues: the ETH shuttle detector fell from F1 0.864 to 0.703 on new locations. Bootstrap a 95% confidence interval over shots.

WEAK OR UNVERIFIED CLAIMS
- Every label count for this domain is an estimate. No source gives a minimum.
- All gates, RANSAC tolerances, inlier counts and confidence floors are untuned starting points.
- There is no benchmark for MPS training speed. Community reports say it is slow with low GPU use, so measure it.
- WASB numbers (F1 82-96) come from broadcast sports and do not transfer to 10-15 px balls behind a translucent frame.
- The gains from copy-paste come from COCO and Mask R-CNN. The sports-ball paper on this (Sensors 2021) could not be read.
- The basketball shot-pipeline accuracies (78-92%) are self-reported GitHub results, not peer-reviewed.
- TrackNetV3's 87.7 to 97.5% gain from inpainting comes from a search snippet only.
- The 1500-images-per-class guidance is old YOLOv5 advice, not verified for YOLO26.
- That Step 1 gives most of the early gain is my assumption, based on detector recall (83%) not being the only bottleneck. Confirm it on Step 0 before investing in Step 4.
