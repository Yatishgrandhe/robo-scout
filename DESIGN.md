# Robo Scout design

## 0. Research log

- Considered Apple’s immediate control feedback, the Aside product reference’s compact framing, and the Luna agents’ film room and pit-wall concepts. Selected the film desk structure with a video-first workspace, a paper ledger, and technical marks confined to the footage and timecode.
- Lazyweb searches for video timeline review and sports footage returned Vimeo, Vidyard, and Descript screens. Reviewed all three for a large footage viewport, transport grouped below the frame, and an adjacent evidence list.
- The supplied Robo Knights mark sets the dark palette and hot-pink accent. The film desk keeps the existing workflow and uses the mark in the masthead and browser tab.

## 1. Atmosphere

The desk feels like a dark film room for Robo Knights. A bright knight mark anchors the masthead, ruled panels frame the evidence, and hot pink marks the primary action and selected team. Preserve the footage's own colors and a large playback surface.

## 2. Color

Graphite #111016, panel #1b1820, ink #f7eff4, muted ink #bdb0bd, border #3c3441, brand pink #ff5c91, dark pink selection #3b2030, and near-black film #0c0a0f. Pink marks playback and the active scrubber. Amber #f3cb69 marks calibration, single-robot tracks, and Hive entries. Coral #ff9a94 marks fouls and review warnings. Red #fa7187 and blue #78baff identify the alliances beside written team numbers.

## 3. Typography

System UI for titles, labels, and controls. Use medium-heavy headings with tight tracking and tabular figures for the ledger. System monospace carries clock, rule IDs, and team numbers. Scale body text at 14 px and compact metadata at 12 px.

## 4. Spacing and layout

Four-pixel spacing basis. Main content maxes out at 1400 px, with 24 px between the playback and 352 px event log. The header and score ledger span the workspace. Below 1000 px, the event log follows playback. Below 640 px, sections and primary actions wrap naturally.

## 5. Primitives

shadcn Button, Badge, Dialog, Field, Input, Progress, and Empty share the graphite and pink theme. Buttons show active, hover, focus, disabled, and waiting states. Event rows carry time, title, and review status. The player pairs a video with a seek slider, event marks, frame overlays, and transport controls. The review panel exposes rule, evidence, and explicit confirm and dismiss actions.

The player header carries the Robo Knights crest, a compact footage label, and source and resolution metadata. A pink rim and a muted pink wash frame the header. The transport uses a pink play button and scrubber, with distinct markers for comments, Hive entries, and fouls. Keep the crest and the transport outside the recorded picture.

The robot ledger puts Pollen and Nectar in rows, with shots, successful Hive entries, and final pieces in separate columns. Penalty points and Nectar pickup checks sit below the table. Unavailable counts use a dash, and suggestions stay separate from confirmed counts. Selecting a launch trajectory shows its path over the footage. Portrait footage is centered and height-limited so the transport and launch controls remain reachable.
For a match roster, team controls use written team numbers and alliance colors. Selecting a team changes its event log and robot stats while all marked tracks stay on the video. The defense estimate shows the visible time behind each rating and offers jumps to candidate pressure intervals.
The match library uses a graphite dialog with a muted folder rail and a ruled list of recording thumbnails, names, and durations. Dark pink marks the selected folder. New folder stays visible in the header, with rename and move beside the affected item. Edits happen inline with a labeled field and explicit save and cancel actions. On small screens, folder navigation sits above the recording list.

## 6. Motion

Use immediate press feedback and a short color transition. Timeline scrubbing tracks the input position. Media and overlay motion follows the playhead. Reduced-motion preferences suppress motion on controls and overlays.

## 7. Surface

The background is matte graphite. Use thin rules to group controls and ledger entries. The dark video has a small functional overlay for track labels and marked zones. The dialog alone casts an elevated shadow.

## 8. Accessibility and constraints

All transport and review actions are keyboard-operable. Calibration supports either clicking two corners or entering normalized coordinates. Overlay information is repeated in text. Use tabular timecode, visible focus, and responsive reflow. A camera cut or a hidden robot can make attribution uncertain. Keep suggestions separate from reviewed totals.
