import { useState } from "react";
import { BoxSelect, Crosshair, Play, Plus, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { scoringSetupIssues, zoneNames, timecode } from "@/types";
import type { Box, Match, Model, Options, RobotSeed, Setup, ZoneKind } from "@/types";
import { isAnchorMark, isTeamMark } from "@/player";
import type { Mark } from "@/player";

type Props = {
  setup: Setup; setSetup: (value: Setup) => void; match: Match; time: number;
  options: Options; drawing: Mark | null; arm: (value: Mark | null) => void;
  zoneAlliance: Setup["alliance"]; setZoneAlliance: (value: Setup["alliance"]) => void;
  mark: (box: Box) => void; run: () => void; working: boolean; hide: () => void;
};

export function Calibration({ setup, setSetup, match, time, options, drawing, arm, zoneAlliance, setZoneAlliance, mark, run, working, hide }: Props) {
  const [zone, setZone] = useState<ZoneKind>("cell");
  const [coords, setCoords] = useState("");
  const [issue, setIssue] = useState("");
  const scoringIssues = scoringSetupIssues(setup);
  const robots = setup.robots ?? [];
  const teams = [setup.team, ...robots.map(seed => seed.team)];
  const rosterReady = !robots.length || (new Set(teams).size === teams.length
    && teams.every(team => /^[0-9]{1,6}$/.test(team)) && robots.every(seed => seed.robot));
  const change = <K extends keyof Setup>(key: K, value: Setup[K]) => setSetup({ ...setup, [key]: value });
  const stamp = (key: "start_time" | "end_time") => change(key, Math.min(match.duration, Math.round(time * 1000) / 1000));
  const editRobot = (index: number, update: Partial<RobotSeed>) =>
    change("robots", robots.map((seed, i) => index === i ? { ...seed, ...update } : seed));

  function applyCoords() {
    const box = coords.trim().split(/[,\s]+/).map(Number);
    if (box.length !== 4 || !box.every(value => Number.isFinite(value) && value >= 0 && value <= 1) || box[2] - box[0] < .012 || box[3] - box[1] < .012) {
      setIssue("Enter four fractions from 0 to 1. Right and bottom must be larger than left and top.");
      return;
    }
    mark(box as Box);
    setIssue("");
    setCoords("");
  }

  return <section id="analysis-setup" className="setup-sheet" aria-labelledby="setup-heading">
    <div className="section-title">
      <div><h2 id="setup-heading">Set up analysis</h2><p>Pause where each robot is clear, then pair its box with its team number.</p></div>
      <Button variant="ghost" size="icon-sm" aria-label="Close setup" onClick={hide}><X /></Button>
    </div>
    <FieldGroup className="settings-grid">
      <Field><FieldLabel htmlFor="team">First team number</FieldLabel><Input id="team" value={setup.team} inputMode="numeric" maxLength={24} onChange={e => change("team", e.target.value)} placeholder="Optional for a single robot" /></Field>
      <Field><FieldLabel htmlFor="alliance">Target alliance</FieldLabel><select id="alliance" value={setup.alliance} onChange={e => { change("alliance", e.target.value as Setup["alliance"]); setZoneAlliance(e.target.value as Setup["alliance"]); }}><option value="red">Red</option><option value="blue">Blue</option></select></Field>
      <Field><FieldLabel htmlFor="model">Ball detector</FieldLabel><select id="model" value={setup.model} onChange={e => change("model", e.target.value as Model)}>
        <option value="generic" disabled={!options.models.generic}>YOLO + color (baseline)</option>
        <option value="synthetic" disabled={!options.models.synthetic}>Synthetic checkpoint</option>
        <option value="ftc" disabled={!options.models.ftc}>FTC-trained checkpoint</option>
      </select></Field>
      <Field><FieldLabel htmlFor="sampling">Sampling rate</FieldLabel><select id="sampling" value={setup.sample_fps} onChange={e => change("sample_fps", Number(e.target.value) as Setup["sample_fps"])}>
        <option value={6}>6 frames/s</option><option value={10}>10 frames/s</option><option value={15}>15 frames/s</option>
      </select></Field>
    </FieldGroup>
    <div className="roster-setup">
      <div className="section-title"><div><h3>Match roster</h3><p>Add the other teams to track all marked robots together.</p></div>
        <Button variant="outline" size="sm" disabled={robots.length >= 5} onClick={() =>
          change("robots", [...robots, { team: "", alliance: "blue", robot: null, seed_time: time }])}>
          <Plus data-icon="inline-start" /> Add robot
        </Button>
      </div>
      {robots.map((seed, i) => <div className="roster-entry" key={i}>
        <Field><FieldLabel htmlFor={`roster-team-${i}`}>Team number</FieldLabel>
          <Input id={`roster-team-${i}`} value={seed.team} inputMode="numeric" maxLength={6}
            onChange={event => editRobot(i, { team: event.target.value.trim() })} /></Field>
        <Field><FieldLabel htmlFor={`roster-alliance-${i}`}>Alliance</FieldLabel>
          <select id={`roster-alliance-${i}`} value={seed.alliance} onChange={event => editRobot(i, { alliance: event.target.value as Setup["alliance"] })}>
            <option value="red">Red</option><option value="blue">Blue</option>
          </select></Field>
        <Button size="sm" variant={drawing === `team:${i}` ? "default" : "outline"}
          onClick={() => arm(drawing === `team:${i}` ? null : `team:${i}`)}>
          <Crosshair data-icon="inline-start" /> {seed.robot ? `Marked ${timecode(seed.seed_time)}` : "Draw robot box"}
        </Button>
        <Button size="sm" variant={drawing === `anchor:${i + 1}` ? "default" : "ghost"}
          disabled={!seed.robot || time <= seed.seed_time + .1 || (setup.anchors?.length ?? 0) >= 128}
          onClick={() => arm(drawing === `anchor:${i + 1}` ? null : `anchor:${i + 1}`)}>Add keyframe</Button>
        <Button variant="ghost" size="icon-sm" aria-label={`Remove team ${seed.team || i + 2}`}
          onClick={() => { arm(null); change("robots", robots.filter((_, index) => i !== index)); }}><X /></Button>
      </div>)}
      {!rosterReady && <p role="status">Enter a different team number and draw a box for each robot.</p>}
    </div>
    <div className="calibration-steps">
      <div>
        <h3>Robot</h3>
        <p>{setup.robot ? `Marked at ${timecode(setup.seed_time, true)}` : "Mark the target robot to lock its identity."}</p>
        <div className="stamp-row">
          <Button variant={drawing === "robot" ? "default" : "outline"} size="sm" onClick={() => arm(drawing === "robot" ? null : "robot")}><Crosshair data-icon="inline-start" /> {setup.robot ? "Redraw robot" : "Draw robot box"}</Button>
          {setup.robot && <Button size="sm" variant={drawing === "anchor:0" ? "default" : "ghost"}
            disabled={time <= setup.seed_time + .1 || (setup.anchors?.length ?? 0) >= 128}
            onClick={() => arm(drawing === "anchor:0" ? null : "anchor:0")}>Add keyframe</Button>}
        </div>
      </div>
      <div>
        <h3>Scoring zones</h3><p>Robo Scout looks for visible Hive Cells. You can adjust the zones or add Flowers and Gardens.</p>
        <div className="zone-controls">
          <select aria-label="Zone type" value={zone} onChange={e => setZone(e.target.value as ZoneKind)}>{Object.entries(zoneNames).map(([kind, label]) => <option key={kind} value={kind}>{label}</option>)}</select>
          <select aria-label="Zone alliance" value={zoneAlliance} onChange={e => setZoneAlliance(e.target.value as Setup["alliance"])}><option value="red">Red</option><option value="blue">Blue</option></select>
          <Button variant="outline" size="sm" disabled={setup.zones.length >= 32} onClick={() => arm(drawing === zone ? null : zone)}><BoxSelect data-icon="inline-start" /> Draw zone</Button>
        </div>
        {setup.zones.length > 0 && <ul className="zone-list">{setup.zones.map((item, i) => <li key={i}>
          {zoneNames[item.kind]} · {item.alliance}{item.automatic ? " · detected" : ""}
          <Button variant="ghost" size="icon-xs" aria-label={`Remove ${item.alliance} ${zoneNames[item.kind]} ${i + 1}`}
            onClick={() => change("zones", setup.zones.filter((_, index) => i !== index))}><X /></Button>
        </li>)}</ul>}
      </div>
      <div>
        <h3>Match clock</h3><p>Seek to the start horn, then to the settled field after the match.</p>
        <div className="stamp-row">
          <Button variant="outline" size="sm" onClick={() => stamp("start_time")}>{setup.start_time === null ? "Set start here" : `Start ${timecode(setup.start_time, true)}`}</Button>
          <Button variant="outline" size="sm" onClick={() => stamp("end_time")}>{setup.end_time === null ? "Set settled end here" : `End ${timecode(setup.end_time, true)}`}</Button>
          {(setup.start_time !== null || setup.end_time !== null) && <Button variant="ghost" size="icon-sm" aria-label="Clear match clock" onClick={() => setSetup({ ...setup, start_time: null, end_time: null })}><RotateCcw /></Button>}
        </div>
      </div>
    </div>
    {drawing && <div className="coordinate-editor">
      <p role="status">Click two opposite corners in the footage to mark the {drawing === "robot" || isTeamMark(drawing) || isAnchorMark(drawing) ? "robot" : drawing === "field" ? "field and shot area" : zoneNames[drawing]}.</p>
      <Field>
        <FieldLabel htmlFor="coordinates">Or enter left, top, right, bottom</FieldLabel>
        <div className="stamp-row"><Input id="coordinates" value={coords} onChange={e => setCoords(e.target.value)} placeholder="0.25, 0.35, 0.42, 0.55" /><Button size="sm" variant="outline" onClick={applyCoords}>Apply</Button></div>
        {issue && <FieldDescription role="alert">{issue}</FieldDescription>}
      </Field>
      <Button variant="ghost" size="sm" onClick={() => arm(null)}>Cancel drawing</Button>
    </div>}
    <div className="setup-footer">
      <label className="check-label"><input type="checkbox" checked={setup.sam} disabled={!options.sam} onChange={e => change("sam", e.target.checked)} /> Refine robot outline with SAM 2.1</label>
      <label className="check-label"><input type="checkbox" checked={!!setup.camera_motion} onChange={e => change("camera_motion", e.target.checked)} /> Follow camera motion</label>
      <Button variant="outline" size="sm" onClick={() => arm(drawing === "field" ? null : "field")}><BoxSelect data-icon="inline-start" /> {setup.field ? "Redraw field crop" : "Draw field crop"}</Button>
      <Button disabled={working || !setup.robot || !rosterReady || !options.models[setup.model]} onClick={run}><Play data-icon="inline-start" /> {robots.length ? "Analyze marked robots" : "Analyze match"}</Button>
    </div>
    {scoringIssues.length > 0 && <p className="calibration-status">When no zones are marked, analysis tries to find the Hive Cells automatically.</p>}
  </section>;
}
