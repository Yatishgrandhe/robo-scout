import { useEffect, useRef, useState } from "react";
import type { CSSProperties, PointerEvent, RefObject } from "react";
import { Crosshair, Eye, EyeOff, Maximize, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { eventTitle, isReviewEvent, timecode } from "@/types";
import type { Box, EventRow, Frame, Match, Setup, ZoneKind } from "@/types";

export type Mark = "robot" | "field" | ZoneKind | `team:${number}` | `anchor:${number}`;
export const isTeamMark = (mark: Mark): mark is `team:${number}` => mark.startsWith("team:");
export const isAnchorMark = (mark: Mark): mark is `anchor:${number}` => mark.startsWith("anchor:");
type Props = {
  match: Match; frames: Frame[]; time: number; setTime: (time: number) => void;
  video: RefObject<HTMLVideoElement | null>; jump: (time: number) => void; select: (event: EventRow) => void;
  setup: Setup; drawing: Mark | null; mark: (box: Box) => void; editing: boolean;
  selected?: EventRow;
};

function frameAt(frames: Frame[], time: number) {
  let start = 0;
  let end = frames.length;
  while (start < end) {
    const mid = (start + end) >>> 1;
    if (frames[mid].time <= time + .022) start = mid + 1;
    else end = mid;
  }
  const frame = frames[start - 1];
  return frame && time - frame.time < .27 ? frame : undefined;
}

function overlayRect(box: Box, key: string, className: string) {
  return <rect key={key} x={box[0] * 1000} y={box[1] * 1000} width={(box[2] - box[0]) * 1000}
    height={(box[3] - box[1]) * 1000} vectorEffect="non-scaling-stroke" className={className} />;
}

export function Player({ match, frames, time, setTime, video, jump, select, setup, drawing, mark, editing, selected }: Props) {
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState(true);
  const [rate, setRate] = useState(1);
  const [anchor, setAnchor] = useState<[number, number] | null>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const [mediaError, setMediaError] = useState(false);
  const stage = useRef<HTMLDivElement>(null);
  const frame = frameAt(frames, time);
  const roster = [setup, ...(setup.robots ?? [])];
  const robots = roster.map(seed => {
    const keyframe = editing && setup.anchors?.find(anchor => anchor.team === seed.team && Math.abs(time - anchor.time) < .15);
    return { ...seed, box: keyframe ? keyframe.box : (editing ? null : frame?.robots?.[seed.team] ??
      (seed.team === setup.team ? frame?.robot : null)) ?? (editing && Math.abs(time - seed.seed_time) < .3 ? seed.robot : null) };
  });
  const shot = selected?.trajectory && Math.abs(time - selected.time) < 1.1 ? selected.trajectory : null;

  useEffect(() => {
    if (!playing) return;
    let animation = 0;
    let last = 0;
    const tick = (now: number) => {
      if (video.current && now - last >= 50) {
        setTime(video.current.currentTime);
        last = now;
      }
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [playing, setTime, video]);

  function toggle() {
    const el = video.current;
    if (!el) return;
    if (el.paused) void el.play().catch(() => setMediaError(true));
    else el.pause();
  }

  function coords(e: PointerEvent<HTMLDivElement>): [number, number] {
    const bounds = e.currentTarget.getBoundingClientRect();
    const clamp = (n: number) => Math.max(0, Math.min(1, n));
    return [clamp((e.clientX - bounds.left) / bounds.width), clamp((e.clientY - bounds.top) / bounds.height)];
  }

  function place(e: PointerEvent<HTMLDivElement>) {
    if (!drawing || e.button !== 0) return;
    const p = coords(e);
    video.current?.pause();
    if (!anchor) { setAnchor(p); setCursor(p); return; }
    const box: Box = [Math.min(p[0], anchor[0]), Math.min(p[1], anchor[1]), Math.max(p[0], anchor[0]), Math.max(p[1], anchor[1])];
    if (Math.min(box[2] - box[0], box[3] - box[1]) >= .012) mark(box);
    setAnchor(null);
    setCursor(null);
  }

  const preview: Box | null = anchor && cursor ? [
    Math.min(anchor[0], cursor[0]), Math.min(anchor[1], cursor[1]),
    Math.max(anchor[0], cursor[0]), Math.max(anchor[1], cursor[1]),
  ] : null;
  const clockPosition = Math.max(0, Math.min(100, 100 * time / match.duration));

  return <section className="player" aria-label="Match footage">
    <div className="player-header">
      <div><span className="rec-dot" /> MATCH FOOTAGE <span className="player-separator">/</span> {match.demo ? "PRACTICE SIMULATION" : "LOCAL VIDEO"}</div>
      <span>{match.width} × {match.height}</span>
    </div>
    <div ref={stage} className="stage" data-drawing={!!drawing} data-portrait={match.height > match.width}
      style={{ aspectRatio: match.width / match.height, "--portrait-width": `${72 * match.width / match.height}vh` } as CSSProperties}
      onPointerDown={place} onPointerMove={e => drawing && anchor && setCursor(coords(e))}>
      <video ref={video} preload="metadata" playsInline src={match.video} poster={match.poster}
        aria-label="Match recording" onPlay={() => setPlaying(true)} onPause={e => {
          setPlaying(false); setTime(e.currentTarget.currentTime);
        }}
        onEnded={e => { setPlaying(false); setTime(e.currentTarget.currentTime); }}
        onSeeked={e => setTime(e.currentTarget.currentTime)}
        onError={() => setMediaError(true)} />
      {(overlay || drawing) && <svg className="footage-layer" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
        {(editing || !match.demo) && setup.zones.map((zone, i) => overlayRect(
          editing ? zone.box : frame?.zones?.[i] ?? zone.box, `zone-${i}`, `zone-box zone-${zone.kind}`))}
        {editing && setup.field && overlayRect(setup.field, "field", "zone-box zone-field")}
        {frame?.mask && <polygon points={frame.mask.map(([x,y]) => `${x*1000},${y*1000}`).join(" ")} className="robot-mask" vectorEffect="non-scaling-stroke" />}
        {robots.map((seed, index) => seed.box && overlayRect(seed.box, `robot-${index}`,
          `robot-box ${roster.length > 1 ? `robot-${seed.alliance}` : ""}`))}
        {frame?.balls.map((ball, i) => overlayRect(ball.box, `ball-${i}`, "ball-box"))}
        {shot && <polyline className="launch-trace" vectorEffect="non-scaling-stroke"
          points={shot.map(([,x,y]) => `${x*1000},${y*1000}`).join(" ")} />}
        {preview && drawing && overlayRect(preview, "preview", "preview-box")}
      </svg>}
      {overlay && robots.map((seed, index) => seed.box && <span key={`${seed.team}-${index}`}
        className={`track-label ${roster.length > 1 ? `track-${seed.alliance}` : ""}`}
        style={{ left: `${Math.min(seed.box[0] * 100, 76)}%`, top: `${Math.max(1, seed.box[1] * 100 - 5)}%` }}>
        <Crosshair /> {match.demo ? "ROBOT 408" : seed.team ? `TEAM ${seed.team}` : "TARGET ROBOT"}
      </span>)}
      {drawing && <span className="draw-tip">{anchor ? "Click the opposite corner" : "Click the first corner"}</span>}
      {mediaError && <span className="draw-tip" role="alert">Playback failed. Try importing the video again.</span>}
    </div>
    <div className="transport">
      <div className="timeline">
        <div className="timeline-track" style={{ "--position": `${clockPosition}%` } as CSSProperties} />
        <input type="range" min={0} max={match.duration} step={.01} value={time}
          aria-label="Seek match video" aria-valuetext={timecode(time, true)} onChange={e => jump(Number(e.target.value))} />
        {match.events.filter(row => row.status !== "dismissed" && isReviewEvent(row.kind)).map(row => <button
          key={row.id} type="button" className={`timeline-marker marker-${row.kind}`}
          style={{ left: `${Math.max(1, Math.min(99, row.time / match.duration * 100))}%` }}
          title={`${timecode(row.time, true)} · ${eventTitle(row)}`} aria-label={`Jump to ${eventTitle(row)} at ${timecode(row.time)}`}
          onClick={() => select(row)} />)}
      </div>
      <div className="transport-controls">
        <div className="transport-group">
          <Button variant="ghost" size="icon-sm" aria-label="Back five seconds" title="Back five seconds" onClick={() => jump(time - 5)}><SkipBack /></Button>
          <Button className="play-button" size="icon-sm" aria-label={playing ? "Pause" : "Play"} title={playing ? "Pause" : "Play"} onClick={toggle}>{playing ? <Pause /> : <Play />}</Button>
          <Button variant="ghost" size="icon-sm" aria-label="Forward five seconds" title="Forward five seconds" onClick={() => jump(time + 5)}><SkipForward /></Button>
          <span className="timecode">{timecode(time, true)} <span>/ {timecode(match.duration)}</span></span>
        </div>
        <div className="transport-group">
          <select className="play-rate" value={rate} aria-label="Playback speed" onChange={e => {
            const speed = Number(e.target.value); setRate(speed); if (video.current) video.current.playbackRate = speed;
          }}>{[.25, .5, 1, 1.5, 2].map(speed => <option value={speed} key={speed}>{speed}×</option>)}</select>
          <Button variant="ghost" size="icon-sm" aria-label="Show tracking overlays" aria-pressed={overlay} title="Tracking overlays" onClick={() => setOverlay(!overlay)}>{overlay ? <Eye /> : <EyeOff />}</Button>
          <Button variant="ghost" size="icon-sm" aria-label="Fullscreen video" title="Fullscreen video" onClick={() => void stage.current?.requestFullscreen().catch(() => {})}><Maximize /></Button>
        </div>
      </div>
    </div>
  </section>;
}
