import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowDownToLine, ArrowUpRight, CircleDot, Clapperboard, Film, FolderOpen, ShieldCheck, SlidersHorizontal, Upload, UserRoundPlus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Calibration } from "@/calibration";
import { EventDialog, EventLog } from "@/events";
import { isAnchorMark, isTeamMark, Player } from "@/player";
import type { Mark } from "@/player";
import { RobotStats } from "@/robot-stats";
import { api, isReviewEvent, json, timecode } from "@/types";
import type { Alliance, Box, EventRow, Match, Options, Piece, Setup, Status, TeamEvidence } from "@/types";
import "./index.css";

const MatchLibrary = lazy(() => import("@/library"));
const initial: Setup = { team: "", alliance: "red", robot: null, seed_time: 0, start_time: null, end_time: null, zones: [], model: "generic", sample_fps: 10, sam: true };
const emptyEvidence: TeamEvidence = { frames: [], warnings: [], coverage: null, segmentation: false };
const initialOptions: Options = { models: { generic: true, synthetic: false, ftc: false }, sam: true };
const matchUrl = (id: string) => `/api/matches/${id}`;
function App() {
  const [match, setMatch] = useState<Match>();
  const [evidence, setEvidence] = useState(emptyEvidence);
  const [options, setOptions] = useState(initialOptions);
  const [time, setTime] = useState(0);
  const [setup, setSetup] = useState(initial);
  const [editing, setEditing] = useState(false);
  const [drawing, setDrawing] = useState<Mark | null>(null);
  const [zoneAlliance, setZoneAlliance] = useState<Alliance>("red");
  const [selectedId, setSelectedId] = useState<string>();
  const [selectedTeam, setSelectedTeam] = useState<string>();
  const [eventTime, setEventTime] = useState<number | null>(null);
  const [library, setLibrary] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const video = useRef<HTMLVideoElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const footage = useRef<HTMLDivElement>(null);
  const libraryButton = useRef<HTMLButtonElement>(null);

  const activate = useCallback(async (next: Match) => {
    setMatch(next);
    setEvidence(emptyEvidence);
    setSetup(next.config ?? { ...initial, ...next.field_preset });
    setZoneAlliance(next.config?.alliance ?? "red");
    setEditing(next.state === "setup" || next.state === "error");
    setDrawing(null);
    setTime(0);
    setSelectedTeam(undefined);
    setSelectedId(next.demo ? next.events.find(row => isReviewEvent(row.kind))?.id : undefined);
    localStorage.setItem("pitlane-match", next.id);
    const url = new URL(window.location.href);
    if (next.demo) url.searchParams.delete("match");
    else url.searchParams.set("match", next.id);
    window.history.replaceState(null, "", url);
  }, []);

  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("match");
    const id = requested ?? localStorage.getItem("pitlane-match") ?? "demo";
    void api<Match>(matchUrl(id)).catch(() => api<Match>(matchUrl("demo"))).then(activate).catch(e => setError(e.message));
    void api<Options>("/api/options").then(setOptions).catch(() => {});
  }, [activate]);

  const matchId = match?.id;
  const matchState = match?.state;
  useEffect(() => {
    if (!matchId || matchState !== "complete") return;
    let ignore = false;
    void api<TeamEvidence>(`${matchUrl(matchId)}/frames`).then(next => { if (!ignore) setEvidence(next); })
      .catch(e => { if (!ignore) setError(e.message); });
    return () => { ignore = true; };
  }, [matchId, matchState]);

  useEffect(() => {
    if (!matchId || matchState !== "processing") return;
    const request = new AbortController();
    let pending = false;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const next = await api<Match>(matchUrl(matchId), { signal: request.signal });
        if (request.signal.aborted) return;
        setMatch(next);
        if (next.state === "complete" && next.config) setSetup(next.config);
        if (next.state === "error") setEditing(true);
      } catch (e) {
        if (!request.signal.aborted) setError((e as Error).message);
      } finally { pending = false; }
    }, 700);
    return () => { window.clearInterval(timer); request.abort(); };
  }, [matchId, matchState]);

  const jump = useCallback((point: number) => {
    if (!match) return;
    const next = Math.max(0, Math.min(point, match.duration));
    if (video.current) video.current.currentTime = next;
    setTime(next);
  }, [match]);

  function select(row: EventRow) {
    setSelectedId(row.id);
    video.current?.pause();
    jump(row.time);
  }

  async function task(label: string, work: () => Promise<void>) {
    setBusy(label);
    setError("");
    try { await work(); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  }

  function upload(file: File | undefined) {
    if (!file) return;
    if (file.size > 2 * 1024 ** 3) { setError("Choose a video under 2 GB."); return; }
    const body = new FormData();
    body.set("video", file);
    void task("Preparing video", async () => activate(await api<Match>("/api/matches", { method: "POST", body })));
  }

  function arm(mark: Mark | null) {
    setDrawing(mark);
    if (mark) {
      video.current?.pause();
      footage.current?.scrollIntoView({ block: "center", behavior: "instant" });
    }
  }

  function mark(box: Box) {
    if (!drawing) return;
    const stamp = Math.round(time * 1000) / 1000;
    if (drawing === "robot") setSetup({ ...setup, robot: box, seed_time: stamp,
      anchors: setup.anchors?.filter(anchor => anchor.team !== setup.team) });
    else if (drawing === "field") setSetup({ ...setup, field: box });
    else if (isTeamMark(drawing)) {
      const index = Number(drawing.slice(5));
      const team = setup.robots?.[index]?.team;
      setSetup({ ...setup, robots: setup.robots?.map((seed, i) => i === index
        ? { ...seed, robot: box, seed_time: stamp } : seed),
        anchors: setup.anchors?.filter(anchor => anchor.team !== team) });
    }
    else if (isAnchorMark(drawing)) {
      const seed = [setup, ...(setup.robots ?? [])][Number(drawing.slice(7))];
      if (seed && stamp > seed.seed_time && (setup.anchors?.length ?? 0) < 128) {
        const anchors = setup.anchors?.filter(anchor => anchor.team !== seed.team || Math.abs(anchor.time - stamp) > .12) ?? [];
        setSetup({ ...setup, anchors: [...anchors, { time: stamp, box, team: seed.team }] });
      }
    }
    else setSetup({ ...setup, zones: [...setup.zones, { kind: drawing, alliance: zoneAlliance, box }] });
    setDrawing(null);
  }

  function run() {
    if (!match || !setup.robot) return;
    void task("Starting analysis", async () => {
      setMatch(await api<Match>(`${matchUrl(match.id)}/analyze`, json("POST", setup)));
      setEvidence(emptyEvidence);
      setSelectedId(undefined);
      setSelectedTeam(undefined);
      setEditing(false);
      setDrawing(null);
    });
  }

  function decide(row: EventRow, status: Status, piece?: Piece) {
    if (!match) return;
    void task("Saving decision", async () => setMatch(await api<Match>(`${matchUrl(match.id)}/events/${row.id}`, json("PATCH", { status, piece }))));
  }

  function download() {
    if (!match) return;
    const blob = new Blob([JSON.stringify(match, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${match.name.replace(/[^a-zA-Z0-9-]+/g, "-").toLowerCase()}-review.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function openCalibration() {
    setEditing(true);
    requestAnimationFrame(() => document.getElementById("analysis-setup")?.scrollIntoView({ block: "start" }));
  }

  const working = Boolean(busy) || match?.state === "processing";
  const config = match?.config;
  const roster = config?.robots?.length ? [config, ...config.robots] : [];
  const activeTeam = roster.find(seed => seed.team === selectedTeam) ?? roster[0];
  const viewedMatch = match && activeTeam && config ? {
    ...match, config: { ...config, team: activeTeam.team, alliance: activeTeam.alliance },
    summary: match.team_summaries?.[activeTeam.team] ?? match.summary,
    events: match.events.filter(row => (row.team ?? config.team) === activeTeam.team),
    open_attribution: activeTeam.team === config.team ? match.open_attribution : undefined,
  } : match;
  const selected = viewedMatch?.events.find(row => row.id === selectedId);
  const assignedCells = { red: 0, blue: 0 };
  for (const row of match?.events ?? []) {
    if (row.kind !== "cell_ball" || row.status !== "confirmed" || !row.piece) continue;
    const robot = roster.find(seed => seed.team === (row.team ?? config?.team));
    if (robot) assignedCells[robot.alliance] += row.count;
  }

  return <>
    <header className="masthead">
      <div className="masthead-inner">
        <div className="brand"><img className="brand-mark" src="/robo-knights.png" width="96" height="121" alt="" /><span>Robo Scout</span></div>
        <span className="header-divider" />
        <span className="product-label">Robo Knights · FTC film desk</span>
        <div className="header-actions">
          <Button ref={libraryButton} variant="ghost" onClick={() => setLibrary(true)} disabled={!!busy}><FolderOpen data-icon="inline-start" /> <span>Library</span></Button>
          <Button onClick={() => fileInput.current?.click()} disabled={!!busy}><Upload data-icon="inline-start" /> Import video</Button>
          <input ref={fileInput} type="file" accept="video/*,.mkv,.avi,.mp4,.mov" hidden aria-label="Choose match video" onChange={e => {
            upload(e.currentTarget.files?.[0]); e.currentTarget.value = "";
          }} />
        </div>
      </div>
    </header>
    <main className="desk">
      {error && <p className="banner error-copy" role="alert">{error}<Button variant="ghost" size="sm" onClick={() => setError("")}>Dismiss</Button></p>}
      {busy && <p className="banner" role="status">{busy}… {busy === "Preparing video" && "Large recordings can take a moment."}</p>}
      {!match || !viewedMatch ? <div className="initial-state"><Film /><h1>{error ? "Match unavailable" : "Opening the film desk…"}</h1>{error && <p>Import a recording to start a review.</p>}</div> : <>
        <div className="match-head">
          <div>
            <h1>Match review</h1>
            <div className="match-context"><Clapperboard /> <span title={match.name}>{match.name}</span><span className="context-dot" /><span>{match.ruleset}</span>
              <Badge variant="secondary">{match.demo ? "Practice clip" : activeTeam ? `${roster.length} marked robots` : match.config?.team ? `Team ${match.config.team}` :
                match.config ? `${match.config.alliance === "red" ? "Red" : "Blue"} robot` : "Local match"}</Badge>
            </div>
          </div>
          <div className="match-actions">
            {!match.demo && <Button variant="outline" disabled={working} onClick={openCalibration}><SlidersHorizontal data-icon="inline-start" /> Calibrate & analyze</Button>}
            {!match.demo && !roster.length && <Button variant="outline" disabled={working} onClick={() =>
              void task("Creating robot review", async () => activate(await api<Match>(`${matchUrl(match.id)}/robot-reviews`, { method: "POST" })))}>
              <UserRoundPlus data-icon="inline-start" /> Review another robot
            </Button>}
            {!match.demo && <a className={buttonVariants({ variant: "outline" })} data-slot="button"
              href={`${match.video}?download=true`} download>
              <Film data-icon="inline-start" /> Download video
            </a>}
            <Button variant="outline" onClick={download}><ArrowDownToLine data-icon="inline-start" /> Export review</Button>
          </div>
        </div>
        {match.state === "processing" && <div className="progress-panel"><Progress value={match.progress * 100} aria-label="Match analysis progress" /><span role="status">{match.stage} · {Math.round(match.progress * 100)}%</span></div>}
        {match.state === "error" && <p className="banner error-copy" role="alert">{match.stage}</p>}
        <div className="workspace">
          <div className="video-column">
            {activeTeam && <section className="ledger team-overview" aria-labelledby="team-overview-heading">
              <div className="ledger-heading"><div><h2 id="team-overview-heading">Team review</h2>
                <p>Reviewed shots and successful Hive entries. Pieces spilled by a later tip still count as entries. Select a team for its events.</p></div></div>
              <div className="team-table-scroll">
                <table className="team-table">
                  <caption className="sr-only">Reviewed shots and Hive entries for every marked robot</caption>
                  <thead><tr><th scope="col">Team</th><th scope="col">Pollen shot</th><th scope="col">Pollen entered</th>
                    <th scope="col">Nectar shot</th><th scope="col">Nectar entered</th><th scope="col">Defense</th></tr></thead>
                  <tbody>{roster.map(seed => {
                    const counts = match.team_summaries?.[seed.team]?.confirmed;
                    const defense = evidence.defense?.[seed.team];
                    return <tr key={seed.team} data-selected={activeTeam.team === seed.team}>
                      <th scope="row"><Button variant="ghost" size="sm"
                        aria-pressed={activeTeam.team === seed.team} onClick={() => { setSelectedTeam(seed.team); setSelectedId(undefined); }}>
                        <span className={`alliance-dot alliance-${seed.alliance}`} aria-label={`${seed.alliance} alliance`} />
                        {seed.team}
                      </Button></th>
                      <td>{counts?.pollen.shot || "—"}</td>
                      <td>{counts?.pollen.entered || "—"}</td>
                      <td>{counts?.nectar.shot || "—"}</td>
                      <td>{counts?.nectar.entered || "—"}</td>
                      <td>{defense?.rating == null ? "—" : `${defense.rating}/100`}</td>
                    </tr>;
                  })}</tbody>
                </table>
              </div>
              {evidence.defense?.[activeTeam.team] && <div className="defense-detail">
                <p>{evidence.defense[activeTeam.team].pressure_seconds}s of pressure over {evidence.defense[activeTeam.team].observed_seconds}s with an opponent visible. Estimate from tracking.</p>
                {evidence.defense[activeTeam.team].episodes.map(episode => <Button key={episode.start} variant="ghost" size="sm" onClick={() => {
                  video.current?.pause(); jump(episode.start); footage.current?.scrollIntoView({ block: "nearest" });
                }}>{timecode(episode.start)} · vs. {episode.opponent}</Button>)}
              </div>}
            </section>}
            {match.official_cells ? <div className="cell-evidence">
              <strong>Official final Cell count</strong>
              <span>Red · {assignedCells.red} of {match.official_cells.red} attributed / Blue · {assignedCells.blue} of {match.official_cells.blue} attributed</span>
              <small>Alliance totals. Individual credit requires a visible release and a retained piece.</small>
              <a href={match.official_cells.source} target="_blank" rel="noreferrer">FIRST results</a>
            </div> : evidence.end_cells && Object.keys(evidence.end_cells).length > 0 && <div className="cell-evidence">
              <strong>Visible Cell pieces</strong>
              <span>{Object.entries(evidence.end_cells).map(([alliance, pieces]) =>
                `${alliance === "red" ? "Red" : "Blue"} · ${pieces.pollen} Pollen, ${pieces.nectar} Nectar`).join("  /  ")}</span>
              <small>Camera estimate. Team attribution is reviewed separately.</small>
            </div>}
            <RobotStats match={viewedMatch} evidence={activeTeam
              ? { ...evidence, coverage: evidence.team_coverage?.[activeTeam.team] ?? null } : evidence} openSetup={openCalibration}
              finalCellsAttributed={!!activeTeam && assignedCells[activeTeam.alliance] === match.official_cells?.[activeTeam.alliance]} />
            <div ref={footage} id="match-footage"><Player key={match.id} match={viewedMatch} frames={evidence.frames} time={time} setTime={setTime} video={video}
              jump={jump} select={select} setup={setup} drawing={drawing} mark={mark} editing={editing} selected={selected} /></div>
            {!match.demo && <p className="calibration-status">
              {setup.robot ? `Robot marked at ${timecode(setup.seed_time, true)}` : "Robot unmarked"}
              {" · "}{setup.zones.length} {setup.zones.length === 1 ? "zone" : "zones"}
              {" · "}{setup.start_time !== null && setup.end_time !== null ? "Clock set" : "Clock incomplete"}
            </p>}
          </div>
          <EventLog key={`${match.id}-${activeTeam?.team ?? ""}`} match={viewedMatch} selected={selected} select={select} decide={decide}
            seek={point => { video.current?.pause(); jump(point); footage.current?.scrollIntoView({ block: "nearest" }); }} add={() => {
            video.current?.pause(); setEventTime(video.current?.currentTime ?? time);
          }} working={!!busy} />
        </div>
        {editing && <Calibration setup={setup} setSetup={setSetup} match={match} time={time} options={options} drawing={drawing} arm={arm} zoneAlliance={zoneAlliance}
          setZoneAlliance={setZoneAlliance} mark={mark} run={run} working={working} hide={() => { setEditing(false); setDrawing(null); }} />}
        {evidence.warnings.length > 0 && <details className="warnings"><summary>Analysis notes ({evidence.warnings.length})</summary><ul>{evidence.warnings.map(message => <li key={message}>{message}</li>)}</ul></details>}
        <footer className="desk-footer">
          <span><ShieldCheck /> Private video analysis</span>
          {match.demo ? <span><CircleDot /> Scripted example · import a clip to run the models</span> : evidence.coverage !== null ?
            <span><CircleDot /> {Math.round(evidence.coverage * 100)}% robot tracking coverage · {evidence.segmentation ? "SAM outline applied" : "Box tracking"}{evidence.device ? ` · ${evidence.device.toUpperCase()}` : ""}{match.config?.model === "synthetic" ? " · Synthetic weights" : ""}</span> : <span><CircleDot /> Runs on this Mac</span>}
          <a href={match.manual} target="_blank" rel="noreferrer">BIOBUZZ rulebook <ArrowUpRight /></a>
        </footer>
        {eventTime !== null && <EventDialog open close={() => setEventTime(null)} time={eventTime} events={viewedMatch.events} save={async data => {
          const next = await api<Match>(`${matchUrl(match.id)}/events`, json("POST", {
            ...(data as object), ...(activeTeam ? { team: activeTeam.team } : {}),
          }));
          setMatch(next); setSelectedId(next.events.find(row => !match.events.some(old => old.id === row.id))?.id);
        }} />}
      </>}
      {library && <Suspense fallback={<p className="banner" role="status">Opening library…</p>}>
        <MatchLibrary activeId={match?.id} trigger={libraryButton} close={() => {
          setLibrary(false);
          requestAnimationFrame(() => libraryButton.current?.focus());
        }} openMatch={async id => activate(await api<Match>(matchUrl(id)))}
          updateMatch={item => setMatch(previous => previous?.id === item.id
            ? { ...previous, name: item.name, folder_id: item.folder_id } : previous)} />
      </Suspense>}
    </main>
  </>;
}

if (import.meta.env.DEV && import.meta.env.VITE_REACT_INSPECT === "1") {
  void import("react-grab");
  void import("react-scan").then(({ scan }) => scan({ enabled: true }));
}

createRoot(document.getElementById("root")!).render(<App />);
