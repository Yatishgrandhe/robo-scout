import { Flag, MessageSquarePlus } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { scoringSetupIssues, timecode } from "@/types";
import type { Defense, Evidence, FlowerEvidence, Match, RobotScouting } from "@/types";

export function DefenseReview({ team, defense, scouting, seek, addComment, saving }: {
  team: string; defense?: Defense; scouting?: RobotScouting; seek: (time: number) => void; addComment: () => void; saving: boolean;
}) {
  const highlight = scouting?.clip;
  return <section className="defense-detail" aria-labelledby="defense-heading">
    <div className="defense-heading">
      <h3 id="defense-heading">Defense review <span>· Team {team}</span></h3>
      <Button variant="outline" size="sm" disabled={saving} onClick={addComment}><MessageSquarePlus data-icon="inline-start" /> Add comment</Button>
    </div>
    {scouting && <p><strong>Performance comment</strong> {scouting.comment}</p>}
    {highlight && <Button variant="ghost" size="sm" className="scout-highlight" onClick={() => seek(highlight.time)}>
      Watch {timecode(highlight.time)} · {highlight.title}
    </Button>}
    {defense ? <>
      <p><strong>Tracked defense</strong> {defense.pressure_seconds}s of direct pressure{defense.lane_seconds != null && ` · ${defense.lane_seconds}s of lane defense outside direct pressure`} · {defense.observed_seconds}s with an opponent visible. {defense.rating == null
        ? "More shared tracking is needed for a rating."
        : defense.lane_seconds == null ? "Lane defense needs a new analysis for this recording."
        : "Lane estimates carry half the weight of direct pressure."}</p>
      {defense.reviewed_clips?.map(clip => <p key={clip.time}><strong>Footage review</strong> {clip.comment} <Button
        variant="ghost" size="sm" onClick={() => seek(clip.time)}>Watch {timecode(clip.time)}</Button></p>)}
      {(defense.episodes.length > 0 || !!defense.lane_episodes?.length) && <details className="alliance-method defense-evidence">
        <summary>Tracked defense clips</summary>
        <div className="defense-clips">
          {defense.episodes.map(episode => <Button key={episode.start} variant="ghost" size="sm" onClick={() => seek(episode.start)}>
            {timecode(episode.start)} · Pressure vs. {episode.opponent}
          </Button>)}
          {defense.lane_episodes?.map(episode => <Button key={`${episode.start}-${episode.opponent}`} variant="ghost" size="sm" onClick={() => seek(episode.start)}>
            {timecode(episode.start)} · {episode.kind === "lane_hold" ? "Lane hold" : "Lane block"} vs. {episode.opponent}
          </Button>)}
        </div>
      </details>}
    </> : <p>Defense observations appear after tracking opposing robots together.</p>}
  </section>;
}

export function FlowerReview({ flowers, time, seek }: {
  flowers: FlowerEvidence[]; time?: number | null; seek: (time: number) => void;
}) {
  return <section className="ledger" aria-labelledby="flower-review-heading">
    <div className="ledger-heading"><div><h2 id="flower-review-heading">Visible Flower contents</h2>
      <p>Camera estimates after settling. Check depth and Nectar order against the footage.</p></div>
      {time != null && <Button variant="outline" size="sm" onClick={() => seek(time)}>View frame</Button>}</div>
    <ul className="flower-results">{flowers.map((flower, index) => <li key={flower.zone}>
      <strong>Flower {index + 1}</strong>
      <span>{flower.pieces.pollen} Pollen · {flower.pieces.red_nectar} Red Nectar · {flower.pieces.blue_nectar} Blue Nectar</span>
      <small>Owner {flower.owner ?? "unresolved"} · Bottom Nectar {flower.bottom_nectar ?? "unresolved"}</small>
    </li>)}</ul>
  </section>;
}

export function AllianceTeamwork({ alliances, seek }: {
  alliances: NonNullable<Match["scouting"]>["alliances"]; seek: (time: number) => void;
}) {
  const switchWindow = alliances.red?.switch_window ?? alliances.blue?.switch_window;
  if (switchWindow == null) return null;
  return <section className="ledger alliance-review" aria-labelledby="alliance-review-heading">
    <div className="ledger-heading"><div><h2 id="alliance-review-heading">Teamwork fluidity</h2>
      <p>Estimate of how evenly and quickly teammates took turns scoring in the Hive.</p></div></div>
    <div className="alliance-review-grid">
      {(["red", "blue"] as const).map(alliance => {
        const data = alliances[alliance];
        const name = alliance === "red" ? "Red alliance" : "Blue alliance";
        return <section key={alliance} aria-label={name}>
          <h3><span className={`alliance-dot alliance-${alliance}`} />{name}
            <span className="alliance-rating">{data?.rating != null ? `${data.rating}/100` : "Unrated"}</span></h3>
          {data ? <>
            <p>Hive entries · {Object.entries(data.entries).map(([team, count]) => `${team} (${count})`).join(" · ")}</p>
            <p>{data.rating == null ? "Review more Hive entries and scoring attempts from both robots for an estimate."
              : `${data.switches.length} of ${data.transitions} scoring transitions switched robots within ${switchWindow}s.`}</p>
            {data.rating != null && <p>{data.switches.length * 2 >= data.transitions
              ? "Teammates frequently followed each other at the Hive. Review how they approached between turns."
              : "Teammates more often scored in separate runs. Look for ways to set up the next turn earlier."}</p>}
            {data.switches.length > 0 && <div className="defense-clips">
              {data.switches.slice(0, 2).map(clip => <Button key={clip.start} variant="ghost" size="sm" onClick={() => seek(clip.start)}>
                {timecode(clip.start)}–{timecode(clip.end)} · {clip.from_team} → {clip.to_team}
              </Button>)}
            </div>}
          </> : <p>Mark both teammates in one review to compare their scoring rhythm.</p>}
        </section>;
      })}
    </div>
    <details className="alliance-method"><summary>How it’s measured</summary>
      <p>Half the score comes from how evenly teammates split confirmed Hive entries. The other half comes from the share of consecutive scoring moments where the robot changed within {switchWindow}s. Use this estimate to review scoring rhythm.</p>
    </details>
  </section>;
}

function Count({ verified, suggested, checked, showZero = false }: {
  verified: number; suggested: number; checked: boolean; showZero?: boolean;
}) {
  return <>
    {verified || showZero ? verified : <span aria-label={checked ? "None verified" : "Unavailable"}>—</span>}
    {suggested > 0 && <small className="stat-suggested">+{suggested} to review</small>}
  </>;
}

export function RobotStats({ match, evidence, openSetup, finalCellsAttributed = false }: {
  match: Match; evidence: Evidence; openSetup: () => void; finalCellsAttributed?: boolean;
}) {
  const { confirmed, suggested, pending } = match.summary;
  const issues = scoringSetupIssues(match.config);
  const lowCoverage = evidence.coverage !== null && evidence.coverage < .85;
  const scoringChecked = match.demo || (match.state === "complete" && issues.length === 0 && !lowCoverage
    && !match.open_attribution?.includes("balls"));
  const alliance = match.config?.alliance;

  return <section className="ledger robot-stats" aria-labelledby="robot-stats-heading">
    <div className="ledger-heading">
      <div><h2 id="robot-stats-heading">Robot stats</h2>
        <p>Hive entries count during play. Final pieces are counted after the match. Dashes mean no verified count.</p>
      </div>
      {pending > 0 && <a className="review-jump" href="#events-heading"><Badge variant="secondary" className="badge-suggested">{pending} to review</Badge></a>}
    </div>
    <table className="piece-table">
      <caption className="sr-only">Shots, Hive entries, and final pieces for the selected robot</caption>
      <thead><tr><th scope="col">Game piece</th><th scope="col">Shot</th><th scope="col">Entered Hive</th><th scope="col">Final</th></tr></thead>
      <tbody>{(["pollen", "nectar"] as const).map(kind => <tr key={kind}>
        <th scope="row"><span className={`piece-marker piece-${kind}`} aria-hidden="true" />{kind === "pollen" ? "Pollen" : "Nectar"}</th>
        <td><Count verified={confirmed[kind].shot} suggested={suggested[kind].shot} checked={!!evidence.shot_scan} /></td>
        <td><Count verified={confirmed[kind].entered} suggested={suggested[kind].entered} checked={!!evidence.entry_scan} /></td>
        <td><Count verified={confirmed[kind].scored} suggested={suggested[kind].scored} checked={scoringChecked}
          showZero={finalCellsAttributed} /></td>
      </tr>)}</tbody>
    </table>
    {confirmed.unclassified + suggested.unclassified > 0 && <p className="stat-note">
      {confirmed.unclassified + suggested.unclassified} scored {confirmed.unclassified + suggested.unclassified === 1 ? "piece needs" : "pieces need"} a type. Classify {confirmed.unclassified + suggested.unclassified === 1 ? "it" : "them"} in the event log.
    </p>}
    {!match.demo && match.state === "complete" && (issues.length > 0 || lowCoverage) && <div className="scoring-warning">
      <div><strong>Scoring scan incomplete</strong>
        {issues.length > 0 && <p>To find scored pieces, {issues.join(" and ")}.</p>}
        {lowCoverage && <p>The robot was lost for part of this clip. Review tracking gaps before trusting the counts.</p>}
      </div>
      <Button variant="outline" size="sm" onClick={openSetup}>Open setup</Button>
    </div>}
    <div className="robot-checks">
      <section aria-labelledby="pickup-heading">
        <h3 id="pickup-heading">Nectar pickups</h3>
        <p>{alliance ? `Correct color for this robot is ${alliance}.` : "Set the robot’s alliance to check Nectar color."}</p>
        <dl>
          <div><dt>Own color</dt><dd><Count verified={confirmed.pickups.own} suggested={suggested.pickups.own} checked={!!evidence.pickup_scan} /></dd></div>
          <div><dt>Opponent color</dt><dd><Count verified={confirmed.pickups.opponent} suggested={suggested.pickups.opponent} checked={!!evidence.pickup_scan} /></dd></div>
        </dl>
        {confirmed.pickups.unassigned > 0 && <p>{confirmed.pickups.unassigned} pickups awaiting an alliance.</p>}
        {!match.demo && !evidence.pickup_scan && match.state === "complete" && <p>Run analysis to scan for pickups.</p>}
      </section>
      <section aria-labelledby="penalty-heading">
        <h3 id="penalty-heading"><Flag aria-hidden="true" /> Penalty points</h3>
        <p>Foul points awarded to the opponent.</p>
        <div className="penalty-total">{confirmed.penalty_points}<span>pts logged</span></div>
        {suggested.penalty_points > 0 && <small className="stat-suggested">+{suggested.penalty_points} pts to review</small>}
        <p>{confirmed.warnings} {confirmed.warnings === 1 ? "warning" : "warnings"} · {confirmed.cards} {confirmed.cards === 1 ? "card" : "cards"}</p>
      </section>
    </div>
  </section>;
}
