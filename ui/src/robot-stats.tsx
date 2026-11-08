import { Flag } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { scoringSetupIssues } from "@/types";
import type { Evidence, Match } from "@/types";

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
