import { useState } from "react";
import type { FormEvent } from "react";
import { Check, Circle, Flag, ListFilter, Plus, RotateCcw, ScanLine } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { eventTitle, isPieceOutcome, isReviewEvent, isScoredPiece, kinds, labels, pieceNames, timecode } from "@/types";
import type { EventRow, Kind, Match, Piece, ReviewKind, Status } from "@/types";

const filters = ["All events", "Launches", "Hive entries", "Retained pieces", "Nectar pickups", "Penalties", "Needs review", "Reviewed", "Dismissed"] as const;
type Filter = typeof filters[number];

function EventIcon({ kind }: { kind: Kind }) {
  if (kind === "penalty") return <Flag />;
  if (isPieceOutcome(kind) || kind.endsWith("_launch") || kind === "nectar_pickup") return <Circle />;
  return <ScanLine />;
}

export function EventLog({ match, selected, select, seek, decide, add, working }: {
  match: Match; selected: EventRow | undefined; select: (row: EventRow) => void;
  seek: (time: number) => void;
  decide: (row: EventRow, status: Status, piece?: Piece) => void; add: () => void; working: boolean;
}) {
  const [filter, setFilter] = useState<Filter>("All events");
  const rows = match.events.filter(row => {
    if (!isReviewEvent(row.kind)) return false;
    if (filter === "All events") return row.status !== "dismissed";
    if (filter === "Launches") return row.kind.endsWith("_launch") && row.status !== "dismissed";
    if (filter === "Hive entries") return row.kind === "cell_entry" && row.status !== "dismissed";
    if (filter === "Retained pieces") return isScoredPiece(row.kind) && row.status !== "dismissed";
    if (filter === "Nectar pickups") return row.kind === "nectar_pickup" && row.status !== "dismissed";
    if (filter === "Penalties") return row.kind === "penalty" && row.status !== "dismissed";
    return labels[row.status] === filter;
  });
  return <section className="event-log" aria-labelledby="events-heading">
    <div className="log-header">
      <div><h2 id="events-heading">Event log</h2><p>Jump to a call in the footage.</p></div>
      <Button size="icon-sm" variant="outline" aria-label="Add event" title="Add event" onClick={add}><Plus /></Button>
    </div>
    <div className="log-filter">
      <ListFilter aria-hidden="true" />
      <select aria-label="Filter events" value={filter} onChange={e => setFilter(e.target.value as Filter)}>{filters.map(item => <option key={item}>{item}</option>)}</select>
      <span>{rows.length} {rows.length === 1 ? "event" : "events"}</span>
    </div>
    {rows.length ? <ol className="events">{rows.map(row => <li key={row.id}>
      <button type="button" onClick={() => select(row)} aria-current={selected?.id === row.id ? "true" : undefined} className="event-row">
        <span className={`event-glyph event-${row.kind}`}><EventIcon kind={row.kind} /></span>
        <span className="event-copy">
          <span className="event-title">{eventTitle(row)}</span>
          <span className="event-meta">{timecode(row.time, true)}{row.rule ? ` · ${row.rule}` : ""}{row.count > 1 ? ` · ×${row.count}` : ""}</span>
        </span>
        <span className={`event-status status-${row.status}`} title={labels[row.status]} aria-label={labels[row.status]}>{row.status === "confirmed" ? <Check /> : row.status === "dismissed" ? <span>–</span> : <><span className="status-dot" /> Review</>}</span>
      </button>
    </li>)}</ol> : <Empty className="log-empty">
      <EmptyHeader><EmptyMedia variant="icon"><ScanLine /></EmptyMedia><EmptyTitle>{filter === "All events" ? "No events yet" : "No matching events"}</EmptyTitle>
        <EmptyDescription>{filter === "All events" ? "Run an analysis or log an event at the playhead." : "Choose another filter to see more calls."}</EmptyDescription></EmptyHeader>
    </Empty>}
    {selected && <div className="event-detail">
      <div className="detail-meta"><Badge variant="secondary" className={`badge-${selected.status}`}>{labels[selected.status]}</Badge><span>{timecode(selected.time, true)}</span></div>
      <h3>{eventTitle(selected)}</h3>
      <p>{selected.evidence || (selected.source === "demo" ? "Scripted event from the practice simulation." : "Logged during private video review.")}</p>
      {selected.kind === "nectar_pickup" && selected.piece && <p className="severity">{pieceNames[selected.piece]}</p>}
      {isPieceOutcome(selected.kind) && <Field className="scored-piece-field">
        <FieldLabel htmlFor="review-piece">Piece type</FieldLabel>
        <select id="review-piece" disabled={working || !!selected.launch_id} value={selected.piece ?? ""}
          onChange={event => decide(selected, selected.status, event.target.value as Piece)}>
          <option value="" disabled>Choose piece type</option>
          {Object.entries(pieceNames).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
        </select>
      </Field>}
      {selected.severity && <p className="severity">
        {selected.severity === "warning" ? "Verbal warning" : `${selected.severity} foul`}
        {selected.card ? ` · ${selected.card} card` : ""}
        {selected.penalty_points != null && ` · ${selected.penalty_points} penalty points`}
      </p>}
      <div className="review-actions">
        {selected.status === "suggested" ? <>
          <Button size="sm" disabled={working} onClick={() => decide(selected, "confirmed")}><Check data-icon="inline-start" /> Confirm</Button>
          <Button size="sm" variant="outline" disabled={working} onClick={() => decide(selected, "dismissed")}>Dismiss</Button>
        </> : <Button size="sm" variant="outline" disabled={working} onClick={() => decide(selected, "suggested")}><RotateCcw data-icon="inline-start" /> Reopen</Button>}
        {selected.launch_time != null && <Button variant="ghost" size="sm" onClick={() => seek(selected.launch_time!)}>
          Watch release · {timecode(selected.launch_time, true)}
        </Button>}
        <a className="footage-jump" href="#match-footage">View footage</a>
      </div>
    </div>}
  </section>;
}

export function EventDialog({ open, close, time, events, save }: {
  open: boolean; close: () => void; time: number; events: EventRow[]; save: (data: unknown) => Promise<void>;
}) {
  const [kind, setKind] = useState<ReviewKind>("pollen_launch");
  const [piece, setPiece] = useState<Piece | "">("");
  const [severity, setSeverity] = useState("minor");
  const [card, setCard] = useState("");
  const [busy, setBusy] = useState(false);
  const [issue, setIssue] = useState("");
  const outcome = isPieceOutcome(kind);
  const needsPiece = outcome || kind === "nectar_pickup";
  const remaining = new Map<string, number>();
  if (outcome) for (const row of events) {
    if (row.launch_id && row.status !== "dismissed" && (kind === "cell_entry") === (row.kind === "cell_entry"))
      remaining.set(row.launch_id, (remaining.get(row.launch_id) ?? 0) - row.count);
  }
  const releases = outcome ? events.filter(row => row.kind.endsWith("_launch") && row.status === "confirmed"
    && row.piece === piece && row.time <= time && row.count + (remaining.get(row.id) ?? 0) > 0).reverse() : [];
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fields = new FormData(e.currentTarget);
    setBusy(true);
    setIssue("");
    try {
      await save({ kind, title: String(fields.get("title")).trim() || kinds[kind], time,
        rule: fields.get("rule"), count: Number(fields.get("count")), evidence: fields.get("evidence"),
        piece: needsPiece ? fields.get("piece") : null,
        launch_id: outcome ? fields.get("launch_id") || null : null,
        severity: kind === "penalty" ? severity : null, card: kind === "penalty" && card ? card : null });
      close();
    } catch (error) {
      setIssue((error as Error).message);
    } finally { setBusy(false); }
  }
  return <Dialog open={open} onOpenChange={value => !value && !busy && close()}>
    <DialogContent>
      <DialogHeader><DialogTitle>Log an event</DialogTitle><DialogDescription>Add a reviewed call at {timecode(time, true)}.</DialogDescription></DialogHeader>
      <form onSubmit={submit}>
        <FieldGroup className="event-form">
          <Field><FieldLabel htmlFor="kind">Event type</FieldLabel><select id="kind" value={kind} onChange={e => {
            setKind(e.target.value as ReviewKind);
            setPiece("");
          }}>{Object.entries(kinds).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select></Field>
          {needsPiece && <Field key={kind}><FieldLabel htmlFor="piece">Piece type</FieldLabel>
            <select id="piece" name="piece" value={piece} onChange={e => setPiece(e.target.value as Piece)} required>
              <option value="" disabled>Choose piece type</option>
              {Object.entries(pieceNames).filter(([key]) => kind !== "nectar_pickup" || key.endsWith("_nectar"))
                .map(([key, name]) => <option key={key} value={key}>{name}</option>)}
            </select>
          </Field>}
          {outcome && releases.length > 0 && <Field><FieldLabel htmlFor="source-launch">Link reviewed release</FieldLabel>
            <select id="source-launch" name="launch_id" key={piece} defaultValue="">
              <option value="">Choose a release (optional)</option>
              {releases.map(row => <option key={row.id} value={row.id}>
                {timecode(row.time, true)} · {eventTitle(row)} · {row.count + (remaining.get(row.id) ?? 0)} available
              </option>)}
            </select>
          </Field>}
          <Field><FieldLabel htmlFor="event-title">Title</FieldLabel><Input id="event-title" name="title" placeholder={kinds[kind]} maxLength={120} /></Field>
          <div className="event-form-pair">
            <Field><FieldLabel htmlFor="rule">Rule or section</FieldLabel><Input id="rule" name="rule" placeholder="e.g. G410" maxLength={32} /></Field>
            <Field><FieldLabel htmlFor="count">Count</FieldLabel><Input id="count" type="number" name="count" min={1} max={60} defaultValue={1} required /></Field>
          </div>
          {kind === "penalty" && <div className="event-form-pair">
            <Field><FieldLabel htmlFor="severity">Severity</FieldLabel><select id="severity" value={severity} onChange={e => setSeverity(e.target.value)}><option value="warning">Warning</option><option value="minor">Minor foul</option><option value="major">Major foul</option></select></Field>
            <Field><FieldLabel htmlFor="card">Card</FieldLabel><select id="card" value={card} onChange={e => setCard(e.target.value)}><option value="">None</option><option value="yellow">Yellow</option><option value="red">Red</option></select></Field>
          </div>}
          <Field><FieldLabel htmlFor="evidence">Evidence note</FieldLabel><Input id="evidence" name="evidence" maxLength={400} /></Field>
        </FieldGroup>
        {issue && <p className="error-copy" role="alert">{issue}</p>}
        <DialogFooter className="event-form-footer"><Button variant="outline" type="button" onClick={close} disabled={busy}>Cancel</Button><Button type="submit" disabled={busy}>Save event</Button></DialogFooter>
      </form>
    </DialogContent>
  </Dialog>;
}
