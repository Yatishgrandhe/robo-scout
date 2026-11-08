export type Box = [number, number, number, number];
export type Alliance = "red" | "blue";
export type ZoneKind = "hive" | "cell" | "flower" | "garden";
export type Zone = { kind: ZoneKind; alliance: Alliance; box: Box; state_frames?: [number, number] | null; automatic?: boolean };
export type RobotSeed = { team: string; alliance: Alliance; robot: Box | null; seed_time: number };
export type Model = "generic" | "synthetic" | "ftc";
export type Setup = {
  team: string; alliance: Alliance; robot: Box | null; seed_time: number;
  start_time: number | null; end_time: number | null; zones: Zone[];
  model: Model; sample_fps: 6 | 10 | 15; sam: boolean;
  field?: Box | null;
  camera_motion?: boolean;
  scan_shots?: boolean;
  auto_zones?: boolean;
  robots?: RobotSeed[];
  anchors?: { time: number; box: Box; team: string }[];
};
export type Status = "confirmed" | "suggested" | "dismissed";
export const kinds = {
  cell_entry: "Piece entered Hive Cell",
  cell_ball: "Piece scored in Cell", garden_ball: "Piece scored in Garden", flower_ball: "Piece scored in Flower",
  nectar_pickup: "Nectar pickup", penalty: "Penalty", note: "Scouting comment",
  pollen_launch: "Pollen launch", red_nectar_launch: "Red Nectar launch", blue_nectar_launch: "Blue Nectar launch",
  nectar_launch: "Nectar launch (color unclear)",
};
export const pieceNames = {
  pollen: "Pollen", red_nectar: "Red Nectar", blue_nectar: "Blue Nectar", nectar: "Nectar (color unclear)",
} as const;
export type Piece = keyof typeof pieceNames;
export type ReviewKind = keyof typeof kinds;
export type Kind = ReviewKind | "hive_tip" | "bottom_nectar" | "leave" | "park";
export const isReviewEvent = (kind: Kind) => Object.hasOwn(kinds, kind);
export const isScoredPiece = (kind: Kind) => kind === "cell_ball" || kind === "flower_ball" || kind === "garden_ball";
export const isPieceOutcome = (kind: Kind) => kind === "cell_entry" || isScoredPiece(kind);
export type EventRow = {
  id: string; kind: Kind; time: number; title: string; rule: string; status: Status;
  count: number; severity: "warning" | "minor" | "major" | null;
  card: "yellow" | "red" | null; evidence: string; source: string;
  piece?: Piece | null;
  team?: string;
  penalty_points?: number;
  trajectory?: [number, number, number][];
  launch_time?: number;
  launch_id?: string;
};
export type MatchMeta = {
  id: string; name: string; duration: number; width: number; height: number; video: string;
  poster: string; demo: boolean; config: Setup | null;
  folder_id?: string | null;
  penalty_points?: number;
  state: "setup" | "processing" | "complete" | "error"; stage: string; progress: number;
};
export type Match = MatchMeta & {
  ruleset: string; manual: string; events: EventRow[];
  reused_review?: boolean;
  opened_team?: string | null;
  field_preset?: Pick<Setup, "zones" | "start_time" | "end_time" | "field" | "camera_motion">;
  open_attribution?: ("balls" | "tips")[];
  summary: { confirmed: ReviewCounts; suggested: ReviewCounts; pending: number };
  team_summaries?: Record<string, Match["summary"]>;
  scouting?: { robots: Record<string, RobotScouting>; alliances: Partial<Record<Alliance, AllianceScouting>> };
  official_cells?: { red: number; blue: number; source: string };
};
type ReviewCounts = {
  pollen: { shot: number; entered: number; scored: number }; nectar: { shot: number; entered: number; scored: number };
  unclassified: number;
  pickups: { own: number; opponent: number; unassigned: number };
  penalty_points: number; warnings: number; cards: number;
};
export type Frame = { time: number; robot: Box | null; robots?: Record<string, Box | null>;
  mask?: [number, number][]; zones?: Box[]; balls: { class: string; box: Box }[] };
export type Defense = { rating: number | null; pressure_seconds: number; observed_seconds: number; coverage: number;
  episodes: { start: number; end: number; opponent: string }[];
};
export type RobotScouting = { comment: string; clip: { time: number; title: string } | null };
export type AllianceScouting = { rating: number | null; entries: Record<string, number>; transitions: number;
  switches: { from_team: string; to_team: string; start: number; end: number }[]; switch_window: number;
};
export type Evidence = { frames: Frame[]; warnings: string[]; coverage: number | null; segmentation: boolean; device?: string;
  shot_scan?: boolean; pickup_scan?: boolean; entry_scan?: boolean };
export type TeamEvidence = Evidence & {
  defense?: Record<string, Defense>; team_coverage?: Record<string, number>;
  end_cells?: Partial<Record<Alliance, { pollen: number; nectar: number }>>;
};
export type Options = { models: Record<Model, boolean>; sam: boolean };
export const zoneNames: Record<ZoneKind, string> = { hive: "Hive structure", cell: "Hive Cell", flower: "Flower", garden: "Garden" };
export const labels: Record<Status, string> = { confirmed: "Reviewed", suggested: "Needs review", dismissed: "Dismissed" };
export const eventTitle = (row: EventRow) => row.status === "confirmed" && row.source === "vision"
  ? row.title.replace(/^Possible /, "") : row.title;

export function scoringSetupIssues(setup?: Setup | null) {
  const issues = [];
  if (!setup?.zones.some(zone => zone.alliance === setup.alliance && zone.kind !== "hive")) {
    issues.push("mark a scoring zone for the selected alliance");
  }
  if (setup?.start_time == null || setup.end_time == null) issues.push("set the match start and settled end");
  return issues;
}

export function timecode(seconds: number, fractions = false) {
  const units = Math.floor(Math.max(0, seconds) * 100 + 1e-7);
  const m = String(Math.floor(units / 6000)).padStart(2, "0");
  const s = String(Math.floor(units / 100) % 60).padStart(2, "0");
  return fractions ? `${m}:${s}.${String(units % 100).padStart(2, "0")}` : `${m}:${s}`;
}

export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : "The local server could not complete that action.");
  }
  return response.json();
}

export function json(method: string, data: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) };
}
