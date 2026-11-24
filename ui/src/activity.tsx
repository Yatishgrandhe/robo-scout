import { useEffect, useState } from "react";
import { Activity as ActivityIcon, Cpu, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { api } from "@/types";

type Analysis = { id: string; name: string; stage: string; progress: number };
type Process = {
  pid: number; label: string; detail: string; recording: string | null;
  elapsed: number; cpu: number; memory_mb: number; server: boolean;
};
type Snapshot = { analyses: Analysis[]; processes: Process[] };

function duration(seconds: number) {
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds / 60) % 60, s = seconds % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

/** Header button and panel listing analyses and Robo Scout processes running on this Mac. */
export default function ActivityMonitor({ openMatch }: { openMatch: (id: string) => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [snapshot, setSnapshot] = useState<Snapshot>();
  const [issue, setIssue] = useState("");

  useEffect(() => {
    let stop = false;
    const load = () => api<Snapshot>("/api/activity")
      .then(next => { if (!stop) { setSnapshot(next); setIssue(""); } })
      .catch((error: Error) => { if (!stop) setIssue(error.message); });
    void load();
    const timer = window.setInterval(load, open ? 2000 : 5000);
    return () => { stop = true; window.clearInterval(timer); };
  }, [open]);

  const work = snapshot?.processes.filter(row => !row.server) ?? [];
  const busy = (snapshot?.analyses.length ?? 0) + work.filter(row => row.label !== "Match analysis").length;

  return <>
    <Button variant="ghost" onClick={() => setOpen(true)} aria-label={`Activity, ${busy} running`}>
      <ActivityIcon data-icon="inline-start" /> <span>Activity</span>
      {busy > 0 && <Badge>{busy}</Badge>}
    </Button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="activity-panel sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Activity</DialogTitle>
          <DialogDescription>Analyses and Robo Scout processes running on this Mac. Updates every 2 seconds.</DialogDescription>
        </DialogHeader>
        {issue && <p className="error-copy" role="alert">{issue}</p>}
        <section aria-labelledby="activity-analyses">
          <h3 id="activity-analyses" className="activity-heading">Match analyses</h3>
          {!snapshot?.analyses.length && <p className="activity-empty">No analysis is running.</p>}
          <ul className="activity-list">
            {snapshot?.analyses.map(row => <li key={row.id} className="activity-row">
              <div className="activity-main">
                <strong>{row.name}</strong>
                <span className="activity-meta">{row.stage} · {Math.round((row.progress ?? 0) * 100)}%</span>
                <div className="activity-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
                  aria-valuenow={Math.round((row.progress ?? 0) * 100)} aria-label={`${row.name} progress`}>
                  <span style={{ width: `${Math.round((row.progress ?? 0) * 100)}%` }} />
                </div>
                <code className="activity-id">?match={row.id}</code>
              </div>
              <Button size="sm" variant="outline" onClick={() => { setOpen(false); void openMatch(row.id); }}>
                <ExternalLink data-icon="inline-start" /> Open
              </Button>
            </li>)}
          </ul>
        </section>
        <section aria-labelledby="activity-processes">
          <h3 id="activity-processes" className="activity-heading">Processes</h3>
          {snapshot && !snapshot.processes.length && <p className="activity-empty">No Robo Scout processes found.</p>}
          <ul className="activity-list">
            {snapshot?.processes.map(row => <li key={row.pid} className="activity-row">
              <div className="activity-main">
                <strong><Cpu className="activity-icon" aria-hidden /> {row.label}{row.detail && ` · ${row.detail}`}</strong>
                <span className="activity-meta">
                  PID {row.pid} · running {duration(row.elapsed)} · CPU {row.cpu.toFixed(0)}% · {row.memory_mb} MB
                </span>
              </div>
              {row.recording && <Button size="sm" variant="ghost" onClick={() => { setOpen(false); void openMatch(row.recording!); }}>
                Open
              </Button>}
            </li>)}
          </ul>
        </section>
      </DialogContent>
    </Dialog>
  </>;
}
