import { useEffect, useRef, useState } from "react";
import type { FormEvent, RefObject } from "react";
import { Film, Folder, FolderInput, FolderOpen, FolderPlus, Pencil } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { api, json, timecode } from "@/types";
import type { MatchMeta } from "@/types";

type LibraryFolder = { id: string; name: string };
type Edit = { kind: "create" } | { kind: "folder"; folder: LibraryFolder }
  | { kind: "rename" | "move"; item: MatchMeta };

const folderUrl = "/api/library/folders";
const editLabels = { create: "New folder", folder: "Rename folder", rename: "Rename recording", move: "Move recording" };

export default function MatchLibrary({ activeId, trigger, close, openMatch, updateMatch }: {
  activeId?: string;
  trigger: RefObject<HTMLButtonElement | null>;
  close: () => void;
  openMatch: (id: string) => Promise<void>;
  updateMatch: (item: MatchMeta) => void;
}) {
  const [items, setItems] = useState<MatchMeta[]>();
  const [folders, setFolders] = useState<LibraryFolder[]>([]);
  const [location, setLocation] = useState("all");
  const [edit, setEdit] = useState<Edit>();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [issue, setIssue] = useState("");
  const [notice, setNotice] = useState("");
  const [loadAttempt, setLoadAttempt] = useState(0);
  const focusReturn = useRef<HTMLElement | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    const init = { signal: controller.signal };
    void Promise.all([api<MatchMeta[]>("/api/matches", init), api<LibraryFolder[]>(folderUrl, init)])
      .then(([recordings, savedFolders]) => {
        if (controller.signal.aborted) return;
        setItems(recordings);
        setFolders(savedFolders);
      }).catch((error: Error) => {
        if (!controller.signal.aborted) setIssue(error.message);
      });
    return () => controller.abort();
  }, [loadAttempt]);

  const sortedFolders = [...folders].sort((a, b) => a.name.localeCompare(b.name));
  const folderNames = new Map(folders.map(folder => [folder.id, folder.name]));
  const currentFolder = folders.find(folder => folder.id === location);
  const sectionName = currentFolder?.name ?? (location === "unfiled" ? "Unfiled" : "All recordings");
  const visible = items?.filter(item => location === "all" || (item.folder_id ?? "unfiled") === location);
  const counts = new Map<string, number>();
  for (const item of items ?? []) {
    const key = item.folder_id ?? "unfiled";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  function begin(action: Edit, opener: HTMLElement | null) {
    focusReturn.current = opener;
    setEdit(action);
    setValue(action.kind === "create" ? "" : action.kind === "folder" ? action.folder.name
      : action.kind === "rename" ? action.item.name : action.item.folder_id ?? "");
    setIssue("");
    setNotice("");
  }

  function finish() {
    setEdit(undefined);
    setIssue("");
    requestAnimationFrame(() => {
      const opener = focusReturn.current;
      if (opener?.isConnected) opener.focus();
      else heading.current?.focus();
    });
  }

  function navigate(id: string) {
    setLocation(id);
    setEdit(undefined);
    setIssue("");
    setNotice("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!edit || busy) return;
    setBusy(true);
    setIssue("");
    try {
      if (edit.kind === "create" || edit.kind === "folder") {
        const folder = edit.kind === "create"
          ? await api<LibraryFolder>(folderUrl, json("POST", { name: value.trim() }))
          : await api<LibraryFolder>(`${folderUrl}/${edit.folder.id}`, json("PATCH", { name: value.trim() }));
        setFolders(previous => [...previous.filter(row => row.id !== folder.id), folder]);
        setLocation(folder.id);
        setNotice(edit.kind === "create" ? `Created “${folder.name}”.` : `Folder renamed to “${folder.name}”.`);
      } else if ("item" in edit) {
        const item = await api<MatchMeta>(`/api/matches/${edit.item.id}`, json("PATCH", edit.kind === "rename"
          ? { name: value.trim() } : { folder_id: value || null }));
        setItems(previous => previous?.map(row => row.id === item.id ? { ...row, ...item } : row));
        updateMatch(item);
        const destination = folderNames.get(item.folder_id ?? "") ?? "Unfiled";
        setNotice(edit.kind === "rename" ? `Recording renamed to “${item.name}”.` : `Moved “${item.name}” to ${destination}.`);
      }
      finish();
    } catch (error) {
      setIssue((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function view(id: string) {
    setBusy(true);
    setIssue("");
    try {
      await openMatch(id);
      close();
    } catch (error) {
      setIssue((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const places = [
    { id: "all", name: "All recordings", count: items?.length, icon: Film },
    { id: "unfiled", name: "Unfiled", count: counts.get("unfiled") ?? 0, icon: FolderOpen },
    ...sortedFolders.map(folder => ({ ...folder, count: counts.get(folder.id) ?? 0, icon: Folder })),
  ];

  return <Dialog open onOpenChange={(open, details) => {
    if (open || busy) return;
    if (edit && details.reason === "escape-key") finish();
    else close();
  }}>
    <DialogContent className="library-dialog" finalFocus={trigger}>
      <DialogHeader className="library-header">
        <div>
          <DialogTitle>Match library</DialogTitle>
          <DialogDescription>Recordings and reviews saved on this Mac.</DialogDescription>
        </div>
        <Button size="sm" disabled={!items || busy} onClick={event => begin({ kind: "create" }, event.currentTarget)}>
          <FolderPlus data-icon="inline-start" /> New folder
        </Button>
      </DialogHeader>
      <div className="library-body">
        <aside className="library-rail">
          <nav aria-label="Library folders" className="library-folders">
            {places.map(({ id, name, count, icon: Icon }) => <Button key={id}
              variant={location === id ? "secondary" : "ghost"} className="library-place" aria-current={location === id ? "page" : undefined}
              disabled={busy} title={name} onClick={() => navigate(id)}>
              <Icon data-icon="inline-start" /><span>{name}</span>{items && <small aria-hidden="true">{count}</small>}
            </Button>)}
          </nav>
          <Button variant="ghost" className="library-demo" disabled={busy} onClick={() => void view("demo")}>
            <Film data-icon="inline-start" /> Practice simulation
          </Button>
        </aside>
        <section className="library-recordings" aria-labelledby="library-section">
          <div className="library-section">
            <div>
              <h3 id="library-section" ref={heading} tabIndex={-1}>{sectionName}</h3>
              {visible && <p>{visible.length} {visible.length === 1 ? "recording" : "recordings"}</p>}
            </div>
            {currentFolder && <Button size="sm" variant="ghost" disabled={busy} onClick={event =>
              begin({ kind: "folder", folder: currentFolder }, event.currentTarget)}>
              <Pencil data-icon="inline-start" /> Rename folder
            </Button>}
          </div>
          {edit && <form key={edit.kind + ("item" in edit ? edit.item.id : "")} className="library-editor" onSubmit={submit}>
            <FieldGroup>
              <Field data-invalid={!!issue}>
                <FieldLabel htmlFor="library-edit">{editLabels[edit.kind]}</FieldLabel>
                {edit.kind === "move" ? <>
                  <FieldDescription id="library-move-name">{edit.item.name}</FieldDescription>
                  <select id="library-edit" autoFocus value={value} disabled={busy} aria-invalid={!!issue}
                    aria-describedby="library-move-name" onChange={event => setValue(event.target.value)}>
                  <option value="">Unfiled</option>
                  {sortedFolders.map(folder => <option key={folder.id} value={folder.id}>{folder.name}</option>)}
                  </select>
                </> : <Input id="library-edit" autoFocus value={value} disabled={busy} aria-invalid={!!issue}
                  maxLength={edit.kind === "rename" ? 180 : 80} placeholder={edit.kind === "create" ? "Folder name" : undefined}
                  onFocus={event => event.currentTarget.select()} onChange={event => setValue(event.target.value)} />}
              </Field>
            </FieldGroup>
            <div className="library-editor-actions">
              <Button type="button" variant="outline" size="sm" disabled={busy} onClick={finish}>Cancel</Button>
              <Button type="submit" size="sm" disabled={busy || (edit.kind === "move"
                ? value === (edit.item.folder_id ?? "") : !value.trim())}>
                {busy ? "Saving…" : edit.kind === "create" ? "Create folder" : edit.kind === "move" ? "Move recording" : "Save name"}
              </Button>
            </div>
          </form>}
          {issue && <p className="library-error" role="alert">{issue}</p>}
          <div className="library-notice" role="status">{notice}</div>
          {!visible ? <div className="library-loading">
            {issue ? <Button variant="outline" size="sm" onClick={() => { setIssue(""); setLoadAttempt(loadAttempt + 1); }}>Try again</Button>
              : <p role="status">Loading recordings…</p>}
          </div> : visible.length === 0 ? <Empty className="library-empty">
            <EmptyHeader>
              <EmptyMedia variant="icon"><FolderOpen /></EmptyMedia>
              <EmptyTitle>{location === "all" ? "No recordings yet" : "No recordings here"}</EmptyTitle>
            <EmptyDescription>{currentFolder ? "Use a recording’s Move to folder action to put it here." :
                location === "all" ? "Use Import video to add a recording." : "New imports appear here until you move them into a folder."}</EmptyDescription>
            </EmptyHeader>
            {currentFolder && <Button size="sm" variant="outline" onClick={() => navigate("all")}>Browse recordings</Button>}
          </Empty> : <ul className="library-rows">{visible.map(item => {
            const folderName = location === "all" ? folderNames.get(item.folder_id ?? "") : undefined;
            const team = item.config?.team?.trim();
            return <li key={item.id} className="library-row">
            <button type="button" className="library-recording" disabled={busy} onClick={() => void view(item.id)}>
              <span className="library-poster"><img src={item.poster} alt="" loading="lazy" /></span>
              <span className="library-recording-info"><span className="library-recording-name" title={item.name}>{item.name}</span>
                <small>{timecode(item.duration)}
                  {team && !item.name.includes(team) ? ` · Team ${team}` : ""}
                  {item.state === "complete" || !!item.penalty_points ? ` · ${item.penalty_points ?? 0} penalty pts logged` : ""}
                  {folderName ? ` · ${folderName}` : ""}</small>
              </span>
              {item.id === activeId && <Badge variant="secondary">Open</Badge>}
            </button>
            <div className="library-row-actions">
              <Button variant="ghost" size="icon-sm" disabled={busy} aria-label={`Rename ${item.name}`} title="Rename recording"
                onClick={event => begin({ kind: "rename", item }, event.currentTarget)}><Pencil /></Button>
              <Button variant="ghost" size="icon-sm" disabled={busy || !folders.length} aria-label={`Move ${item.name} to folder`}
                title={folders.length ? "Move to folder" : "Create a folder to move recordings"}
                onClick={event => begin({ kind: "move", item }, event.currentTarget)}><FolderInput /></Button>
            </div>
          </li>;
          })}</ul>}
        </section>
      </div>
    </DialogContent>
  </Dialog>;
}
