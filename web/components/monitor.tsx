"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import {
  Activity,
  LayoutDashboard,
  Database,
  BarChart3,
  Settings,
  ArrowUpRight,
  Search,
  Download,
  Upload,
  Radio,
  RefreshCw,
  X,
  Check,
  Shield,
  FileSpreadsheet,
  AlertCircle,
  LogOut,
} from "lucide-react";
import PolicyEditor from "./policy-editor";
import { fieldLabels, type Policy, type Rule } from "@/lib/scoring/policy";
import type { Ticket, Action } from "@/lib/domain";
import { columns, required } from "@/lib/fields";
type Risk = {
  score: number;
  band: string;
  factors: {
    ruleId: string;
    field: Rule["field"];
    observed: number | string;
    label: string;
    points: number;
    action: string;
  }[];
  missing: string[];
  coverage: number;
  ageHours: number;
  idleHours: number;
};
type Row = Ticket & { risk: Risk; action: Action | null };
type State = {
  name: string;
  role: "owner" | "viewer";
  mode: "demo" | "private";
  tickets: Row[];
  connection: null | {
    subdomain: string;
    next_sync: number;
    last_sync: number | null;
    error: string | null;
    paused: number;
    has_cursor: number;
  };
  policy: { revision: number; config: Policy };
  policyHistory: { revision: number; config: Policy; created: number }[];
  history: {
    at: number;
    high: number;
    total: number;
    policy_revision: number;
  }[];
  generatedAt: string;
  workerSeenAt: number | null;
  zendeskAvailable: boolean;
};
const labels: Record<string, string> = {
  id: "Ticket ID",
  subject: "Subject",
  status: "Status",
  priority: "Priority",
  created_at: "Created at",
  updated_at: "Updated at",
  group: "Queue / group",
  assignee: "Assigned agent",
  transfers: "Transfer count",
  reopens: "Reopen count",
  response_delay_hours: "Response delay (hours)",
  sla_due_at: "SLA deadline",
};
async function api(path: string, data?: unknown) {
  const r = await fetch(
    path,
    data
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }
      : { cache: "no-store" },
  );
  const v = await r.json();
  if (!r.ok) throw new Error(v.error || "Request failed.");
  return v;
}
const stamp = (v: number | string | null) =>
  v
    ? new Date(v).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Not yet synced";
export default function Monitor() {
  const [state, setState] = useState<State | null>(null),
    [ready, setReady] = useState(false),
    [view, setView] = useState("Overview"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState(""),
    [accessKey, setAccessKey] = useState(""),
    [newName, setNewName] = useState(""),
    [setupKey, setSetupKey] = useState(""),
    [savedKey, setSavedKey] = useState("");
  const [query, setQuery] = useState(""),
    [band, setBand] = useState("All risk levels"),
    [group, setGroup] = useState("All queues"),
    [source, setSource] = useState("All sources"),
    [review, setReview] = useState("All reviews"),
    [showClosed, setShowClosed] = useState(false),
    [selected, setSelected] = useState<Row | null>(null),
    [actionStatus, setActionStatus] = useState("investigating"),
    [actionOwner, setActionOwner] = useState(""),
    [note, setNote] = useState("");
  const [content, setContent] = useState(""),
    [filename, setFilename] = useState(""),
    [headers, setHeaders] = useState<string[]>([]),
    [mapping, setMapping] = useState<Record<string, string>>({}),
    [preview, setPreview] = useState<{
      total: number;
      errorCount: number;
      errors: string[];
      preview: Row[];
    } | null>(null);
  const [subdomain, setSubdomain] = useState(""),
    [email, setEmail] = useState(""),
    [token, setToken] = useState(""),
    [days, setDays] = useState(30),
    [confirm, setConfirm] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const initialized = useRef(false);
  const refresh = useCallback(async () => {
    const v = await api("/api/workspace");
    setState(v);
  }, []);
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    (async () => {
      try {
        await refresh();
      } catch {
        if (new URLSearchParams(location.search).get("demo") === "1") {
          try {
            await api("/api/session", { operation: "demo" });
            await refresh();
          } catch (e) {
            setError((e as Error).message);
          }
        }
      } finally {
        setReady(true);
      }
    })();
  }, [refresh]);
  const hasWorkspace = !!state;
  useEffect(() => {
    if (!hasWorkspace) return;
    const timer = setInterval(
      () => refresh().catch((e) => setError(e.message)),
      30000,
    );
    return () => clearInterval(timer);
  }, [hasWorkspace, refresh]);
  useEffect(() => {
    if (selected && !dialog.current?.open) dialog.current?.showModal();
  }, [selected]);
  async function run(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function mutate(input: unknown) {
    const v = await api("/api/workspace", input);
    if (v.tickets) setState(v);
    return v;
  }
  function openTicket(t: Row) {
    setSelected(t);
    setActionStatus(t.action?.status || "investigating");
    setActionOwner(t.action?.owner || "");
    setNote(t.action?.note || "");
  }
  function close() {
    dialog.current?.close();
    setSelected(null);
  }
  const owner = state?.role === "owner";
  const workerStale =
    !!state?.connection &&
    (!state.workerSeenAt ||
      Date.parse(state.generatedAt) - state.workerSeenAt > 120000);
  const sourceStale =
    !!state?.connection?.last_sync &&
    Date.parse(state.generatedAt) - state.connection.last_sync > 300000;
  const active =
    state?.tickets.filter((t) => !["solved", "closed"].includes(t.status)) ||
    [];
  const urgent = active.filter((t) =>
    ["High", "Critical"].includes(t.risk.band),
  );
  const filtered = (state?.tickets || [])
    .filter(
      (t) =>
        (showClosed || !["solved", "closed"].includes(t.status)) &&
        (band === "All risk levels" || t.risk.band === band) &&
        (group === "All queues" || t.group === group) &&
        (source === "All sources" || t.source === source) &&
        (review === "All reviews" ||
          (t.action?.status || "unreviewed") === review) &&
        `${t.subject} ${t.id} ${t.assignee}`
          .toLowerCase()
          .includes(query.toLowerCase()),
    )
    .sort(
      (a, b) =>
        b.risk.score - a.risk.score || a.created_at.localeCompare(b.created_at),
    );
  if (!ready)
    return (
      <main className="loading">
        <Activity /> Opening your workspace…
      </main>
    );
  if (!state)
    return (
      <>
        <header className="public-header">
          <Link href="/" className="brand">
            <Activity />
            SignalDesk
          </Link>
          <Link href="/case-study">Product case study ↗</Link>
        </header>
        <main className="onboarding">
          <p className="eyebrow">YOUR OPERATIONS WORKSPACE</p>
          <h1>Start with a clearer queue.</h1>
          <p className="lead">
            Explore fictional tickets, open a private workspace, or return using
            your access key.
          </p>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          <div className="three-columns">
            <section className="panel">
              <span className="icon-tile">
                <Activity />
              </span>
              <h2>Take a look around</h2>
              <p>
                Try a complete workflow with fictional support tickets. Each
                demo is isolated and expires after seven days.
              </p>
              <button
                className="button dark"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    await api("/api/session", { operation: "demo" });
                    await refresh();
                  })
                }
              >
                Open demo →
              </button>
            </section>
            <form
              className="panel"
              onSubmit={(e) => {
                e.preventDefault();
                run(async () => {
                  await api("/api/session", {
                    operation: "login",
                    key: accessKey,
                  });
                  setAccessKey("");
                  await refresh();
                });
              }}
            >
              <span className="icon-tile">
                <Shield />
              </span>
              <h2>Return to your workspace</h2>
              <label>
                Workspace access key
                <input
                  type="password"
                  autoComplete="off"
                  required
                  value={accessKey}
                  onChange={(e) => setAccessKey(e.target.value)}
                />
              </label>
              <p className="small muted">
                Use your owner or read-only key. Keys are secrets: keep them in
                your password manager.
              </p>
              <button className="button dark" disabled={busy}>
                Open workspace
              </button>
            </form>
            <form
              className="panel"
              onSubmit={(e) => {
                e.preventDefault();
                run(async () => {
                  const v = await api("/api/session", {
                    operation: "create",
                    name: newName,
                    setupKey,
                  });
                  setSavedKey(v.ownerKey);
                  setSetupKey("");
                  await refresh();
                });
              }}
            >
              <span className="icon-tile">
                <Database />
              </span>
              <h2>Create a private workspace</h2>
              <label>
                Workspace name
                <input
                  required
                  minLength={2}
                  maxLength={80}
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="Acme customer support"
                />
              </label>
              <label>
                Deployment setup key
                <input
                  required
                  type="password"
                  autoComplete="off"
                  value={setupKey}
                  onChange={(e) => setSetupKey(e.target.value)}
                />
              </label>
              <p className="small muted">
                Provided by your deployment administrator. No sample tickets are
                added.
              </p>
              <button className="button dark" disabled={busy}>
                Create workspace
              </button>
            </form>
          </div>
        </main>
      </>
    );
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" href="/">
          <Activity />
          SignalDesk
        </Link>
        <div className="workspace-label">
          <span className="avatar">{state.name[0]}</span>
          <div>
            <strong>{state.name}</strong>
            <small>
              {state.mode === "demo" ? "Demo workspace" : "Private workspace"} ·{" "}
              {state.role}
            </small>
          </div>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Workspace navigation">
          {[
            [LayoutDashboard, "Overview"],
            [Database, "Data sources"],
            [BarChart3, "Trends"],
            [Settings, "Workspace"],
          ].map(([Icon, label]) => {
            const Component = Icon as typeof Activity;
            return (
              <button
                key={String(label)}
                className={view === label ? "active" : ""}
                onClick={() => setView(String(label))}
              >
                <Component size={18} />
                {String(label)}
                {label === "Overview" && <span>{urgent.length}</span>}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="method-note">
            <Shield size={19} />
            <strong>Explainable by design</strong>
            <p>
              Policy version {state.policy.revision}
              <br />
              Rules, not probabilities.
            </p>
          </div>
          <Link href="/case-study">
            Read the case study <ArrowUpRight size={16} />
          </Link>
          <button
            onClick={() =>
              run(async () => {
                await api("/api/session", { operation: "logout" });
                setState(null);
                history.replaceState(null, "", "/monitor");
              })
            }
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-top">
          <span>
            Operations / <strong>{view}</strong>
          </span>
          <div>
            <span className="fresh">
              <i />{" "}
              {state.mode === "demo"
                ? "Demo data"
                : state.connection?.paused
                  ? "Sync paused"
                  : state.connection?.error || workerStale || sourceStale
                    ? "Sync needs attention"
                    : state.connection
                      ? "Zendesk connected"
                      : "CSV workspace"}
            </span>
            <span className="avatar">{state.role === "owner" ? "O" : "V"}</span>
            <button
              aria-label="Sign out of workspace"
              onClick={() =>
                run(async () => {
                  await api("/api/session", { operation: "logout" });
                  setState(null);
                  history.replaceState(null, "", "/monitor");
                })
              }
            >
              <LogOut size={16} />
            </button>
          </div>
        </header>
        <div className="workspace-content">
          {error && (
            <div className="error" role="alert">
              <AlertCircle size={18} />
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {message && (
            <p className="success" role="status">
              <Check size={16} />
              {message}
            </p>
          )}
          {savedKey && (
            <div className="key-panel" role="status">
              <strong>Save this access key now</strong>
              <p>
                It is shown once. Anyone with it has the corresponding workspace
                access.
              </p>
              <code>{savedKey}</code>
              <button
                className="button outline"
                onClick={() => {
                  setSavedKey("");
                }}
              >
                I saved the key
              </button>
            </div>
          )}
          {view === "Overview" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">THE OPERATIONS PULSE</p>
                  <h1>Ahead of the escalation.</h1>
                  <p>Know where to look. Know what to do next.</p>
                </div>
                <div className="button-row">
                  <a
                    className="button outline"
                    href="/api/workspace?export=csv"
                  >
                    <Download size={16} />
                    Export review
                  </a>
                  <button
                    className="button dark"
                    onClick={() => setView("Data sources")}
                  >
                    <Upload size={16} />
                    Import tickets
                  </button>
                </div>
              </div>
              <div className="stats">
                <article>
                  <span>Open tickets</span>
                  <strong>{active.length}</strong>
                  <small>
                    Across {new Set(active.map((t) => t.group)).size} queues
                  </small>
                </article>
                <article className="stat-alert">
                  <span>Needs attention</span>
                  <strong>
                    {urgent.length}
                    <small>High + critical</small>
                  </strong>
                  <small>
                    {
                      urgent.filter(
                        (t) => !t.action || t.action.status === "unreviewed",
                      ).length
                    }{" "}
                    awaiting review
                  </small>
                </article>
                <article>
                  <span>Interventions in progress</span>
                  <strong>
                    {
                      active.filter((t) => t.action?.status === "investigating")
                        .length
                    }
                  </strong>
                  <small>With an active review</small>
                </article>
                <article>
                  <span>Signal completeness</span>
                  <strong>
                    {active.length
                      ? Math.round(
                          active.reduce((n, t) => n + t.risk.coverage, 0) /
                            active.length,
                        )
                      : 0}
                    <small>%</small>
                  </strong>
                  <small>Presence of 8 scoring fields, not confidence</small>
                </article>
              </div>
              <div className="queue-insight">
                <div>
                  <span className="icon-tile">
                    <Activity size={20} />
                  </span>
                  <p>
                    <strong>
                      {urgent.length
                        ? `${urgent.length} tickets deserve a closer look.`
                        : "A clear starting point for your shift."}
                    </strong>
                    <br />
                    {urgent.length
                      ? "Review the evidence, confirm impact, and give each intervention an owner."
                      : "Import your tickets or adjust the filters to explore your queue."}
                  </p>
                </div>
                <span className="muted small">
                  Scored {stamp(state.generatedAt)}
                </span>
              </div>
              <section className="panel queue-panel">
                <div className="section-title">
                  <div>
                    <h2>
                      Priority queue{" "}
                      <span className="count">{filtered.length}</span>
                    </h2>
                    <p>
                      Ordered by risk index. Source status and review status are
                      separate.
                    </p>
                  </div>
                  <label className="search">
                    <Search size={17} />
                    <input
                      aria-label="Search tickets"
                      placeholder="Find a ticket, subject or agent"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  </label>
                </div>
                <div className="filters">
                  <select
                    aria-label="Filter by risk"
                    value={band}
                    onChange={(e) => setBand(e.target.value)}
                  >
                    {[
                      "All risk levels",
                      "Critical",
                      "High",
                      "Watch",
                      "Low",
                      "Closed",
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                  <select
                    aria-label="Filter by queue"
                    value={group}
                    onChange={(e) => setGroup(e.target.value)}
                  >
                    {[
                      "All queues",
                      ...new Set(state.tickets.map((t) => t.group)),
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                  <select
                    aria-label="Filter by source"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                  >
                    {["All sources", "csv", "zendesk", "demo"].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                  <select
                    aria-label="Filter by review"
                    value={review}
                    onChange={(e) => setReview(e.target.value)}
                  >
                    {[
                      "All reviews",
                      "unreviewed",
                      "investigating",
                      "monitoring",
                      "resolved",
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={showClosed}
                      onChange={(e) => setShowClosed(e.target.checked)}
                    />
                    Include closed
                  </label>
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Ticket / queue</th>
                        <th>Risk index</th>
                        <th>Top signal</th>
                        <th>Assigned agent</th>
                        <th>Review</th>
                        <th>
                          <span className="sr-only">Open ticket</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.slice(0, 100).map((t) => (
                        <tr key={t.key}>
                          <td>
                            <button
                              className="ticket-link"
                              onClick={() => openTicket(t)}
                            >
                              {t.subject}
                            </button>
                            <small>
                              #{t.id} · {t.group} · {t.source}
                            </small>
                          </td>
                          <td>
                            <span
                              className={`badge ${t.risk.band.toLowerCase()}`}
                            >
                              {t.risk.score} / {t.risk.band}
                            </span>
                          </td>
                          <td>
                            <span>
                              {t.risk.factors[0]?.label || "No active signal"}
                            </span>
                            <small>
                              {t.risk.missing.length} optional fields missing
                            </small>
                          </td>
                          <td>{t.assignee}</td>
                          <td>
                            <span className="review-badge">
                              {t.action?.status || "unreviewed"}
                            </span>
                            <small>
                              {t.action?.owner || "No intervention owner"}
                            </small>
                          </td>
                          <td>
                            <button
                              aria-label={`Review ticket ${t.id}`}
                              onClick={() => openTicket(t)}
                            >
                              <ArrowUpRight size={18} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {!filtered.length && (
                  <div className="empty">
                    <Search />
                    <h3>No tickets match this view.</h3>
                    <p>Adjust your filters or import a CSV to get started.</p>
                  </div>
                )}
                {filtered.length > 100 && (
                  <p className="muted small">
                    Showing the top 100 of {filtered.length}. Use filters or
                    export the complete queue.
                  </p>
                )}
              </section>
              <p className="footnote">
                The risk index prioritizes operational review. It is not a
                calibrated escalation probability. Missing inputs are disclosed
                and never inferred as factual events.
              </p>
            </>
          )}
          {view === "Data sources" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">CONNECT YOUR QUEUE</p>
                  <h1>Better inputs. Clearer signals.</h1>
                  <p>Bring a snapshot, or keep your workspace in sync.</p>
                </div>
              </div>
              <div className="source-grid">
                <section className="panel">
                  <div className="section-title">
                    <span className="icon-tile">
                      <FileSpreadsheet />
                    </span>
                    <span className="badge low">Ready to import</span>
                  </div>
                  <h2>CSV upload</h2>
                  <p>
                    Map your columns, review validation, then import. Matching
                    CSV ticket IDs update existing records when the timestamp is
                    newer.
                  </p>
                  {state.mode === "demo" && (
                    <p className="notice">
                      This demo expires after seven days. Use fictional data
                      here.
                    </p>
                  )}
                  <a className="text-link" href="/sample-tickets.csv" download>
                    Download example CSV ↓
                  </a>
                  <label className="upload-zone">
                    <Upload />
                    <strong>{filename || "Choose a CSV file"}</strong>
                    <span>UTF-8 · Up to 1 MB · 2,000 rows</span>
                    <input
                      aria-label="Upload ticket CSV"
                      type="file"
                      accept=".csv,text/csv"
                      disabled={!owner || busy}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (!file) return;
                        setPreview(null);
                        setHeaders([]);
                        run(async () => {
                          if (file.size > 1048576)
                            throw new Error("CSV exceeds 1 MB.");
                          const text = await file.text();
                          const v = await mutate({
                            operation: "inspect",
                            content: text,
                          });
                          setContent(text);
                          setFilename(file.name);
                          setHeaders(v.headers);
                          const aliases: Record<string, string[]> = {
                            id: ["id", "ticket_id", "Ticket ID"],
                            subject: ["subject", "Subject"],
                            status: ["status", "Status"],
                            priority: ["priority", "Priority"],
                            created_at: ["created_at", "Created at"],
                            updated_at: ["updated_at", "Updated at"],
                          };
                          setMapping(
                            Object.fromEntries(
                              columns.map((k) => [
                                k,
                                v.headers.find((h: string) =>
                                  (aliases[k] || [k]).includes(h),
                                ) || "",
                              ]),
                            ),
                          );
                        });
                      }}
                    />
                  </label>
                  {headers.length > 0 && (
                    <>
                      <h3>Match your columns</h3>
                      <p className="small muted">
                        * Required. Dates must be ISO 8601 with timezone, such
                        as 2026-10-01T09:00:00Z. Status:
                        new/open/pending/hold/solved/closed. Priority:
                        low/normal/high/urgent.
                      </p>
                      <div className="mapping">
                        {columns.map((k) => (
                          <label key={k}>
                            {labels[k]}
                            {required.includes(k) ? " *" : ""}
                            <select
                              value={mapping[k] || ""}
                              onChange={(e) => {
                                setMapping({ ...mapping, [k]: e.target.value });
                                setPreview(null);
                              }}
                            >
                              <option value="">
                                {required.includes(k)
                                  ? "Choose a column"
                                  : "Not available"}
                              </option>
                              {headers.map((h) => (
                                <option key={h}>{h}</option>
                              ))}
                            </select>
                          </label>
                        ))}
                      </div>
                      <button
                        className="button outline"
                        disabled={busy}
                        onClick={() =>
                          run(async () =>
                            setPreview(
                              await mutate({
                                operation: "import",
                                content,
                                mapping,
                                preview: true,
                              }),
                            ),
                          )
                        }
                      >
                        Validate & preview
                      </button>
                    </>
                  )}
                  {preview && (
                    <div className="import-preview">
                      <h3>
                        {preview.errorCount
                          ? `${preview.errorCount} rows need attention`
                          : `${preview.total} tickets ready`}
                      </h3>
                      {preview.errors.map((e, i) => (
                        <p className="error small" key={i}>
                          {e}
                        </p>
                      ))}
                      {preview.preview.map((t) => (
                        <p key={t.id}>
                          <span className="badge">{t.risk.score}</span> #{t.id}{" "}
                          · {t.subject}
                        </p>
                      ))}
                      <p className="small muted">
                        No data has changed yet. Imports reject the entire file
                        if any row is invalid.
                      </p>
                      <button
                        className="button dark"
                        disabled={busy || !!preview.errorCount}
                        onClick={() =>
                          run(async () => {
                            await mutate({
                              operation: "import",
                              content,
                              mapping,
                              preview: false,
                            });
                            setPreview(null);
                            setHeaders([]);
                            setContent("");
                            setFilename("");
                            setMessage(
                              "Tickets imported. Open Overview to review your queue.",
                            );
                          })
                        }
                      >
                        Import {preview.total} tickets
                      </button>
                    </div>
                  )}
                </section>
                <section className="panel">
                  <div className="section-title">
                    <span className="icon-tile">
                      <Radio />
                    </span>
                    <span className="badge">Customer-hosted connector</span>
                  </div>
                  <h2>Zendesk Support</h2>
                  <p>
                    Incremental updates with saved progress. Checks
                    approximately every two minutes after catching up; Zendesk
                    excludes the most recent minute.
                  </p>
                  {state.mode === "demo" ? (
                    <div className="empty compact">
                      <Shield />
                      <h3>Connect in a private workspace</h3>
                      <p>
                        Sign out and create a private workspace with your
                        deployment setup key. Real credentials are never
                        accepted in the public demo.
                      </p>
                    </div>
                  ) : !state.zendeskAvailable ? (
                    <p className="notice">
                      Your administrator must configure TOKEN_ENCRYPTION_KEY
                      before connecting.
                    </p>
                  ) : state.connection ? (
                    <div>
                      <div className="connection-state">
                        <span className="dot" />
                        <strong>
                          {state.connection.subdomain}.zendesk.com
                        </strong>
                      </div>
                      <dl>
                        <dt>Last completed catch-up</dt>
                        <dd>{stamp(state.connection.last_sync)}</dd>
                        <dt>Next scheduled check</dt>
                        <dd>
                          {state.connection.paused
                            ? "Paused"
                            : stamp(state.connection.next_sync)}
                        </dd>
                        <dt>Saved progress</dt>
                        <dd>
                          {state.connection.has_cursor
                            ? "Cursor saved"
                            : "Initial import pending"}
                        </dd>
                      </dl>
                      {workerStale && (
                        <p className="error">
                          The sync worker has not checked in recently. Contact
                          your deployment administrator.
                        </p>
                      )}
                      {sourceStale && (
                        <p className="notice">
                          The last completed catch-up is over five minutes old.
                          Treat the queue as potentially stale.
                        </p>
                      )}
                      {state.connection.error && (
                        <p className="error">{state.connection.error}</p>
                      )}
                      <div className="button-row">
                        <button
                          className="button outline"
                          disabled={busy || !owner}
                          onClick={() =>
                            run(async () => {
                              await mutate({
                                operation: "pause",
                                paused: !state.connection?.paused,
                              });
                              setMessage(
                                state.connection?.paused
                                  ? "Sync resumed."
                                  : "Sync paused.",
                              );
                            })
                          }
                        >
                          {state.connection.paused
                            ? "Resume sync"
                            : "Pause sync"}
                        </button>
                        <button
                          className="button outline"
                          disabled={busy || !owner}
                          onClick={() => setConfirm("disconnect")}
                        >
                          Disconnect
                        </button>
                      </div>
                      <p className="small muted">
                        Disconnect removes stored credentials and stops syncing.
                        Imported tickets remain until you clear the Zendesk
                        source in Workspace.
                      </p>
                    </div>
                  ) : (
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        run(async () => {
                          await mutate({
                            operation: "connect",
                            subdomain,
                            email,
                            token,
                            days,
                          });
                          setToken("");
                          setMessage(
                            "Connection saved. The background worker will validate access and begin importing.",
                          );
                        });
                      }}
                    >
                      <label>
                        Zendesk subdomain
                        <input
                          required
                          pattern="[a-z0-9][a-z0-9-]{0,62}"
                          placeholder="your-company"
                          value={subdomain}
                          onChange={(e) => setSubdomain(e.target.value)}
                        />
                      </label>
                      <label>
                        Zendesk user email
                        <input
                          required
                          type="email"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          autoComplete="username"
                        />
                      </label>
                      <label>
                        Zendesk API token
                        <input
                          required
                          type="password"
                          minLength={10}
                          value={token}
                          onChange={(e) => setToken(e.target.value)}
                          autoComplete="off"
                        />
                      </label>
                      <label>
                        Initial lookback (days)
                        <input
                          type="number"
                          min={1}
                          max={365}
                          value={days}
                          onChange={(e) => setDays(Number(e.target.value))}
                        />
                      </label>
                      <p className="small muted">
                        Use a customer-owned account permitted to run
                        incremental exports. Credentials are encrypted on the
                        server. This connector only reads tickets; it does not
                        write to Zendesk.
                      </p>
                      <button className="button dark" disabled={busy || !owner}>
                        Connect Zendesk
                      </button>
                    </form>
                  )}
                  <div className="notice">
                    <strong>Know what is imported</strong>
                    <p>
                      Ticket metadata only. Agent and queue IDs are shown as
                      labels. Transfers, reopens, response delays and SLA
                      deadlines are unavailable through this ticket-only
                      connector and stay marked as missing. No messages or
                      attachments are stored.
                    </p>
                  </div>
                </section>
              </div>
            </>
          )}
          {view === "Trends" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">WATCH THE QUEUE</p>
                  <h1>Patterns worth a closer look.</h1>
                  <p>
                    Observed snapshots from imports and syncs. No simulated
                    improvement claims.
                  </p>
                </div>
                <button className="button outline" onClick={() => run(refresh)}>
                  <RefreshCw size={16} />
                  Refresh
                </button>
              </div>
              <div className="source-grid">
                <section className="panel">
                  <h2>High + critical over time</h2>
                  <p>
                    Each bar is a completed import or synced page. These are
                    counts, not an escalation rate.
                  </p>
                  {state.history.length ? (
                    <div
                      className="history-chart"
                      aria-label="High risk ticket counts over snapshots"
                    >
                      {[...state.history].reverse().map((h, i) => (
                        <div
                          key={i}
                          title={`${stamp(h.at)}: ${h.high} high or critical / ${h.total} open · Policy v${h.policy_revision}`}
                        >
                          <strong>{h.high}</strong>
                          <small>Policy v{h.policy_revision}</small>
                          <i
                            style={{
                              height: Math.max(
                                3,
                                (h.high /
                                  Math.max(
                                    ...state.history.map((x) => x.high),
                                    1,
                                  )) *
                                  130,
                              ),
                            }}
                          />
                          <small>
                            {new Date(h.at).toLocaleTimeString(undefined, {
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </small>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="empty">
                      <BarChart3 />
                      <h3>Your history starts with an import.</h3>
                      <p>
                        Upload a CSV or complete a Zendesk sync to record the
                        first snapshot.
                      </p>
                    </div>
                  )}
                </section>
                <section className="panel">
                  <h2>Queue concentration</h2>
                  <p>Where high and critical tickets sit right now.</p>
                  {[...new Set(active.map((t) => t.group))].map((g) => {
                    const all = active.filter((t) => t.group === g),
                      high = all.filter((t) =>
                        ["High", "Critical"].includes(t.risk.band),
                      ).length;
                    return (
                      <div className="group-row" key={g}>
                        <div>
                          <strong>{g}</strong>
                          <span>
                            {high} / {all.length} open
                          </span>
                        </div>
                        <progress
                          value={high}
                          max={all.length}
                          aria-label={`${g} high risk tickets`}
                        />
                      </div>
                    );
                  })}
                  {!active.length && <p>No open tickets yet.</p>}
                </section>
              </div>
              <section className="panel method">
                <h2>How the risk index works</h2>
                <p>
                  Active policy: {state.policy.config.name} · Version{" "}
                  {state.policy.revision}. Contributions are capped at 100.
                  Closed or solved tickets score zero. Critical ≥
                  {state.policy.config.bands.critical}, High ≥
                  {state.policy.config.bands.high}, Watch ≥
                  {state.policy.config.bands.watch}.
                </p>
                <div className="three-columns">
                  {state.policy.config.rules
                    .filter((r) => r.enabled)
                    .map((r) => (
                      <div key={r.id}>
                        <strong>{r.name}</strong>
                        <p>
                          {r.points} points{r.perUnit ? " per event" : ""},
                          maximum {r.cap}.{" "}
                          {r.queues.length
                            ? "Queues: " + r.queues.join(", ")
                            : "All queues"}
                        </p>
                      </div>
                    ))}
                </div>
                <p className="notice">
                  Highest matching contribution per signal wins. These are
                  configurable operational rules, not a trained model.
                  Historical snapshots keep their original policy version;
                  changes between versions may reflect rule edits rather than
                  changes to ticket risk. “No ticket update” is not customer
                  waiting time, and supplied SLA deadlines do not calculate
                  business hours.
                </p>
              </section>
            </>
          )}
          {view === "Workspace" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">WORKSPACE CONTROLS</p>
                  <h1>{state.name}</h1>
                  <p>
                    {state.mode === "demo"
                      ? "An isolated seven-day sandbox."
                      : "A private workspace using scoped access keys."}
                  </p>
                </div>
              </div>
              <PolicyEditor
                key={state.policy.revision}
                active={state.policy}
                history={state.policyHistory}
                owner={owner}
                onPublished={async () => {
                  await refresh();
                  setMessage(
                    "Business rules applied. All current tickets now use the new policy.",
                  );
                }}
              />
              <div className="source-grid">
                <section className="panel">
                  <h2>Share a read-only view</h2>
                  <p>
                    Viewers can inspect tickets, trends and exports. They cannot
                    import data, edit interventions or manage connections.
                    Access keys are shared credentials, not individual user
                    identities.
                  </p>
                  <div className="button-row">
                    <button
                      className="button dark"
                      disabled={!owner || busy}
                      onClick={() =>
                        run(async () => {
                          const v = await mutate({ operation: "viewer" });
                          setSavedKey(v.key);
                        })
                      }
                    >
                      Create viewer key
                    </button>
                    <button
                      className="button outline"
                      disabled={!owner || busy}
                      onClick={() => setConfirm("revoke")}
                    >
                      Revoke all viewers
                    </button>
                  </div>
                </section>
                <section className="panel">
                  <h2>Data controls</h2>
                  <p>
                    Clear one source without changing the others. Disconnect
                    Zendesk before clearing its imported tickets.
                  </p>
                  <div className="button-row">
                    {(["demo", "csv", "zendesk"] as const).map((s) => (
                      <button
                        key={s}
                        className="button outline"
                        disabled={
                          !owner ||
                          busy ||
                          !state.tickets.some((t) => t.source === s)
                        }
                        onClick={() => setConfirm(`clear:${s}`)}
                      >
                        Clear {s} data
                      </button>
                    ))}
                  </div>
                  <p className="small muted">
                    Ticket subjects and notes may contain customer information.
                    Follow your organization’s data handling policy.
                  </p>
                </section>
              </div>
            </>
          )}
          {confirm && (
            <section className="confirmation" role="alert">
              <strong>
                {confirm === "disconnect"
                  ? "Disconnect Zendesk?"
                  : confirm === "revoke"
                    ? "Revoke all viewer access and sessions?"
                    : `Permanently clear ${confirm.split(":")[1]} tickets and their intervention notes?`}
              </strong>
              <div className="button-row">
                <button
                  className="button dark"
                  disabled={busy}
                  onClick={() =>
                    run(async () => {
                      await mutate(
                        confirm.startsWith("clear:")
                          ? {
                              operation: "clear",
                              source: confirm.split(":")[1],
                            }
                          : { operation: confirm },
                      );
                      setConfirm("");
                      setMessage("Workspace updated.");
                    })
                  }
                >
                  Confirm
                </button>
                <button
                  className="button outline"
                  onClick={() => setConfirm("")}
                >
                  Cancel
                </button>
              </div>
            </section>
          )}
        </div>
      </main>
      <dialog
        ref={dialog}
        onCancel={() => setSelected(null)}
        onClick={(e) => {
          if (e.target === dialog.current) close();
        }}
        aria-labelledby="ticket-title"
      >
        {selected && (
          <div className="ticket-detail">
            <button
              className="close-button"
              aria-label="Close ticket"
              onClick={close}
            >
              <X />
            </button>
            <p className="eyebrow">
              TICKET #{selected.id} / {selected.source}
            </p>
            <h2 id="ticket-title">{selected.subject}</h2>
            <p className="muted">
              {selected.group} · {selected.assignee} · Source status:{" "}
              {selected.status}
            </p>
            <div className="risk-summary">
              <span className={`risk-big ${selected.risk.band.toLowerCase()}`}>
                {selected.risk.score}
                <small>/ 100</small>
              </span>
              <div>
                <span className={`badge ${selected.risk.band.toLowerCase()}`}>
                  {selected.risk.band} priority for review
                </span>
                <p>
                  Risk index · {selected.risk.coverage}% signal completeness
                </p>
              </div>
            </div>
            <h3>Why this ticket stands out</h3>
            {selected.risk.factors.map((f) => (
              <div className="factor" key={f.ruleId}>
                <span>+{f.points}</span>
                <div>
                  <strong>{f.label}</strong>
                  <p>
                    {fieldLabels[f.field]}: {f.observed}
                  </p>
                  <p>{f.action}</p>
                </div>
              </div>
            ))}
            {!selected.risk.factors.length && (
              <p>No active rules contributed to this score.</p>
            )}
            {selected.risk.missing.length > 0 && (
              <p className="notice">
                Unavailable signals:{" "}
                {selected.risk.missing.map((k) => labels[k]).join(", ")}.
                Missing values add no points; a low score does not establish low
                risk.
              </p>
            )}
            <hr />
            <h3>Give the next step an owner.</h3>
            {error && (
              <p role="alert" className="error">
                {error}
              </p>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                run(async () => {
                  await mutate({
                    operation: "action",
                    key: selected.key,
                    status: actionStatus,
                    owner: actionOwner,
                    note,
                  });
                  close();
                  setMessage(
                    "Intervention saved. The source ticket was not changed.",
                  );
                });
              }}
            >
              <div className="two-columns">
                <label>
                  Review state
                  <select
                    disabled={!owner}
                    value={actionStatus}
                    onChange={(e) => setActionStatus(e.target.value)}
                  >
                    {[
                      "unreviewed",
                      "investigating",
                      "monitoring",
                      "resolved",
                    ].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Intervention owner
                  <input
                    disabled={!owner}
                    maxLength={80}
                    value={actionOwner}
                    onChange={(e) => setActionOwner(e.target.value)}
                    placeholder="Team lead or resolver"
                  />
                </label>
              </div>
              <label>
                Next step / handover note
                <textarea
                  disabled={!owner}
                  rows={3}
                  maxLength={2000}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="Record the next action and the agreed follow-up."
                />
              </label>
              <p className="small muted">
                Changes apply within SignalDesk only. Resolved means the
                intervention is complete, not that the source ticket is closed.
              </p>
              {owner && (
                <button className="button dark" disabled={busy}>
                  <Check size={16} />
                  Save intervention
                </button>
              )}
            </form>
          </div>
        )}
      </dialog>
    </div>
  );
}
