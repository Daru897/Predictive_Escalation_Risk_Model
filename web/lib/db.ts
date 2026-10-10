import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { createHash, randomBytes } from "node:crypto";
import { sampleTickets, type Ticket, type Action, scoreTicket } from "./domain";
import { DEFAULT_POLICY, policySchema, type Policy } from "./scoring/policy";
let instance: DatabaseSync;
export function db() {
  if (instance) return instance;
  const path = process.env.DATABASE_PATH || "data/monitor.sqlite";
  mkdirSync(dirname(path), { recursive: true });
  instance = new DatabaseSync(path);
  instance.exec(`PRAGMA journal_mode=WAL;PRAGMA busy_timeout=5000;PRAGMA foreign_keys=ON;
 CREATE TABLE IF NOT EXISTS workspaces(id TEXT PRIMARY KEY,name TEXT NOT NULL,mode TEXT NOT NULL,created INTEGER NOT NULL);
 CREATE TABLE IF NOT EXISTS keys(hash TEXT PRIMARY KEY,workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,role TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS sessions(hash TEXT PRIMARY KEY,workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,role TEXT NOT NULL,expires INTEGER NOT NULL);
 CREATE TABLE IF NOT EXISTS tickets(workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,key TEXT,data TEXT NOT NULL,PRIMARY KEY(workspace,key));
 CREATE TABLE IF NOT EXISTS actions(workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,key TEXT,data TEXT NOT NULL,PRIMARY KEY(workspace,key));
 CREATE TABLE IF NOT EXISTS connections(workspace TEXT PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,subdomain TEXT NOT NULL,secret TEXT NOT NULL,cursor TEXT,start_time INTEGER NOT NULL,next_sync INTEGER NOT NULL,last_sync INTEGER,error TEXT,version TEXT NOT NULL,paused INTEGER DEFAULT 0);
 CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY,workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,at INTEGER,high INTEGER,total INTEGER);
 CREATE TABLE IF NOT EXISTS policies(workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,revision INTEGER,config TEXT NOT NULL,created INTEGER NOT NULL,PRIMARY KEY(workspace,revision));
 CREATE TABLE IF NOT EXISTS policy_previews(id TEXT PRIMARY KEY,workspace TEXT REFERENCES workspaces(id) ON DELETE CASCADE,revision INTEGER,policy_hash TEXT,tickets_hash TEXT,expires INTEGER);
 CREATE TABLE IF NOT EXISTS runtime_status(key TEXT PRIMARY KEY,value INTEGER);
 CREATE TABLE IF NOT EXISTS limits(key TEXT PRIMARY KEY,count INTEGER,until INTEGER);
 `);
  instance.exec("BEGIN IMMEDIATE");
  try {
    const columns = instance.prepare("PRAGMA table_info(snapshots)").all() as {
      name: string;
    }[];
    if (!columns.some((c) => c.name === "policy_revision"))
      instance.exec(
        "ALTER TABLE snapshots ADD COLUMN policy_revision INTEGER NOT NULL DEFAULT 0",
      );
    instance
      .prepare(
        "INSERT OR IGNORE INTO policies(workspace,revision,config,created) SELECT id,0,?,created FROM workspaces",
      )
      .run(JSON.stringify(DEFAULT_POLICY));
    instance.exec("COMMIT");
  } catch (error) {
    instance.exec("ROLLBACK");
    throw error;
  }
  return instance;
}
export const hash = (s: string) => createHash("sha256").update(s).digest("hex");
export const secret = () => randomBytes(32).toString("base64url");
export function tx<T>(fn: () => T): T {
  const d = db();
  d.exec("BEGIN IMMEDIATE");
  try {
    const v = fn();
    d.exec("COMMIT");
    return v;
  } catch (e) {
    d.exec("ROLLBACK");
    throw e;
  }
}
export function createWorkspace(name: string, mode: "demo" | "private") {
  return tx(() => {
    const id = secret(),
      ownerKey = secret();
    db()
      .prepare("INSERT INTO workspaces VALUES(?,?,?,?)")
      .run(id, name, mode, Date.now());
    db()
      .prepare("INSERT INTO keys VALUES(?,?,?)")
      .run(hash(ownerKey), id, "owner");
    db()
      .prepare("INSERT INTO policies VALUES(?,?,?,?)")
      .run(id, 0, JSON.stringify(DEFAULT_POLICY), Date.now());
    if (mode === "demo") upsert(id, sampleTickets());
    return { id, ownerKey };
  });
}
export function sessionFor(key: string) {
  const r = db()
    .prepare("SELECT workspace,role FROM keys WHERE hash=?")
    .get(hash(key)) as { workspace: string; role: string } | undefined;
  if (!r) throw new Error("Access key not recognized.");
  const token = secret();
  db()
    .prepare("INSERT INTO sessions VALUES(?,?,?,?)")
    .run(hash(token), r.workspace, r.role, Date.now() + 7 * 86400000);
  return token;
}
export function session(token: string) {
  return db()
    .prepare(
      "SELECT s.workspace,s.role,w.name,w.mode FROM sessions s JOIN workspaces w ON w.id=s.workspace WHERE s.hash=? AND s.expires>?",
    )
    .get(hash(token), Date.now()) as
    | {
        workspace: string;
        role: "owner" | "viewer";
        name: string;
        mode: "demo" | "private";
      }
    | undefined;
}
export function tickets(workspace: string): Ticket[] {
  return (
    db()
      .prepare("SELECT data FROM tickets WHERE workspace=?")
      .all(workspace) as { data: string }[]
  ).map((x) => JSON.parse(x.data));
}
export function upsert(workspace: string, items: Ticket[]) {
  const stmt = db().prepare(
    "INSERT INTO tickets VALUES(?,?,?) ON CONFLICT(workspace,key) DO UPDATE SET data=excluded.data WHERE json_extract(excluded.data,'$.updated_at') >= json_extract(tickets.data,'$.updated_at')",
  );
  const count = Number(
    (
      db()
        .prepare("SELECT count(*) AS n FROM tickets WHERE workspace=?")
        .get(workspace) as { n: number }
    ).n,
  );
  const current = new Set(tickets(workspace).map((t) => t.key));
  if (count + items.filter((t) => !current.has(t.key)).length > 10000)
    throw new Error("Workspace limit of 10,000 tickets reached.");
  for (const t of items) stmt.run(workspace, t.key, JSON.stringify(t));
}
export function currentPolicy(workspace: string): {
  revision: number;
  config: Policy;
} {
  const row = db()
    .prepare(
      "SELECT revision,config FROM policies WHERE workspace=? ORDER BY revision DESC LIMIT 1",
    )
    .get(workspace) as { revision: number; config: string } | undefined;
  return row
    ? {
        revision: row.revision,
        config: policySchema.parse(JSON.parse(row.config)),
      }
    : { revision: 0, config: structuredClone(DEFAULT_POLICY) };
}
export function snapshot(workspace: string) {
  const policy = currentPolicy(workspace),
    now = Date.now();
  const all = tickets(workspace).filter(
    (t) => !["solved", "closed"].includes(t.status),
  );
  const high = all.filter(
    (t) => scoreTicket(t, now, policy.config).score >= policy.config.bands.high,
  ).length;
  db()
    .prepare(
      "INSERT INTO snapshots(workspace,at,high,total,policy_revision) VALUES(?,?,?,?,?)",
    )
    .run(workspace, now, high, all.length, policy.revision);
  db()
    .prepare(
      "DELETE FROM snapshots WHERE workspace=? AND id NOT IN (SELECT id FROM snapshots WHERE workspace=? ORDER BY id DESC LIMIT 100)",
    )
    .run(workspace, workspace);
}
export function readWorkspace(s: NonNullable<ReturnType<typeof session>>) {
  const policy = currentPolicy(s.workspace),
    now = Date.now();
  const actions = Object.fromEntries(
    (
      db()
        .prepare("SELECT key,data FROM actions WHERE workspace=?")
        .all(s.workspace) as { key: string; data: string }[]
    ).map((x) => [x.key, JSON.parse(x.data) as Action]),
  );
  const connection = db()
    .prepare(
      "SELECT subdomain,next_sync,last_sync,error,paused,cursor IS NOT NULL AS has_cursor FROM connections WHERE workspace=?",
    )
    .get(s.workspace);
  return {
    ...s,
    workspace: undefined,
    policy,
    policyHistory: (
      db()
        .prepare(
          "SELECT revision,config,created FROM policies WHERE workspace=? ORDER BY revision DESC LIMIT 20",
        )
        .all(s.workspace) as {
        revision: number;
        config: string;
        created: number;
      }[]
    ).map((r) => ({ ...r, config: JSON.parse(r.config) as Policy })),
    tickets: tickets(s.workspace).map((t) => ({
      ...t,
      risk: scoreTicket(t, now, policy.config),
      action: actions[t.key] || null,
    })),
    connection: connection || null,
    history: db()
      .prepare(
        "SELECT at,high,total,policy_revision FROM snapshots WHERE workspace=? ORDER BY id DESC LIMIT 30",
      )
      .all(s.workspace),
    workerSeenAt:
      (
        db()
          .prepare("SELECT value FROM runtime_status WHERE key='worker'")
          .get() as { value: number } | undefined
      )?.value || null,
    generatedAt: new Date().toISOString(),
    zendeskAvailable: !!process.env.TOKEN_ENCRYPTION_KEY,
  };
}
export function limit(key: string, max = 100, ms = 60000) {
  const now = Date.now();
  const r = db()
    .prepare(
      "INSERT INTO limits VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=CASE WHEN limits.until<? THEN 1 ELSE limits.count+1 END, until=CASE WHEN limits.until<? THEN excluded.until ELSE limits.until END RETURNING count",
    )
    .get(key, now + ms, now, now) as { count: number };
  if (r.count > max) throw new Error("Too many requests. Try again later.");
}
export function prune() {
  db().prepare("DELETE FROM policy_previews WHERE expires<?").run(Date.now());
  db()
    .prepare("DELETE FROM workspaces WHERE mode='demo' AND created<?")
    .run(Date.now() - 7 * 86400000);
  db().prepare("DELETE FROM sessions WHERE expires<?").run(Date.now());
  db().prepare("DELETE FROM limits WHERE until<?").run(Date.now());
}
