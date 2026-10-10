import { createCipheriv, createDecipheriv, randomBytes } from "node:crypto";
import { db, tx, upsert, snapshot, secret } from "./db";
import { ticketSchema, type Ticket } from "./domain";
export type Connection = {
  workspace: string;
  subdomain: string;
  secret: string;
  cursor: string | null;
  start_time: number;
  next_sync: number;
  last_sync: number | null;
  error: string | null;
  version: string;
  paused: number;
};
function encryptionKey() {
  const k = process.env.TOKEN_ENCRYPTION_KEY || "";
  if (!/^[a-f\d]{64}$/i.test(k))
    throw new Error("Zendesk needs a server encryption key.");
  return Buffer.from(k, "hex");
}
export function encrypt(value: string, workspace: string) {
  const iv = randomBytes(12),
    c = createCipheriv("aes-256-gcm", encryptionKey(), iv);
  c.setAAD(Buffer.from(workspace));
  const body = Buffer.concat([c.update(value, "utf8"), c.final()]);
  return Buffer.concat([iv, c.getAuthTag(), body]).toString("base64");
}
export function decrypt(value: string, workspace: string) {
  const b = Buffer.from(value, "base64"),
    c = createDecipheriv("aes-256-gcm", encryptionKey(), b.subarray(0, 12));
  c.setAAD(Buffer.from(workspace));
  c.setAuthTag(b.subarray(12, 28));
  return Buffer.concat([c.update(b.subarray(28)), c.final()]).toString("utf8");
}
export function connect(
  workspace: string,
  subdomain: string,
  email: string,
  token: string,
  days: number,
) {
  if (
    !/^[a-z0-9][a-z0-9-]{0,62}$/.test(subdomain) ||
    !email.includes("@") ||
    token.length < 10 ||
    token.length > 1000 ||
    !Number.isInteger(days) ||
    days < 1 ||
    days > 365
  )
    throw new Error(
      "Check the subdomain, email, token and lookback (1–365 days).",
    );
  tx(() => {
    const existing = db()
      .prepare("SELECT subdomain FROM connections WHERE workspace=?")
      .get(workspace) as { subdomain: string } | undefined;
    if (
      !existing &&
      db()
        .prepare(
          "SELECT 1 FROM tickets WHERE workspace=? AND key LIKE 'zendesk:%'",
        )
        .get(workspace)
    )
      throw new Error(
        "Clear retained Zendesk tickets in Workspace before connecting again.",
      );
    if (existing && existing.subdomain !== subdomain)
      throw new Error(
        "Disconnect and clear the previous Zendesk source before switching accounts.",
      );
    const encoded = encrypt(JSON.stringify({ email, token }), workspace);
    db()
      .prepare(
        "INSERT INTO connections(workspace,subdomain,secret,start_time,next_sync,version) VALUES(?,?,?,?,?,?) ON CONFLICT(workspace) DO UPDATE SET secret=excluded.secret,version=excluded.version,paused=0,error=NULL,next_sync=excluded.next_sync",
      )
      .run(
        workspace,
        subdomain,
        encoded,
        Math.floor(Date.now() / 1000) - days * 86400,
        Date.now(),
        secret(),
      );
  });
}
export function mapTicket(raw: Record<string, unknown>): Ticket {
  const parsed = ticketSchema.parse({
    id: String(raw.id),
    subject: String(raw.subject || "(No subject)").slice(0, 300),
    status: raw.status,
    priority: raw.priority || "normal",
    created_at: raw.created_at,
    updated_at: raw.updated_at,
    group: raw.group_id ? `Group ${raw.group_id}` : "Unassigned",
    assignee: raw.assignee_id ? `Agent ${raw.assignee_id}` : "Unassigned",
  });
  return { ...parsed, key: `zendesk:${parsed.id}`, source: "zendesk" };
}
export async function syncOne(
  connection: Connection,
  fetcher: typeof fetch = fetch,
) {
  let retry = 120000;
  try {
    const credentials = JSON.parse(
      decrypt(connection.secret, connection.workspace),
    );
    const url = new URL(
      `https://${connection.subdomain}.zendesk.com/api/v2/incremental/tickets/cursor.json`,
    );
    if (connection.cursor) url.searchParams.set("cursor", connection.cursor);
    else url.searchParams.set("start_time", String(connection.start_time));
    url.searchParams.set("per_page", "100");
    const response = await fetcher(url, {
      headers: {
        Authorization: `Basic ${Buffer.from(`${credentials.email}/token:${credentials.token}`).toString("base64")}`,
        Accept: "application/json",
      },
      redirect: "error",
      signal: AbortSignal.timeout(20000),
    });
    if (response.status === 429) {
      const value = response.headers.get("retry-after");
      const seconds = Number(value);
      const delay =
        value &&
        (Number.isFinite(seconds)
          ? seconds * 1000
          : Date.parse(value) - Date.now());
      retry = Math.min(86400000, Math.max(60000, delay || 120000));
      throw new Error(
        "Zendesk rate limit reached. Sync will retry automatically.",
      );
    }
    if ([401, 403].includes(response.status))
      throw new Error(
        "Zendesk denied access. Check credentials, permissions and incremental export availability.",
      );
    if (!response.ok)
      throw new Error(
        "Zendesk is unavailable. The saved cursor will be retried.",
      );
    const reader = response.body?.getReader();
    if (!reader) throw new Error("Empty Zendesk response.");
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > 8 * 1024 * 1024) {
        await reader.cancel();
        throw new Error("Zendesk page is too large.");
      }
      chunks.push(value);
    }
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (
      !Array.isArray(payload.tickets) ||
      payload.tickets.length > 1000 ||
      typeof payload.after_cursor !== "string" ||
      payload.after_cursor.length > 10000 ||
      typeof payload.end_of_stream !== "boolean"
    )
      throw new Error("Unexpected Zendesk page. Cursor was not advanced.");
    const deleted: string[] = [],
      items: Ticket[] = [];
    for (const raw of payload.tickets) {
      if (!raw || typeof raw.id !== "number" || !Number.isSafeInteger(raw.id))
        throw new Error("Zendesk ticket is missing its ID.");
      if (raw.status === "deleted") deleted.push(`zendesk:${raw.id}`);
      else items.push(mapTicket(raw));
    }
    tx(() => {
      const current = db()
        .prepare("SELECT version,paused FROM connections WHERE workspace=?")
        .get(connection.workspace) as
        { version: string; paused: number } | undefined;
      if (!current || current.version !== connection.version || current.paused)
        return;
      upsert(connection.workspace, items);
      for (const key of deleted) {
        db()
          .prepare("DELETE FROM tickets WHERE workspace=? AND key=?")
          .run(connection.workspace, key);
        db()
          .prepare("DELETE FROM actions WHERE workspace=? AND key=?")
          .run(connection.workspace, key);
      }
      db()
        .prepare(
          "UPDATE connections SET cursor=?,next_sync=?,last_sync=?,error=NULL WHERE workspace=? AND version=?",
        )
        .run(
          payload.after_cursor,
          Date.now() + (payload.end_of_stream ? 120000 : 15000),
          payload.end_of_stream ? Date.now() : connection.last_sync,
          connection.workspace,
          connection.version,
        );
      snapshot(connection.workspace);
    });
  } catch (error) {
    const safe =
      error instanceof Error && error.message.startsWith("Zendesk")
        ? error.message
        : "Sync could not finish. Check source data and credentials; saved progress is retained.";
    db()
      .prepare(
        "UPDATE connections SET error=?,next_sync=? WHERE workspace=? AND version=?",
      )
      .run(safe, Date.now() + retry, connection.workspace, connection.version);
  }
}
let running = false;
export async function tick() {
  if (running) return;
  running = true;
  try {
    const due = db()
      .prepare(
        "SELECT * FROM connections WHERE paused=0 AND next_sync<=? ORDER BY next_sync LIMIT 5",
      )
      .all(Date.now()) as Connection[];
    for (const c of due) {
      const lease = db()
        .prepare(
          "UPDATE connections SET next_sync=? WHERE workspace=? AND version=? AND next_sync<=? AND paused=0",
        )
        .run(Date.now() + 60000, c.workspace, c.version, Date.now());
      if (lease.changes) await syncOne(c);
    }
  } finally {
    running = false;
  }
}
