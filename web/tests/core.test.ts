import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
process.env.DATABASE_PATH = join(
  mkdtempSync(join(tmpdir(), "signaldesk-")),
  "test.sqlite",
);
process.env.TOKEN_ENCRYPTION_KEY = "a".repeat(64);
import { scoreTicket, sampleTickets, ticketSchema } from "../lib/domain";
import { importCSV, inspectCSV, csvCell } from "../lib/imports";
import {
  db,
  createWorkspace,
  sessionFor,
  session,
  readWorkspace,
  upsert,
  tx,
  tickets,
  secret,
  hash,
} from "../lib/db";
import {
  connect,
  encrypt,
  decrypt,
  syncOne,
  type Connection,
} from "../lib/zendesk";
const now = Date.parse("2026-10-09T10:00:00Z");
test("risk index is deterministic, bounded, evidence-based, and closes solved tickets", () => {
  const t = sampleTickets(now)[0];
  const risk = scoreTicket(t, now);
  assert.equal(risk.score, 50);
  assert.equal(risk.band, "High");
  assert.equal(scoreTicket({ ...t, status: "closed" }, now).score, 0);
  assert.equal(
    scoreTicket(
      {
        ...t,
        transfers: 999,
        reopens: 999,
        response_delay_hours: 100,
        sla_due_at: "2026-10-01T00:00:00Z",
      },
      now,
    ).score,
    100,
  );
  assert.equal(
    scoreTicket({ ...t, transfers: null }, now).missing.includes("transfers"),
    true,
  );
});
test("timestamps normalize to UTC and reject reverse order or future data", () => {
  const t = sampleTickets(now)[0];
  assert.equal(
    ticketSchema.parse({ ...t, created_at: "2026-10-01T10:00:00+05:30" })
      .created_at,
    "2026-10-01T04:30:00.000Z",
  );
  assert.throws(() =>
    ticketSchema.parse({ ...t, created_at: "2099-01-01T00:00:00Z" }),
  );
  assert.throws(() =>
    ticketSchema.parse({ ...t, updated_at: "2020-01-01T00:00:00Z" }),
  );
});
const csv =
  'Ticket ID,Subject,status,priority,created_at,updated_at\n001,"Refund, missing",open,urgent,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z';
const mapping = {
  id: "Ticket ID",
  subject: "Subject",
  status: "status",
  priority: "priority",
  created_at: "created_at",
  updated_at: "updated_at",
};
test("CSV supports mapping, quoted fields, BOM, optional missing values and row-level rejection", () => {
  const r = importCSV("\ufeff" + csv, mapping);
  assert.equal(r.errorCount, 0);
  assert.equal(r.tickets[0].subject, "Refund, missing");
  assert.equal(r.tickets[0].transfers, null);
  assert.equal(
    importCSV(csv + "\n001,Bad,broken,urgent,no,no", mapping).errorCount,
    1,
  );
  assert.equal(
    importCSV(
      csv +
        "\n001,Duplicate,open,normal,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z",
      mapping,
    ).errorCount,
    1,
  );
  assert.throws(() => inspectCSV("a,a\n1,2"));
  assert.throws(() => inspectCSV("x".repeat(1048577)));
  assert.throws(() => importCSV(csv, { ...mapping, id: "absent" }));
});
test("spreadsheet exports escape formulas and quotes", () => {
  assert.equal(csvCell("=1+2"), '"\'=1+2"');
  assert.equal(csvCell('a"b'), '"a""b"');
  assert.equal(csvCell(" @SUM(1)"), '"\' @SUM(1)"');
});
test("workspace sessions are isolated, private workspaces empty, stale source updates ignored, transaction atomic", () => {
  const a = createWorkspace("Private", "private"),
    b = createWorkspace("Demo", "demo");
  assert.equal(tickets(a.id).length, 0);
  assert.equal(tickets(b.id).length, 12);
  const s = session(sessionFor(a.ownerKey))!;
  assert.equal(s.workspace, a.id);
  assert.equal(session("fabricated"), undefined);
  upsert(a.id, importCSV(csv, mapping).tickets);
  upsert(a.id, [
    {
      ...tickets(a.id)[0],
      subject: "Old title",
      updated_at: "2026-01-01T00:00:00.000Z",
    },
  ]);
  assert.equal(tickets(a.id)[0].subject, "Refund, missing");
  assert.throws(() =>
    tx(() => {
      upsert(a.id, [{ ...tickets(a.id)[0], key: "csv:two", id: "two" }]);
      throw new Error("rollback");
    }),
  );
  assert.equal(tickets(a.id).length, 1);
  assert.equal(readWorkspace(s).tickets.length, 1);
});
test("encrypted credentials are authenticated and bound to workspace", () => {
  const ciphertext = encrypt("never-print-this", "one");
  assert.equal(decrypt(ciphertext, "one"), "never-print-this");
  assert.throws(() => decrypt(ciphertext, "two"));
  assert.throws(() => decrypt(ciphertext.slice(0, -5) + "AAAA", "one"));
});
function connection(id: string) {
  return db()
    .prepare("SELECT * FROM connections WHERE workspace=?")
    .get(id) as Connection;
}
const ticket = {
  id: 123,
  subject: "Imported ticket",
  status: "open",
  priority: "urgent",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
  group_id: 3,
  assignee_id: 4,
};
const mock = (
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
) => (async () => Response.json(value, { status, headers })) as typeof fetch;
test("Zendesk sync commits tickets and cursor together, tolerates duplicates and handles deletion", async () => {
  const w = createWorkspace("Zendesk", "private");
  connect(w.id, "customer", "admin@example.com", "test-token-only", 30);
  let called = "";
  await syncOne(connection(w.id), (async (input) => {
    called = String(input);
    return Response.json({
      tickets: [ticket],
      after_cursor: "page2",
      end_of_stream: false,
    });
  }) as typeof fetch);
  assert.ok(
    called.startsWith(
      "https://customer.zendesk.com/api/v2/incremental/tickets/cursor.json?",
    ),
  );
  assert.equal(connection(w.id).cursor, "page2");
  assert.equal(connection(w.id).last_sync, null);
  assert.equal(tickets(w.id).length, 1);
  assert.equal(tickets(w.id)[0].transfers, null);
  await syncOne(
    connection(w.id),
    mock({ tickets: [ticket], after_cursor: "done", end_of_stream: true }),
  );
  assert.equal(tickets(w.id).length, 1);
  assert.ok(connection(w.id).last_sync);
  await syncOne(
    connection(w.id),
    mock({
      tickets: [{ id: 123, status: "deleted" }],
      after_cursor: "deleted",
      end_of_stream: true,
    }),
  );
  assert.equal(tickets(w.id).length, 0);
});
test("Zendesk rate limiting, provider errors and malformed pages preserve the cursor", async () => {
  const w = createWorkspace("Failures", "private");
  connect(w.id, "customer", "admin@example.com", "test-token-only", 30);
  for (const response of [
    mock({}, 429, { "retry-after": "180" }),
    mock({}, 403),
    mock({
      tickets: [ticket, { id: 456, status: "open" }],
      after_cursor: "bad",
      end_of_stream: true,
    }),
  ]) {
    await syncOne(connection(w.id), response);
    assert.equal(connection(w.id).cursor, null);
    assert.equal(tickets(w.id).length, 0);
    assert.ok(connection(w.id).error);
    assert.ok(connection(w.id).next_sync > Date.now());
  }
});
test("disconnect during a request prevents late data and credentials never appear in workspace responses", async () => {
  const w = createWorkspace("Disconnected", "private");
  connect(w.id, "customer", "admin@example.com", "test-token-only", 30);
  const state = JSON.stringify(readWorkspace(session(sessionFor(w.ownerKey))!));
  assert.ok(!state.includes("test-token-only"));
  assert.ok(!state.includes("secret"));
  await syncOne(connection(w.id), (async () => {
    db().prepare("DELETE FROM connections WHERE workspace=?").run(w.id);
    return Response.json({
      tickets: [ticket],
      after_cursor: "late",
      end_of_stream: true,
    });
  }) as typeof fetch);
  assert.equal(tickets(w.id).length, 0);
});
test("Zendesk host input rejects arbitrary URLs and credential changes keep cursor", () => {
  const w = createWorkspace("Host", "private");
  assert.throws(() =>
    connect(w.id, "localhost/path", "admin@example.com", "test-token-only", 30),
  );
  connect(w.id, "customer", "admin@example.com", "test-token-only", 30);
  db()
    .prepare("UPDATE connections SET cursor=? WHERE workspace=?")
    .run("saved", w.id);
  connect(w.id, "customer", "admin@example.com", "new-token-only", 30);
  assert.equal(connection(w.id).cursor, "saved");
  assert.throws(() =>
    connect(
      w.id,
      "another-customer",
      "admin@example.com",
      "test-token-only",
      30,
    ),
  );
});
test("viewer sessions are invalidated when revoked", () => {
  const w = createWorkspace("Read only", "private");
  const key = secret();
  db().prepare("INSERT INTO keys VALUES(?,?,?)").run(hash(key), w.id, "viewer");
  const token = sessionFor(key);
  assert.equal(session(token)?.role, "viewer");
  db()
    .prepare("DELETE FROM sessions WHERE workspace=? AND role='viewer'")
    .run(w.id);
  db()
    .prepare("DELETE FROM keys WHERE workspace=? AND role='viewer'")
    .run(w.id);
  assert.equal(session(token), undefined);
  assert.throws(() => sessionFor(key));
});
