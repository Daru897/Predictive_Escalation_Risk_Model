import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
process.env.DATABASE_PATH = join(
  mkdtempSync(join(tmpdir(), "policy-test-")),
  "legacy.sqlite",
);
// Exercise additive migration from the original product schema.
const legacy = new DatabaseSync(process.env.DATABASE_PATH);
legacy.exec(
  "CREATE TABLE snapshots(id INTEGER PRIMARY KEY,workspace TEXT,at INTEGER,high INTEGER,total INTEGER);INSERT INTO snapshots(workspace,at,high,total) VALUES('historical',1,2,3)",
);
legacy.close();
import {
  DEFAULT_POLICY,
  policySchema,
  type Policy,
} from "../lib/scoring/policy";
import { scoreTicket } from "../lib/scoring/engine";
import { sampleTickets } from "../lib/domain";
import {
  db,
  createWorkspace,
  currentPolicy,
  readWorkspace,
  session,
  sessionFor,
  tickets,
  upsert,
} from "../lib/db";
import { previewPolicy, publishPolicy } from "../lib/scoring/service";
const now = Date.parse("2026-10-09T10:00:00Z");
const copy = (): Policy => structuredClone(DEFAULT_POLICY);
test("default modular policy preserves all previous demo scores", () => {
  assert.deepEqual(
    sampleTickets(now).map((t) => scoreTicket(t, now).score),
    [50, 30, 48, 100, 58, 28, 65, 55, 63, 86, 28, 0],
  );
});
test("policy schema rejects executable fields, invalid thresholds, duplicate rules and unbounded values", () => {
  const p = copy();
  assert.throws(() =>
    policySchema.parse({ ...p, bands: { watch: 70, high: 45, critical: 25 } }),
  );
  assert.throws(() => policySchema.parse({ ...p, script: "fetch(secret)" }));
  assert.throws(() =>
    policySchema.parse({ ...p, rules: [p.rules[0], p.rules[0]] }),
  );
  assert.throws(() =>
    policySchema.parse({ ...p, rules: [{ ...p.rules[0], points: -5 }] }),
  );
  assert.throws(() =>
    policySchema.parse({ ...p, rules: [{ ...p.rules[0], operator: "gte" }] }),
  );
  assert.throws(() =>
    policySchema.parse({
      ...p,
      rules: [{ ...p.rules[0], field: "age_hours", value: 1, perUnit: true }],
    }),
  );
});
test("queue-scoped rules, disabled rules, caps and missing values are deterministic", () => {
  const t = sampleTickets(now)[0];
  const p = copy();
  p.rules = [{ ...p.rules[0], queues: ["Special queue"], points: 50 }];
  assert.equal(scoreTicket(t, now, p).score, 0);
  assert.equal(scoreTicket({ ...t, group: "Special queue" }, now, p).score, 50);
  p.rules[0].enabled = false;
  assert.equal(scoreTicket({ ...t, group: "Special queue" }, now, p).score, 0);
  p.rules = [{ ...DEFAULT_POLICY.rules.find((r) => r.id === "transfers")! }];
  assert.equal(scoreTicket({ ...t, transfers: 100 }, now, p).score, 15);
  assert.equal(scoreTicket({ ...t, transfers: null }, now, p).score, 0);
});
test("overlapping tiers use the maximum per signal and closed tickets stay excluded", () => {
  const t = { ...sampleTickets(now)[0], created_at: "2026-10-01T00:00:00Z" };
  const p = copy();
  p.rules = p.rules.filter((r) => r.field === "age_hours");
  assert.equal(scoreTicket(t, now, p).score, 20);
  assert.equal(scoreTicket({ ...t, status: "solved" }, now, p).band, "Closed");
  p.bands = { watch: 5, high: 10, critical: 15 };
  assert.equal(scoreTicket(t, now, p).band, "Critical");
});
test("old snapshots migrate as version zero; previews do not mutate active policy", () => {
  assert.equal(
    (
      db()
        .prepare(
          "SELECT policy_revision FROM snapshots WHERE workspace='historical'",
        )
        .get() as { policy_revision: number }
    ).policy_revision,
    0,
  );
  const w = createWorkspace("One", "demo");
  const p = copy();
  p.rules = [];
  const result = previewPolicy(w.id, p, 0);
  assert.ok(result.changed > 0);
  assert.equal(currentPolicy(w.id).revision, 0);
  assert.equal(currentPolicy(w.id).config.rules.length, 13);
  assert.throws(() => publishPolicy(w.id, p, 0, "fake-preview"));
  publishPolicy(w.id, p, 0, result.token);
  assert.equal(currentPolicy(w.id).revision, 1);
  assert.ok(
    readWorkspace(session(sessionFor(w.ownerKey))!).tickets.every(
      (t) => t.risk.score === 0,
    ),
  );
  assert.throws(() => publishPolicy(w.id, p, 0, result.token));
});
test("preview is bound to workspace, payload, ticket data, expiration and active revision", () => {
  const a = createWorkspace("A", "demo"),
    b = createWorkspace("B", "demo");
  const p = copy();
  p.name = "Changed";
  let preview = previewPolicy(a.id, p, 0);
  assert.throws(() => publishPolicy(b.id, p, 0, preview.token));
  assert.throws(() =>
    publishPolicy(a.id, { ...p, name: "Another" }, 0, preview.token),
  );
  const t = tickets(a.id)[0];
  upsert(a.id, [
    {
      ...t,
      subject: "Changed since preview",
      updated_at: new Date().toISOString(),
    },
  ]);
  assert.throws(() => publishPolicy(a.id, p, 0, preview.token));
  preview = previewPolicy(a.id, p, 0);
  db()
    .prepare("UPDATE policy_previews SET expires=0 WHERE id=?")
    .run(preview.token);
  assert.throws(() => publishPolicy(a.id, p, 0, preview.token));
  preview = previewPolicy(a.id, p, 0);
  publishPolicy(a.id, p, 0, preview.token);
  assert.throws(() => previewPolicy(a.id, p, 0));
  assert.equal(currentPolicy(b.id).revision, 0);
});
test("restoring previous settings publishes a new version and retains historical snapshot attribution", () => {
  const w = createWorkspace("History", "demo");
  const p = copy();
  p.rules = [];
  publishPolicy(w.id, p, 0, previewPolicy(w.id, p, 0).token);
  publishPolicy(
    w.id,
    DEFAULT_POLICY,
    1,
    previewPolicy(w.id, DEFAULT_POLICY, 1).token,
  );
  const state = readWorkspace(session(sessionFor(w.ownerKey))!);
  assert.equal(state.policy.revision, 2);
  assert.deepEqual(state.policy.config, DEFAULT_POLICY);
  assert.equal(state.policyHistory.length, 3);
  const history = state.history as { policy_revision: number; high: number }[];
  assert.equal(history[0].policy_revision, 2);
  assert.equal(history[1].policy_revision, 1);
  assert.equal(history[1].high, 0);
});
