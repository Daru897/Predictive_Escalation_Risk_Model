import { db, tx, currentPolicy, tickets, hash, secret, snapshot } from "../db";
import { policySchema, type Policy } from "./policy";
import { scoreTicket } from "./engine";
function fingerprint(workspace: string) {
  return hash(
    JSON.stringify(
      tickets(workspace).sort((a, b) => a.key.localeCompare(b.key)),
    ),
  );
}
export function previewPolicy(
  workspace: string,
  draft: unknown,
  expectedRevision: number,
) {
  const config = policySchema.parse(draft);
  return tx(() => {
    const current = currentPolicy(workspace);
    if (current.revision !== expectedRevision)
      throw new Error(
        "The active policy changed. Reload the editor and preview again.",
      );
    const now = Date.now();
    const rows = tickets(workspace).map((t) => {
      const before = scoreTicket(t, now, current.config),
        after = scoreTicket(t, now, config);
      return {
        key: t.key,
        id: t.id,
        subject: t.subject,
        before: before.score,
        after: after.score,
        beforeBand: before.band,
        afterBand: after.band,
      };
    });
    const changed = rows.filter(
      (r) => r.before !== r.after || r.beforeBand !== r.afterBand,
    );
    const token = secret();
    db()
      .prepare("DELETE FROM policy_previews WHERE workspace=?")
      .run(workspace);
    db()
      .prepare("INSERT INTO policy_previews VALUES(?,?,?,?,?,?)")
      .run(
        token,
        workspace,
        current.revision,
        hash(JSON.stringify(config)),
        fingerprint(workspace),
        now + 300000,
      );
    return {
      token,
      at: now,
      total: rows.length,
      changed: changed.length,
      beforeHigh: rows.filter((r) =>
        ["High", "Critical"].includes(r.beforeBand),
      ).length,
      afterHigh: rows.filter((r) => ["High", "Critical"].includes(r.afterBand))
        .length,
      examples: changed
        .sort(
          (a, b) => Math.abs(b.after - b.before) - Math.abs(a.after - a.before),
        )
        .slice(0, 10),
    };
  });
}
export function publishPolicy(
  workspace: string,
  draft: unknown,
  expectedRevision: number,
  previewToken: string,
) {
  const config = policySchema.parse(draft);
  return tx(() => {
    const current = currentPolicy(workspace);
    if (current.revision !== expectedRevision)
      throw new Error(
        "The active policy changed. Reload the editor and preview again.",
      );
    const preview = db()
      .prepare("SELECT * FROM policy_previews WHERE id=? AND workspace=?")
      .get(previewToken, workspace) as
      | {
          revision: number;
          policy_hash: string;
          tickets_hash: string;
          expires: number;
        }
      | undefined;
    if (
      !preview ||
      preview.expires < Date.now() ||
      preview.revision !== current.revision ||
      preview.policy_hash !== hash(JSON.stringify(config)) ||
      preview.tickets_hash !== fingerprint(workspace)
    )
      throw new Error(
        "Preview expired or the tickets or policy changed. Preview again before applying.",
      );
    if (current.revision >= 200)
      throw new Error(
        "Policy version limit reached. Ask your administrator to archive this workspace.",
      );
    const revision = current.revision + 1;
    db()
      .prepare("INSERT INTO policies VALUES(?,?,?,?)")
      .run(workspace, revision, JSON.stringify(config), Date.now());
    db()
      .prepare("DELETE FROM policy_previews WHERE workspace=?")
      .run(workspace);
    snapshot(workspace);
    return { revision, config: config as Policy };
  });
}
