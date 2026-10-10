import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { identity, owner, body, failure } from "@/lib/http";
import {
  db,
  readWorkspace,
  tx,
  upsert,
  snapshot,
  secret,
  hash,
  limit,
  tickets,
} from "@/lib/db";
import { inspectCSV, importCSV, csvCell } from "@/lib/imports";
import { connect } from "@/lib/zendesk";
import { scoreTicket } from "@/lib/domain";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(req: NextRequest) {
  try {
    const s = identity(req);
    if (req.nextUrl.searchParams.get("export") === "csv") {
      const rows = [
        [
          "source",
          "id",
          "subject",
          "status",
          "priority",
          "risk_index",
          "risk_band",
          "review_status",
          "owner",
          "note",
        ],
      ];
      const state = readWorkspace(s);
      for (const t of state.tickets)
        rows.push([
          t.source,
          t.id,
          t.subject,
          t.status,
          t.priority,
          String(t.risk.score),
          t.risk.band,
          t.action?.status || "unreviewed",
          t.action?.owner || "",
          t.action?.note || "",
        ]);
      return new NextResponse(
        rows.map((r) => r.map(csvCell).join(",")).join("\r\n"),
        {
          headers: {
            "Content-Type": "text/csv",
            "Content-Disposition":
              'attachment; filename="escalation-review.csv"',
            "Cache-Control": "no-store",
          },
        },
      );
    }
    return NextResponse.json(readWorkspace(s), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (e) {
    return failure(e);
  }
}
const operation = z.discriminatedUnion("operation", [
  z.object({ operation: z.literal("inspect"), content: z.string() }),
  z.object({
    operation: z.literal("import"),
    content: z.string(),
    mapping: z.record(z.string(), z.string()),
    preview: z.boolean(),
  }),
  z.object({
    operation: z.literal("action"),
    key: z.string().max(120),
    status: z.enum(["unreviewed", "investigating", "monitoring", "resolved"]),
    owner: z.string().trim().max(80),
    note: z.string().trim().max(2000),
  }),
  z.object({
    operation: z.literal("connect"),
    subdomain: z.string().max(63),
    email: z.string().email().max(200),
    token: z.string().max(1000),
    days: z.number().int().min(1).max(365),
  }),
  z.object({ operation: z.literal("sync") }),
  z.object({ operation: z.literal("pause"), paused: z.boolean() }),
  z.object({ operation: z.literal("disconnect") }),
  z.object({ operation: z.literal("viewer") }),
  z.object({ operation: z.literal("revoke") }),
  z.object({
    operation: z.literal("clear"),
    source: z.enum(["csv", "demo", "zendesk"]),
  }),
]);
export async function POST(req: NextRequest) {
  try {
    const s = owner(req);
    limit(s.workspace);
    const input = operation.parse(await body(req));
    switch (input.operation) {
      case "inspect": {
        const r = inspectCSV(input.content);
        return NextResponse.json({
          headers: r.headers,
          sample: r.rows.slice(0, 3),
          total: r.rows.length,
        });
      }
      case "import": {
        const r = importCSV(input.content, input.mapping);
        if (input.preview)
          return NextResponse.json({
            total: r.total,
            errorCount: r.errorCount,
            errors: r.errors,
            preview: r.tickets
              .slice(0, 5)
              .map((t) => ({ ...t, risk: scoreTicket(t) })),
          });
        if (r.errorCount)
          throw new Error(
            `Import rejected: ${r.errorCount} invalid rows. Preview and fix the CSV first.`,
          );
        tx(() => {
          upsert(s.workspace, r.tickets);
          snapshot(s.workspace);
        });
        break;
      }
      case "action":
        if (!tickets(s.workspace).some((t) => t.key === input.key))
          throw new Error("Ticket not found.");
        db()
          .prepare(
            "INSERT INTO actions VALUES(?,?,?) ON CONFLICT(workspace,key) DO UPDATE SET data=excluded.data",
          )
          .run(
            s.workspace,
            input.key,
            JSON.stringify({ ...input, updatedAt: new Date().toISOString() }),
          );
        break;
      case "connect":
        if (s.mode === "demo")
          throw new Error(
            "Create a private workspace to connect real Zendesk data.",
          );
        connect(
          s.workspace,
          input.subdomain,
          input.email,
          input.token,
          input.days,
        );
        break;
      case "sync":
        db()
          .prepare(
            "UPDATE connections SET next_sync=? WHERE workspace=? AND next_sync<? AND paused=0",
          )
          .run(Date.now(), s.workspace, Date.now());
        break;
      case "pause":
        db()
          .prepare(
            "UPDATE connections SET paused=?,version=?,next_sync=? WHERE workspace=?",
          )
          .run(input.paused ? 1 : 0, secret(), Date.now(), s.workspace);
        break;
      case "disconnect":
        db()
          .prepare("DELETE FROM connections WHERE workspace=?")
          .run(s.workspace);
        break;
      case "viewer": {
        const key = secret();
        db()
          .prepare("INSERT INTO keys VALUES(?,?,?)")
          .run(hash(key), s.workspace, "viewer");
        return NextResponse.json({ key });
      }
      case "revoke":
        tx(() => {
          db()
            .prepare("DELETE FROM keys WHERE workspace=? AND role='viewer'")
            .run(s.workspace);
          db()
            .prepare("DELETE FROM sessions WHERE workspace=? AND role='viewer'")
            .run(s.workspace);
        });
        break;
      case "clear":
        tx(() => {
          if (
            input.source === "zendesk" &&
            db()
              .prepare("SELECT 1 FROM connections WHERE workspace=?")
              .get(s.workspace)
          )
            throw new Error("Disconnect Zendesk before clearing its tickets.");
          for (const t of tickets(s.workspace).filter(
            (t) => t.source === input.source,
          )) {
            db()
              .prepare("DELETE FROM tickets WHERE workspace=? AND key=?")
              .run(s.workspace, t.key);
            db()
              .prepare("DELETE FROM actions WHERE workspace=? AND key=?")
              .run(s.workspace, t.key);
          }
          snapshot(s.workspace);
        });
        break;
    }
    return NextResponse.json(readWorkspace(s));
  } catch (e) {
    return failure(e);
  }
}
