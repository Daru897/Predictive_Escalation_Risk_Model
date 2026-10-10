import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { body, failure } from "@/lib/http";
import { createWorkspace, sessionFor, db, hash, limit, prune } from "@/lib/db";
export const runtime = "nodejs";
const schema = z.discriminatedUnion("operation", [
  z.object({ operation: z.literal("demo") }),
  z.object({ operation: z.literal("login"), key: z.string().min(20).max(100) }),
  z.object({
    operation: z.literal("create"),
    name: z.string().trim().min(2).max(80),
    setupKey: z.string().max(200),
  }),
  z.object({ operation: z.literal("logout") }),
]);
export async function POST(req: NextRequest) {
  try {
    const input = schema.parse(await body(req));
    limit("session-global", 100);
    prune();
    if (input.operation === "logout") {
      db()
        .prepare("DELETE FROM sessions WHERE hash=?")
        .run(hash(req.cookies.get("monitor_session")?.value || ""));
      const r = NextResponse.json({ ok: true });
      r.cookies.delete("monitor_session");
      return r;
    }
    let key = "",
      ownerKey: string | undefined;
    if (input.operation === "login") key = input.key;
    else {
      if (
        input.operation === "create" &&
        (!process.env.SETUP_KEY ||
          hash(input.setupKey) !== hash(process.env.SETUP_KEY))
      )
        throw new Error(
          "Private workspace setup key is invalid or not configured.",
        );
      const n = Number(
        (
          db().prepare("SELECT count(*) AS n FROM workspaces").get() as {
            n: number;
          }
        ).n,
      );
      if (n >= 500) throw new Error("Workspace capacity reached.");
      const created = createWorkspace(
        input.operation === "demo" ? "Northstar operations" : input.name,
        input.operation === "demo" ? "demo" : "private",
      );
      key = created.ownerKey;
      if (input.operation === "create") ownerKey = key;
    }
    const token = sessionFor(key);
    const response = NextResponse.json({ ok: true, ownerKey });
    response.cookies.set("monitor_session", token, {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.COOKIE_SECURE === "true",
      path: "/",
      maxAge: 7 * 86400,
    });
    response.headers.set("Cache-Control", "no-store");
    return response;
  } catch (e) {
    return failure(e);
  }
}
