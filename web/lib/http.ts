import { NextRequest, NextResponse } from "next/server";
import { session } from "./db";
export function identity(req: NextRequest) {
  const s = session(req.cookies.get("monitor_session")?.value || "");
  if (!s) throw new Error("Sign in to a workspace first.");
  return s;
}
export function owner(req: NextRequest) {
  const s = identity(req);
  if (s.role !== "owner") throw new Error("Owner access is required.");
  return s;
}
export function sameOrigin(req: NextRequest) {
  const expected =
    process.env.APP_ORIGIN ||
    `${req.nextUrl.protocol}//${req.headers.get("host")}`;
  if (req.headers.get("origin") !== expected)
    throw new Error("Request origin is not allowed.");
}
export async function body(req: NextRequest) {
  sameOrigin(req);
  if (!req.headers.get("content-type")?.startsWith("application/json"))
    throw new Error("JSON is required.");
  const reader = req.body?.getReader();
  if (!reader) throw new Error("Missing request.");
  let size = 0;
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    if (size > 1200000) {
      await reader.cancel();
      throw new Error("Request exceeds 1 MB upload allowance.");
    }
    chunks.push(value);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}
export function failure(error: unknown) {
  const message = error instanceof Error ? error.message : "Request failed.";
  const safe = /SQLITE|database|constraint|Unexpected token|JSON/.test(message)
    ? "Request could not be processed. Check your input."
    : message;
  return NextResponse.json(
    { error: safe.slice(0, 350) },
    { status: 400, headers: { "Cache-Control": "no-store" } },
  );
}
