import { test, expect } from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { once } from "node:events";
test("standalone restart retains ticket interventions and access sessions", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "signaldesk-restart-"));
  let child: ChildProcess | undefined;
  const origin = "http://127.0.0.1:3102";
  async function start() {
    child = spawn(process.execPath, [".next/standalone/server.js"], {
      env: {
        ...process.env,
        PORT: "3102",
        HOSTNAME: "127.0.0.1",
        APP_ORIGIN: origin,
        DATABASE_PATH: path.join(dir, "db.sqlite"),
      },
      stdio: "ignore",
    });
    for (let i = 0; i < 100; i++) {
      try {
        if ((await fetch(origin + "/api/health")).ok) return;
      } catch {}
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error("Server did not start");
  }
  async function stop() {
    if (child && child.exitCode === null) {
      const ended = once(child, "exit");
      child.kill("SIGTERM");
      await ended;
    }
  }
  try {
    await start();
    const response = await fetch(origin + "/api/session", {
      method: "POST",
      headers: { Origin: origin, "Content-Type": "application/json" },
      body: JSON.stringify({ operation: "demo" }),
    });
    expect(response.ok).toBe(true);
    const cookie = response.headers.get("set-cookie")!.split(";")[0];
    expect(
      (
        await fetch(origin + "/api/workspace", {
          method: "POST",
          headers: {
            Origin: origin,
            "Content-Type": "application/json",
            Cookie: cookie,
          },
          body: JSON.stringify({
            operation: "action",
            key: "demo:1001",
            status: "investigating",
            owner: "Persistence lead",
            note: "Survives a restart",
          }),
        })
      ).ok,
    ).toBe(true);
    await stop();
    await start();
    const state = await (
      await fetch(origin + "/api/workspace", { headers: { Cookie: cookie } })
    ).json();
    expect(
      state.tickets.find((t: { key: string }) => t.key === "demo:1001").action
        .note,
    ).toBe("Survives a restart");
  } finally {
    await stop();
    await rm(dir, { recursive: true, force: true });
  }
});
