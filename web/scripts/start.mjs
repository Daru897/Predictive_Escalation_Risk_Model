import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
for (const f of [".env.local", ".env"])
  if (existsSync(f)) process.loadEnvFile(f);
const env = {
  ...process.env,
  APP_ORIGIN:
    process.env.APP_ORIGIN || `http://127.0.0.1:${process.env.PORT || "3001"}`,
  DATABASE_PATH: resolve(process.env.DATABASE_PATH || "data/monitor.sqlite"),
  HOSTNAME: process.env.HOSTNAME || "127.0.0.1",
  PORT: process.env.PORT || "3001",
};
const server = spawn(process.execPath, [".next/standalone/server.js"], {
  stdio: "inherit",
  env,
});
const worker = spawn(
  process.execPath,
  ["--import", "tsx", "scripts/worker.ts"],
  { stdio: "inherit", env },
);
let stopping = false;
function stop(code) {
  if (stopping) return;
  stopping = true;
  server.kill("SIGTERM");
  worker.kill("SIGTERM");
  process.exitCode = code;
}
for (const p of [server, worker]) {
  p.on("error", () => stop(1));
  p.on("exit", (code) => stop(code || 0));
}
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => stop(0));
