import { tick } from "../lib/zendesk";
import { prune, db } from "../lib/db";
async function run() {
  try {
    prune();
    db()
      .prepare(
        "INSERT INTO runtime_status VALUES('worker',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      )
      .run(Date.now());
    await tick();
  } catch {
    console.error("Sync worker iteration failed; retrying.");
  }
}
awaitLoop();
async function awaitLoop() {
  await run();
  setTimeout(awaitLoop, 15000);
}
