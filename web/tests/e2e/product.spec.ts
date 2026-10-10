import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import path from "node:path";
test("demo: prioritize, inspect, intervene, import and retain state", async ({
  page,
}) => {
  await page.goto("/monitor?demo=1");
  await expect(
    page.getByRole("heading", { name: "Ahead of the escalation." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Review ticket 1001" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByLabel("Intervention owner").fill("Duty lead");
  await page
    .getByLabel("Next step / handover note")
    .fill("Verify refund and contact customer by end of shift.");
  await page.getByRole("button", { name: "Save intervention" }).click();
  await expect(
    page.getByText("Intervention saved. The source ticket was not changed."),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByText("Duty lead", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Data sources", exact: true }).click();
  await page
    .getByLabel("Upload ticket CSV")
    .setInputFiles("public/sample-tickets.csv");
  await page.getByRole("button", { name: "Validate & preview" }).click();
  await expect(
    page.getByRole("heading", { name: "3 tickets ready" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Import 3 tickets" }).click();
  await page.getByRole("button", { name: /Overview/ }).click();
  await page.getByLabel("Search tickets").fill("Refund delayed after return");
  await expect(
    page.getByRole("button", {
      name: "Refund delayed after return",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Trends", exact: true }).click();
  await expect(
    page.getByLabel("High risk ticket counts over snapshots"),
  ).toBeVisible();
});
test("private onboarding, viewer restrictions, source isolation and origin validation", async ({
  browser,
}) => {
  const owner = await browser.newContext(),
    viewer = await browser.newContext();
  const p = await owner.newPage();
  await p.goto("/monitor");
  await p.getByLabel("Workspace name").fill("Acme support");
  await p.getByLabel("Deployment setup key").fill("test-only-setup-key");
  await p
    .getByRole("button", { name: "Create workspace", exact: true })
    .click();
  await expect(
    p.getByRole("heading", { name: "Ahead of the escalation." }),
  ).toBeVisible();
  await expect(p.getByText("No tickets match this view.")).toBeVisible();
  await p.getByRole("button", { name: "I saved the key" }).click();
  await p.getByRole("button", { name: "Workspace", exact: true }).click();
  await p.getByRole("button", { name: "Create viewer key" }).click();
  const key = await p.locator(".key-panel code").innerText();
  const v = await viewer.newPage();
  await v.goto("/monitor");
  await v.getByLabel("Workspace access key").fill(key);
  await v.getByRole("button", { name: "Open workspace", exact: true }).click();
  await expect(
    v.getByRole("heading", { name: "Ahead of the escalation." }),
  ).toBeVisible();
  const denied = await viewer.request.post("/api/workspace", {
    headers: { Origin: "http://127.0.0.1:3101" },
    data: { operation: "viewer" },
  });
  expect(denied.status()).toBe(400);
  expect((await denied.json()).error).toContain("Owner access");
  const cross = await owner.request.post("/api/workspace", {
    headers: { Origin: "https://evil.example" },
    data: { operation: "viewer" },
  });
  expect(cross.status()).toBe(400);
  await p.getByRole("button", { name: "Revoke all viewers" }).click();
  await p.getByRole("button", { name: "Confirm", exact: true }).click();
  await expect(p.getByText("Workspace updated.")).toBeVisible();
  expect((await viewer.request.get("/api/workspace")).status()).toBe(400);
  await owner.close();
  await viewer.close();
});
test("CSV preview is nonmutating, bad rows reject atomically and tenants stay isolated", async ({
  browser,
}) => {
  const a = await browser.newContext(),
    b = await browser.newContext();
  for (const c of [a, b])
    expect(
      (
        await c.request.post("/api/session", {
          headers: { Origin: "http://127.0.0.1:3101" },
          data: { operation: "demo" },
        })
      ).ok(),
    ).toBe(true);
  const before = await (await a.request.get("/api/workspace")).json();
  const bad =
    "id,subject,status,priority,created_at,updated_at\nx,Valid,open,normal,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\ny,Invalid,nope,normal,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z";
  const mapping = Object.fromEntries(
    ["id", "subject", "status", "priority", "created_at", "updated_at"].map(
      (x) => [x, x],
    ),
  );
  const response = await a.request.post("/api/workspace", {
    headers: { Origin: "http://127.0.0.1:3101" },
    data: { operation: "import", content: bad, mapping, preview: false },
  });
  expect(response.status()).toBe(400);
  expect(
    (await (await a.request.get("/api/workspace")).json()).tickets.length,
  ).toBe(before.tickets.length);
  await a.request.post("/api/workspace", {
    headers: { Origin: "http://127.0.0.1:3101" },
    data: {
      operation: "action",
      key: "demo:1001",
      status: "investigating",
      owner: "Private owner",
      note: "Private note",
    },
  });
  expect(
    JSON.stringify(await (await b.request.get("/api/workspace")).json()),
  ).not.toContain("Private note");
  await a.close();
  await b.close();
});
test("public and workspace pages are accessible at desktop and mobile widths", async ({
  page,
}) => {
  for (const url of ["/", "/case-study", "/monitor"]) {
    await page.goto(url);
    await expect(page.locator("main")).toBeVisible();
    if (url === "/monitor")
      await expect(
        page.getByRole("heading", { name: "Start with a clearer queue." }),
      ).toBeVisible();
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => ({
          target: n.target,
          summary: n.failureSummary,
        })),
      })),
    ).toEqual([]);
  }
  await page.goto("/monitor?demo=1");
  await expect(
    page.getByRole("heading", { name: "Ahead of the escalation." }),
  ).toBeVisible();
  for (const view of ["Overview", "Data sources", "Trends", "Workspace"]) {
    await page
      .getByRole("button", {
        name: view === "Overview" ? /Overview/ : view,
        exact: view !== "Overview",
      })
      .click();
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => ({
          target: n.target,
          summary: n.failureSummary,
        })),
      })),
    ).toEqual([]);
  }
  await page.getByRole("button", { name: /Overview/ }).click();
  await page.screenshot({
    path: path.resolve("../docs/dashboard-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  for (const url of ["/", "/monitor"]) {
    await page.goto(url);
    await page.waitForTimeout(300);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => ({
          target: n.target,
          summary: n.failureSummary,
        })),
      })),
    ).toEqual([]);
  }
  await page.screenshot({
    path: path.resolve("../docs/dashboard-mobile.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", {
      name: "Payment reversed but refund missing",
      exact: true,
    })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        summary: n.failureSummary,
      })),
    })),
  ).toEqual([]);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
});
