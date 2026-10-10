import { z } from "zod";
export const date = z
  .string()
  .datetime({ offset: true })
  .refine(
    (v) => Date.parse(v) <= Date.now() + 300000,
    "Date cannot be in the future",
  );
export const ticketSchema = z
  .object({
    id: z.string().trim().min(1).max(100),
    subject: z.string().trim().min(1).max(300),
    status: z.enum(["new", "open", "pending", "hold", "solved", "closed"]),
    priority: z.enum(["low", "normal", "high", "urgent"]),
    created_at: date.transform((v) => new Date(v).toISOString()),
    updated_at: date.transform((v) => new Date(v).toISOString()),
    group: z.string().trim().max(100).default("Unassigned"),
    assignee: z.string().trim().max(100).default("Unassigned"),
    transfers: z.number().int().min(0).max(1000).nullable().default(null),
    reopens: z.number().int().min(0).max(1000).nullable().default(null),
    response_delay_hours: z
      .number()
      .min(0)
      .max(100000)
      .nullable()
      .default(null),
    sla_due_at: z.string().datetime({ offset: true }).nullable().default(null),
  })
  .refine(
    (t) => Date.parse(t.updated_at) >= Date.parse(t.created_at),
    "updated_at must not precede created_at",
  );
export type Ticket = z.infer<typeof ticketSchema> & {
  source: "csv" | "zendesk" | "demo";
  key: string;
};
export type Action = {
  status: "unreviewed" | "investigating" | "monitoring" | "resolved";
  owner: string;
  note: string;
  updatedAt: string;
};
export function scoreTicket(t: Ticket, now = Date.now()) {
  const age = Math.max(0, (now - Date.parse(t.created_at)) / 3600000),
    idle = Math.max(0, (now - Date.parse(t.updated_at)) / 3600000);
  const factors: { label: string; points: number; action: string }[] = [];
  const add = (points: number, label: string, action: string) => {
    if (points) factors.push({ label, points, action });
  };
  const closed = ["solved", "closed"].includes(t.status);
  if (!closed) {
    add(
      t.priority === "urgent" ? 25 : t.priority === "high" ? 15 : 0,
      `${t.priority} priority`,
      "Confirm the impact and escalation route.",
    );
    add(
      age >= 72 ? 20 : age >= 24 ? 12 : age >= 8 ? 5 : 0,
      `Ticket open for ${Math.floor(age)} hours`,
      "Review blockers and agree the next customer update.",
    );
    add(
      idle >= 24 ? 15 : idle >= 8 ? 8 : 0,
      `No ticket update for ${Math.floor(idle)} hours`,
      "Check whether progress or a customer update is overdue.",
    );
    add(
      t.transfers === null ? 0 : Math.min(15, t.transfers * 5),
      `${t.transfers} recorded transfers`,
      "Assign a clear resolution owner.",
    );
    add(
      t.reopens === null ? 0 : Math.min(10, t.reopens * 5),
      `${t.reopens} recorded reopens`,
      "Check whether the previous resolution addressed the root cause.",
    );
    add(
      t.response_delay_hours === null
        ? 0
        : t.response_delay_hours >= 8
          ? 15
          : t.response_delay_hours >= 2
            ? 8
            : 0,
      "Recorded response delay",
      "Review the response backlog.",
    );
    if (t.sla_due_at) {
      const hours = (Date.parse(t.sla_due_at) - now) / 3600000;
      add(
        hours < 0 ? 30 : hours < 2 ? 20 : 0,
        hours < 0
          ? "Supplied SLA deadline passed"
          : "Supplied SLA deadline within two hours",
        "Verify the contractual SLA and escalate to the duty lead.",
      );
    }
  }
  const score = Math.min(
    100,
    factors.reduce((s, f) => s + f.points, 0),
  );
  const missing = [
    "transfers",
    "reopens",
    "response_delay_hours",
    "sla_due_at",
  ].filter((k) => t[k as keyof Ticket] === null);
  return {
    score,
    band: closed
      ? "Closed"
      : score >= 70
        ? "Critical"
        : score >= 45
          ? "High"
          : score >= 25
            ? "Watch"
            : "Low",
    factors: factors.sort((a, b) => b.points - a.points),
    missing,
    coverage: Math.round(((8 - missing.length) / 8) * 100),
    ageHours: age,
    idleHours: idle,
  };
}
export function sampleTickets(now = Date.now()): Ticket[] {
  const subjects = [
    "Payment reversed but refund missing",
    "Account locked before renewal",
    "Shipment marked delivered, not received",
    "Repeated invoice mismatch",
    "Cancellation confirmation pending",
    "Cannot update billing address",
    "Replacement device not dispatched",
    "Refund timeline clarification",
    "Service unavailable after upgrade",
    "Duplicate subscription charge",
    "Password reset link expired",
    "Update contact preferences",
  ];
  return subjects.map((subject, i) => ({
    key: `demo:${1001 + i}`,
    source: "demo",
    id: String(1001 + i),
    subject,
    status: i === 11 ? "solved" : i % 4 === 0 ? "pending" : "open",
    priority: i % 4 === 0 ? "urgent" : i % 3 === 0 ? "high" : "normal",
    created_at: new Date(now - ((i % 5) + 1) * 18 * 3600000).toISOString(),
    updated_at: new Date(now - (i % 4) * 9 * 3600000).toISOString(),
    group: i % 2 ? "Commerce support" : "Account services",
    assignee: ["Maya", "Ravi", "Alex"][i % 3],
    transfers: i % 4,
    reopens: i % 3,
    response_delay_hours: i % 5,
    sla_due_at:
      i % 3 === 0
        ? new Date(now + (i % 2 ? -2 : 1) * 3600000).toISOString()
        : null,
  }));
}
