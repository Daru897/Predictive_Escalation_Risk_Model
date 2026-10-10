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
export { scoreTicket } from "./scoring/engine";
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
