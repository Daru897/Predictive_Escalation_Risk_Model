import { z } from "zod";
export const fields = [
  "priority",
  "age_hours",
  "idle_hours",
  "transfers",
  "reopens",
  "response_delay_hours",
  "sla_hours_remaining",
] as const;
export const fieldLabels: Record<(typeof fields)[number], string> = {
  priority: "Ticket priority",
  age_hours: "Ticket age (hours)",
  idle_hours: "Time since update (hours)",
  transfers: "Transfer count",
  reopens: "Reopen count",
  response_delay_hours: "Response delay (hours)",
  sla_hours_remaining: "Hours until SLA deadline",
};
const ruleSchema = z
  .object({
    id: z.string().regex(/^[a-zA-Z0-9_-]{1,80}$/),
    name: z.string().trim().min(1).max(100),
    enabled: z.boolean(),
    field: z.enum(fields),
    operator: z.enum(["gte", "lt", "eq"]),
    value: z.union([
      z.number().finite().min(-100000).max(100000),
      z.enum(["low", "normal", "high", "urgent"]),
    ]),
    points: z.number().int().min(0).max(100),
    perUnit: z.boolean(),
    cap: z.number().int().min(0).max(100),
    queues: z.array(z.string().trim().min(1).max(100)).max(20),
    action: z.string().trim().min(1).max(400),
  })
  .strict()
  .superRefine((r, ctx) => {
    if (
      r.field === "priority" &&
      (r.operator !== "eq" || typeof r.value !== "string")
    )
      ctx.addIssue({
        code: "custom",
        message: "Priority rules need equals and a priority value.",
      });
    if (r.field !== "priority" && typeof r.value !== "number")
      ctx.addIssue({
        code: "custom",
        message: "Numeric signals need a numeric threshold.",
      });
    if (
      r.field !== "sla_hours_remaining" &&
      typeof r.value === "number" &&
      r.value < 0
    )
      ctx.addIssue({
        code: "custom",
        message: "Only SLA hours may have a negative threshold.",
      });
    if (r.perUnit && !["transfers", "reopens"].includes(r.field))
      ctx.addIssue({
        code: "custom",
        message:
          "Per-unit scoring is supported only for transfers and reopens.",
      });
  });
export const policySchema = z
  .object({
    schemaVersion: z.literal(1),
    name: z.string().trim().min(2).max(80),
    bands: z
      .object({
        watch: z.number().int().min(1).max(98),
        high: z.number().int().min(2).max(99),
        critical: z.number().int().min(3).max(100),
      })
      .strict(),
    rules: z.array(ruleSchema).max(30),
  })
  .strict()
  .superRefine((p, ctx) => {
    if (!(p.bands.watch < p.bands.high && p.bands.high < p.bands.critical))
      ctx.addIssue({
        code: "custom",
        message: "Risk thresholds must increase: Watch < High < Critical.",
      });
    if (new Set(p.rules.map((r) => r.id)).size !== p.rules.length)
      ctx.addIssue({ code: "custom", message: "Rule IDs must be unique." });
  });
export type Policy = z.infer<typeof policySchema>;
export type Rule = Policy["rules"][number];
const rule = (
  id: string,
  field: Rule["field"],
  operator: Rule["operator"],
  value: Rule["value"],
  points: number,
  name: string,
  action: string,
  perUnit = false,
  cap = 100,
): Rule => ({
  id,
  field,
  operator,
  value,
  points,
  name,
  action,
  perUnit,
  cap,
  enabled: true,
  queues: [],
});
export const DEFAULT_POLICY: Policy = {
  schemaVersion: 1,
  name: "Standard support operations",
  bands: { watch: 25, high: 45, critical: 70 },
  rules: [
    rule(
      "urgent",
      "priority",
      "eq",
      "urgent",
      25,
      "Urgent priority",
      "Confirm the impact and escalation route.",
    ),
    rule(
      "high",
      "priority",
      "eq",
      "high",
      15,
      "High priority",
      "Confirm the impact and escalation route.",
    ),
    rule(
      "age-8",
      "age_hours",
      "gte",
      8,
      5,
      "Ticket open at least 8 hours",
      "Review blockers and agree the next customer update.",
    ),
    rule(
      "age-24",
      "age_hours",
      "gte",
      24,
      12,
      "Ticket open at least 24 hours",
      "Review blockers and agree the next customer update.",
    ),
    rule(
      "age-72",
      "age_hours",
      "gte",
      72,
      20,
      "Ticket open at least 72 hours",
      "Review blockers and agree the next customer update.",
    ),
    rule(
      "idle-8",
      "idle_hours",
      "gte",
      8,
      8,
      "No ticket update for at least 8 hours",
      "Check whether progress or a customer update is overdue.",
    ),
    rule(
      "idle-24",
      "idle_hours",
      "gte",
      24,
      15,
      "No ticket update for at least 24 hours",
      "Check whether progress or a customer update is overdue.",
    ),
    rule(
      "transfers",
      "transfers",
      "gte",
      1,
      5,
      "Recorded transfers",
      "Assign a clear resolution owner.",
      true,
      15,
    ),
    rule(
      "reopens",
      "reopens",
      "gte",
      1,
      5,
      "Recorded reopens",
      "Check whether the previous resolution addressed the root cause.",
      true,
      10,
    ),
    rule(
      "response-2",
      "response_delay_hours",
      "gte",
      2,
      8,
      "Recorded response delay of at least 2 hours",
      "Review the response backlog.",
    ),
    rule(
      "response-8",
      "response_delay_hours",
      "gte",
      8,
      15,
      "Recorded response delay of at least 8 hours",
      "Review the response backlog.",
    ),
    rule(
      "sla-soon",
      "sla_hours_remaining",
      "lt",
      2,
      20,
      "Supplied SLA deadline within two hours",
      "Verify the contractual SLA and escalate to the duty lead.",
    ),
    rule(
      "sla-overdue",
      "sla_hours_remaining",
      "lt",
      0,
      30,
      "Supplied SLA deadline passed",
      "Verify the contractual SLA and escalate to the duty lead.",
    ),
  ],
};
