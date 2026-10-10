import type { Ticket } from "../domain";
import { DEFAULT_POLICY, type Policy, type Rule } from "./policy";
/** Pure evaluator: no storage, network, expression execution or side effects. */
export function scoreTicket(
  t: Ticket,
  now = Date.now(),
  policy: Policy = DEFAULT_POLICY,
) {
  const age = Math.max(0, (now - Date.parse(t.created_at)) / 3600000),
    idle = Math.max(0, (now - Date.parse(t.updated_at)) / 3600000);
  const values: Record<Rule["field"], number | string | null> = {
    priority: t.priority,
    age_hours: age,
    idle_hours: idle,
    transfers: t.transfers,
    reopens: t.reopens,
    response_delay_hours: t.response_delay_hours,
    sla_hours_remaining: t.sla_due_at
      ? (Date.parse(t.sla_due_at) - now) / 3600000
      : null,
  };
  const chosen = new Map<
    string,
    {
      ruleId: string;
      field: Rule["field"];
      label: string;
      points: number;
      action: string;
      observed: number | string;
    }
  >();
  const closed = ["solved", "closed"].includes(t.status);
  if (!closed)
    for (const r of policy.rules) {
      if (!r.enabled || (r.queues.length && !r.queues.includes(t.group)))
        continue;
      const value = values[r.field];
      if (value === null) continue;
      const matches =
        r.operator === "eq"
          ? value === r.value
          : typeof value === "number" &&
            typeof r.value === "number" &&
            (r.operator === "gte" ? value >= r.value : value < r.value);
      if (!matches) continue;
      const points = Math.min(
        r.cap,
        r.perUnit ? Number(value) * r.points : r.points,
      );
      // Tiers on the same signal never stack. Equal points retain policy order.
      if (points > 0 && points > (chosen.get(r.field)?.points || 0))
        chosen.set(r.field, {
          ruleId: r.id,
          field: r.field,
          label: r.name,
          points,
          action: r.action,
          observed:
            typeof value === "number" ? Math.round(value * 100) / 100 : value,
        });
    }
  const factors = [...chosen.values()].sort((a, b) => b.points - a.points),
    score = Math.min(
      100,
      factors.reduce((n, r) => n + r.points, 0),
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
      : score >= policy.bands.critical
        ? "Critical"
        : score >= policy.bands.high
          ? "High"
          : score >= policy.bands.watch
            ? "Watch"
            : "Low",
    factors,
    missing,
    coverage: Math.round(((8 - missing.length) / 8) * 100),
    ageHours: age,
    idleHours: idle,
  };
}
