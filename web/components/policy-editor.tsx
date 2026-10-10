"use client";
import { useState } from "react";
import {
  DEFAULT_POLICY,
  policySchema,
  fields,
  fieldLabels,
  type Policy,
  type Rule,
} from "@/lib/scoring/policy";
type Preview = {
  token: string;
  at: number;
  total: number;
  changed: number;
  beforeHigh: number;
  afterHigh: number;
  examples: {
    key: string;
    id: string;
    subject: string;
    before: number;
    after: number;
    beforeBand: string;
    afterBand: string;
  }[];
};
type Saved = { revision: number; config: Policy };
export default function PolicyEditor({
  active,
  history,
  owner,
  onPublished,
}: {
  active: Saved;
  history: (Saved & { created: number })[];
  owner: boolean;
  onPublished: () => Promise<void>;
}) {
  const [draft, setDraft] = useState<Policy>(() =>
      structuredClone(active.config),
    ),
    [preview, setPreview] = useState<Preview | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  function change(config: Policy) {
    setDraft(config);
    setPreview(null);
    setMessage("");
    setError("");
  }
  function updateRule(index: number, patch: Partial<Rule>) {
    change({
      ...draft,
      rules: draft.rules.map((r, i) => (i === index ? { ...r, ...patch } : r)),
    });
  }
  async function request(operation: string) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const config = policySchema.parse(draft);
      const response = await fetch("/api/workspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          operation,
          config,
          revision: active.revision,
          previewToken: preview?.token,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Policy request failed.");
      if (operation === "policy-preview") setPreview(data);
      else {
        setPreview(null);
        await onPublished();
      }
    } catch (e) {
      const parsed = policySchema.safeParse(draft);
      setError(
        parsed.success
          ? (e as Error).message
          : parsed.error.issues
              .map((i) => `${i.path.join(".")}: ${i.message}`)
              .join("; "),
      );
    } finally {
      setBusy(false);
    }
  }
  function exportDraft() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "signaldesk-policy.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="panel policy-editor">
      <div className="section-title">
        <div>
          <h2>Business rules</h2>
          <p>
            Active policy: {active.config.name} · Version {active.revision}
          </p>
        </div>
        <button className="button outline" onClick={exportDraft}>
          Export policy JSON
        </button>
      </div>
      <p>
        Changes stay in this draft until you preview and apply them. The highest
        matching contribution per signal is used, so overlapping tiers never
        double-count. Scores are capped at 100; solved and closed tickets always
        score zero.
      </p>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="success" role="status">
          {message}
        </p>
      )}
      <fieldset disabled={!owner || busy}>
        <legend className="sr-only">Scoring policy configuration</legend>
        <label>
          Policy name
          <input
            maxLength={80}
            value={draft.name}
            onChange={(e) => change({ ...draft, name: e.target.value })}
          />
        </label>
        <div className="three-columns band-inputs">
          {(["watch", "high", "critical"] as const).map((b) => (
            <label key={b}>
              {b[0].toUpperCase() + b.slice(1)} starts at
              <input
                type="number"
                min={1}
                max={100}
                value={draft.bands[b]}
                onChange={(e) =>
                  change({
                    ...draft,
                    bands: { ...draft.bands, [b]: Number(e.target.value) },
                  })
                }
              />
            </label>
          ))}
        </div>
        <div className="policy-rules">
          {draft.rules.map((r, i) => (
            <details key={r.id}>
              <summary>
                <span>{r.name}</span>
                <span className="badge">
                  {r.enabled
                    ? `${r.points} points${r.perUnit ? " / event" : ""}`
                    : "Disabled"}
                </span>
              </summary>
              <div className="policy-rule-body">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={r.enabled}
                    onChange={(e) =>
                      updateRule(i, { enabled: e.target.checked })
                    }
                  />
                  Enable this rule
                </label>
                <div className="two-columns">
                  <label>
                    Rule name
                    <input
                      value={r.name}
                      maxLength={100}
                      onChange={(e) => updateRule(i, { name: e.target.value })}
                    />
                  </label>
                  <label>
                    Signal
                    <select
                      value={r.field}
                      onChange={(e) => {
                        const field = e.target.value as Rule["field"];
                        updateRule(i, {
                          field,
                          operator: field === "priority" ? "eq" : "gte",
                          value: field === "priority" ? "urgent" : 1,
                          perUnit: false,
                        });
                      }}
                    >
                      {fields.map((f) => (
                        <option key={f} value={f}>
                          {fieldLabels[f]}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Comparison
                    <select
                      value={r.operator}
                      onChange={(e) =>
                        updateRule(i, {
                          operator: e.target.value as Rule["operator"],
                        })
                      }
                    >
                      {(r.field === "priority"
                        ? ["eq"]
                        : ["gte", "lt", "eq"]
                      ).map((o) => (
                        <option value={o} key={o}>
                          {o === "gte"
                            ? "At least"
                            : o === "lt"
                              ? "Less than"
                              : "Equals"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Threshold
                    {r.field === "priority" ? (
                      <select
                        value={r.value}
                        onChange={(e) =>
                          updateRule(i, {
                            value: e.target.value as Rule["value"],
                          })
                        }
                      >
                        {["low", "normal", "high", "urgent"].map((v) => (
                          <option key={v}>{v}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={r.value}
                        min={r.field === "sla_hours_remaining" ? -100000 : 0}
                        max={100000}
                        onChange={(e) =>
                          updateRule(i, { value: Number(e.target.value) })
                        }
                      />
                    )}
                  </label>
                  <label>
                    Points
                    <input
                      type="number"
                      min={0}
                      max={100}
                      value={r.points}
                      onChange={(e) =>
                        updateRule(i, { points: Number(e.target.value) })
                      }
                    />
                  </label>
                  <label>
                    Maximum contribution
                    <input
                      type="number"
                      min={0}
                      max={100}
                      value={r.cap}
                      onChange={(e) =>
                        updateRule(i, { cap: Number(e.target.value) })
                      }
                    />
                  </label>
                </div>
                {["transfers", "reopens"].includes(r.field) && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={r.perUnit}
                      onChange={(e) =>
                        updateRule(i, { perUnit: e.target.checked })
                      }
                    />
                    Multiply points by the event count, up to the maximum
                  </label>
                )}
                <label>
                  Apply only to these queues (one exact name per line; blank
                  means all)
                  <textarea
                    rows={2}
                    value={r.queues.join("\n")}
                    onChange={(e) =>
                      updateRule(i, {
                        queues: e.target.value
                          ? e.target.value.split("\n")
                          : [],
                      })
                    }
                  />
                </label>
                <label>
                  Suggested next action
                  <textarea
                    rows={2}
                    maxLength={400}
                    value={r.action}
                    onChange={(e) => updateRule(i, { action: e.target.value })}
                  />
                </label>
                <button
                  type="button"
                  className="button outline"
                  onClick={() =>
                    change({
                      ...draft,
                      rules: draft.rules.filter((_, n) => n !== i),
                    })
                  }
                >
                  Remove rule
                </button>
              </div>
            </details>
          ))}
        </div>
        <div className="button-row">
          <button
            className="button outline"
            disabled={draft.rules.length >= 30}
            onClick={() =>
              change({
                ...draft,
                rules: [
                  ...draft.rules,
                  {
                    id: crypto.randomUUID(),
                    name: "New rule",
                    enabled: true,
                    field: "age_hours",
                    operator: "gte",
                    value: 24,
                    points: 10,
                    perUnit: false,
                    cap: 100,
                    queues: [],
                    action: "Review this ticket with the duty lead.",
                  },
                ],
              })
            }
          >
            Add rule
          </button>
          <button
            className="button outline"
            onClick={() => change(structuredClone(DEFAULT_POLICY))}
          >
            Load standard defaults
          </button>
          <label className="button outline import-policy">
            Import policy JSON
            <input
              aria-label="Import policy JSON"
              type="file"
              accept=".json,application/json"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                try {
                  if (f.size > 50000)
                    throw new Error("Policy JSON must be under 50 KB.");
                  change(policySchema.parse(JSON.parse(await f.text())));
                } catch {
                  setError(
                    "Invalid policy file. Use an exported SignalDesk policy under 50 KB.",
                  );
                }
                e.target.value = "";
              }}
            />
          </label>
        </div>
        <button
          className="button dark preview-policy"
          onClick={() => request("policy-preview")}
        >
          Preview policy impact
        </button>
        {preview && (
          <div className="policy-impact" role="status">
            <h3>
              Preview: {preview.changed} of {preview.total} tickets change
            </h3>
            <p>
              High + critical:{" "}
              <strong>
                {preview.beforeHigh} → {preview.afterHigh}
              </strong>
              . Evaluated at {new Date(preview.at).toLocaleTimeString()}.
              Preview expires in five minutes; changes to ticket data require a
              new preview.
            </p>
            {preview.examples.map((r) => (
              <p key={r.key}>
                #{r.id} · {r.subject}
                <br />
                <strong>
                  {r.before} / {r.beforeBand} → {r.after} / {r.afterBand}
                </strong>
              </p>
            ))}
            {!draft.rules.some(
              (r) => r.enabled && r.points > 0 && r.cap > 0,
            ) && (
              <p className="notice">
                All rules are disabled or removed. Every ticket will score zero.
              </p>
            )}
            <button
              className="button dark"
              onClick={() => request("policy-publish")}
            >
              Apply policy version {active.revision + 1}
            </button>
          </div>
        )}
      </fieldset>
      {!owner && (
        <p className="notice">
          You have read-only access. A workspace owner can change these rules.
        </p>
      )}
      {history.length > 0 && (
        <div className="policy-history">
          <h3>Recent policy versions</h3>
          <p>
            Restoring loads an earlier configuration into the draft. Preview it,
            then apply it as a new version.
          </p>
          {history.map((h) => (
            <div key={h.revision}>
              <span>
                Version {h.revision} · {h.config.name} ·{" "}
                {new Date(h.created).toLocaleString()}
              </span>
              <button
                className="button outline"
                disabled={!owner || busy}
                onClick={() => {
                  change(structuredClone(h.config));
                  setMessage(
                    `Version ${h.revision} loaded into the draft. Preview before applying.`,
                  );
                }}
              >
                Load version {h.revision}
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
