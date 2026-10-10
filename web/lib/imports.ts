import { parse } from "csv-parse/sync";
import { ticketSchema, type Ticket } from "./domain";
import { columns, required } from "./fields";
export function inspectCSV(content: string) {
  if (
    Buffer.byteLength(content) > 1024 * 1024 ||
    !content.trim() ||
    content.includes("\0")
  )
    throw new Error("Use a non-empty UTF-8 CSV under 1 MB.");
  const rows: string[][] = parse(content, {
    bom: true,
    skip_empty_lines: true,
    trim: true,
    max_record_size: 20000,
  });
  if (rows.length < 2 || rows.length > 2001)
    throw new Error("Upload between 1 and 2,000 ticket rows.");
  const headers = rows.shift()!;
  if (
    headers.length > 40 ||
    headers.some((x) => !x) ||
    new Set(headers).size !== headers.length
  )
    throw new Error(
      "Column names must be unique and non-empty (up to 40 columns).",
    );
  return { headers, rows };
}
export function importCSV(content: string, mapping: Record<string, string>) {
  const { headers, rows } = inspectCSV(content);
  if (required.some((k) => !mapping[k] || !headers.includes(mapping[k])))
    throw new Error("Map every required column before previewing.");
  const used = Object.values(mapping).filter(Boolean);
  if (new Set(used).size !== used.length)
    throw new Error("Map each source column only once.");
  const tickets: Ticket[] = [],
    errors: string[] = [],
    seen = new Set();
  rows.forEach((row, i) => {
    const raw: Record<string, unknown> = {};
    for (const key of columns) {
      const val = row[headers.indexOf(mapping[key])];
      if (val === undefined || val === "") continue;
      raw[key] = ["transfers", "reopens", "response_delay_hours"].includes(key)
        ? Number(val)
        : val;
    }
    if (typeof raw.status === "string") raw.status = raw.status.toLowerCase();
    if (typeof raw.priority === "string")
      raw.priority = raw.priority.toLowerCase();
    const result = ticketSchema.safeParse(raw);
    if (!result.success) {
      errors.push(
        `Row ${i + 2}: ${result.error.issues.map((x) => `${x.path.join(".")}: ${x.message}`).join("; ")}`,
      );
      return;
    }
    if (seen.has(result.data.id)) {
      errors.push(`Row ${i + 2}: duplicate ticket ID ${result.data.id}`);
      return;
    }
    seen.add(result.data.id);
    tickets.push({
      ...result.data,
      source: "csv",
      key: `csv:${result.data.id}`,
    });
  });
  return {
    tickets,
    errors: errors.slice(0, 30),
    errorCount: errors.length,
    total: rows.length,
  };
}
export function csvCell(value: unknown) {
  let s = String(value ?? "");
  if (/^[\s]*[=+@\-\t\r]/.test(s)) s = "'" + s;
  return '"' + s.replaceAll('"', '""') + '"';
}
