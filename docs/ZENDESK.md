# Zendesk connector

## Account setup

Use a customer-owned Zendesk Support account with API-token authentication enabled and a user permitted to access incremental ticket exports. In a private workspace, open Data sources and enter the subdomain (without `.zendesk.com`), email, API token and lookback days (1–365). This uses `email/token` basic authentication over HTTPS. The app does not write tickets or send messages.

The server encryption key must be configured first. Tokens are AES-256-GCM encrypted at rest with workspace-bound authenticated data, never returned by the API, and never placed in URLs or logs. The request host is built from a strictly validated Zendesk subdomain; redirects are rejected.

## Polling semantics

The first request uses the cursor incremental-ticket endpoint with a start time. Subsequent requests use the saved `after_cursor`. One page (requested size 100) is processed per worker iteration; catch-up pages are scheduled 15 seconds apart. After end-of-stream, polling waits approximately two minutes. Zendesk excludes the latest minute; this is near-real-time, not streaming.

Tickets and cursor advancement commit in a single transaction. Malformed pages do not advance the cursor. Namespaced IDs prevent duplicates. Older source timestamps cannot overwrite newer data. Deleted tickets remove stored tickets and their intervention notes. HTTP 429 honors Retry-After with bounded delay; transient failures retain the cursor. Credentials/access errors are visible and retry at the normal backoff until paused or disconnected.

The worker claims a short database lease to avoid overlapping work. Disconnect/pause changes are rechecked after network completion; an in-flight result cannot repopulate a disconnected source. Reconnect after disconnect requires clearing retained Zendesk tickets first, preventing mixed-account data. Updating an existing same-account connection preserves its cursor.

Last completed catch-up is only updated at end-of-stream. During the initial import it remains empty even when some pages exist. A completed catch-up older than five minutes is marked potentially stale. A missing worker heartbeat over two minutes prompts administrator attention.

## Data boundaries

Imported: ID, subject, status, priority, created/updated timestamps, group ID and assignee ID. No comments, attachments or requester records. Group and agent IDs are displayed as labels, not resolved names. Unknown or unsupported ticket data causes the page to fail atomically rather than silently skip tickets.

Not supplied by this endpoint: genuine transfers, reopens, response delay, business-hour SLA deadlines or escalation outcomes. These fields remain null. `updated_at` is used only as time since any ticket update, never described as response latency. The initial lookback is a changed-since window, so older unchanged tickets may be missing. Deleted events must be ingested within Zendesk's retention window; a long-disconnected replica may need a rebuild.

## Integration availability

Zendesk now restricts incremental/bulk export APIs for third-party integrations and Marketplace apps; direct customer accounts retain access. This release is a **customer-hosted API-token connector**, not a general Marketplace integration. Verify API eligibility and production behavior in the customer's account before a pilot. Live credentials were not available for development, so provider behavior is tested using mocked responses.

Sources: [Incremental export workflow](https://developer.zendesk.com/documentation/api-basics/working-with-data/using-the-incremental-export-api/), [Incremental exports API](https://developer.zendesk.com/api-reference/ticketing/ticket-management/incremental_exports/).
