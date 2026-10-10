# SignalDesk product brief

## Audience and job

BPO team leads and operations managers need a short, explainable list of tickets to review during shift handover. Their job is to detect unresolved operational pressure, agree a next step, and keep ownership visible.

## First release

A polished portfolio demo and customer-hosted pilot. Bring CSV data without an integration; graduate to a Zendesk API-token connection when account access is available. One workspace represents one operational data boundary. Queue/group filters support multiple teams inside that boundary.

The core loop is import → inspect signals → assign intervention → monitor updates → export handover. Review status never changes source ticket status. A “resolved” intervention does not imply a solved customer issue.

## Decisions

- Explainable risk index first. The original training labels cannot establish escalation probabilities.
- One canonical validated ticket contract for both CSV and Zendesk.
- CSV and Zendesk IDs have separate namespaces; cross-source duplicates are not automatically merged.
- Missing optional fields are visible, add no points, and are not imputed as real events.
- Completed catch-up time, errors, cursor state, paused state and worker heartbeat make freshness inspectable.
- Customer-owned Zendesk token flow; no third-party Marketplace OAuth claims.
- Scoped owner/viewer keys for small private pilots. This is not identity-based enterprise RBAC.

## Pilot outcomes to measure

Time from a flag to first review, proportion of high-risk tickets with a named intervention owner, import rejection rate, source freshness, false-positive feedback from leads, and genuine escalation recall at a fixed daily review capacity. None has yet been measured with customers. Snapshot counts are not business-impact metrics.

## Next increments

1. Pilot with authorized real ticket metadata; validate missing fields and account API availability.
2. Individual SSO users, revocable invitations, audit identities and managed Postgres.
3. Per-client rule configuration and review-capacity thresholds, backed by operations feedback.
4. Zendesk metric/SLA and audit ingestion, including business-hour semantics, before claiming those signals.
5. Genuine escalation labels and robust chronological evaluation; external holdout, calibration and capacity-based threshold selection before model serving.
6. Zendesk-approved third-party integration strategy, distributed worker queues, retention policies and alert delivery with deduplication.
