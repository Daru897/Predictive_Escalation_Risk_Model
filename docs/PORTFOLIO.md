# Portfolio material

## Project summary

SignalDesk is a BPO escalation-monitoring application that turns CSV and Zendesk ticket metadata into an explainable review queue. It combines data onboarding, prioritized signals, intervention ownership, source freshness and exportable handover records.

## Resume bullet

Rebuilt a legacy escalation-model prototype into a full-stack BPO monitoring product using Next.js, TypeScript and SQLite; implemented validated CSV imports, encrypted Zendesk incremental sync with transactional checkpoints, explainable risk scoring, owner/viewer access and automated workflow tests.

## LinkedIn draft

I rebuilt my Predictive Escalation Risk Model project as SignalDesk, a product demo for BPO operations teams.

The biggest change was not adding a newer model. Auditing the original data revealed that the labels and tiny dataset did not support the predictive claims. I replaced the serving path with a transparent risk index and created a separate temporal evaluation workflow for genuine escalation outcomes.

The product now supports CSV column mapping, a customer-hosted Zendesk connector, ticket-level explanations, assigned interventions, queue snapshots and read-only sharing. The focus is helping a team lead move from a backlog to a clear next action.

This is a portfolio application and pilot foundation. Live customer validation and production model performance are still ahead.

## Showcase walkthrough

1. Open the demo and explain the four operational metrics.
2. Filter for high/critical tickets, open one, and show individual score contributions.
3. Assign an intervention owner and save a handover note.
4. Import the sample CSV: map, preview, commit and filter the source.
5. Show Trends and explain why counts are not proof of reduced escalations.
6. Show private onboarding and the Zendesk connection form without exposing credentials.
7. End with the case study: what the audit invalidated and what evidence is needed next.

No adoption, SLA reduction, customer ROI or model accuracy claim should be added until actually measured. Link the repository branch and the real deployment URL when one exists; a local preview is not a public demo link.
