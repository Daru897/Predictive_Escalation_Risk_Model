# Risk index and candidate-model card

## Serving behavior

The product serves a deterministic operational score from 0–100 through a versioned workspace policy. The original `risk-index-v1` configuration is preserved as policy version 0. Owners can edit conditions and contributions through [Business rules](BUSINESS-RULES.md), preview their impact, and publish a new revision. It does not serve a machine-learned probability. Optional missing fields add zero, but their absence is shown. Therefore low score does not prove low escalation likelihood. Closed/solved tickets score zero. Rules and contributions are visible in Trends and each ticket detail.

Priority, age, inactivity, supplied transfer/reopen counts, supplied response delay and supplied SLA deadline drive the score. Age/inactivity are recomputed on each workspace read. They are wall-clock values, not business-hours calculations. Agent experience, geography, customer monetary value, demographics, text sentiment and LLM explanations are not used. Recommendations are deterministic workflow prompts for human review.

## Why the old models are not served

See the repository audit. The old target construction, feature timing and 40-row dataset cannot establish genuine prospective escalation performance. The new runtime never loads `.joblib` or arbitrary uploaded model files.

## Offline model workflow

Use Python 3.12 in an isolated environment:

```sh
python -m venv .venv
.venv/bin/pip install -r research/requirements.txt
.venv/bin/python research/train_candidate.py /path/to/labeled-snapshots.csv
.venv/bin/python -m unittest discover -s research/tests -v
```

Required CSV columns: `ticket_id,observed_at,created_at,updated_at,observation_end,escalated_at,status,priority`. Optional numeric columns: `transfers,reopens,response_delay_hours`; optional timestamp: `sla_due_at`. Timestamps must include a timezone. Blank escalation time means no observed escalation during the observation window, **not** a missing label; only supply blank after verifying absence of an event. Capture features as they were at `observed_at`, never reconstruct using outcome-time values without a defensible event log.

Target: a genuine escalation event within 24 hours after `observed_at`. The full 24-hour outcome window must be observable. Existing escalations, solved/closed snapshots, duplicate tickets, reversed timestamps, nonfinite counts and incomplete windows are rejected. At least 200 distinct tickets are required as a software floor, not a statistical sufficiency claim.

The final 30% of observed-time rows form a holdout; training rows whose outcome window overlaps the cutoff are excluded with a 24-hour embargo. A standardized logistic regression is fit only to the training partition. Missingness indicators are explicit. Numeric clipping and feature order are in `research/train_candidate.py`.

Reports include ROC-AUC, average precision, prevalence, Brier score against a train-prevalence baseline, precision/recall at a fixed 0.5 threshold, precision at top 10% review capacity, confusion matrix, calibration bins, row counts and dataset hash. The script writes a plain-JSON candidate and evaluation report to ignored `research/output/`. It does not deploy or activate a model. No sensitive row examples are written to the report.

## Evidence and release gate

The pipeline smoke test uses deliberately synthetic fixtures only to verify execution, labels, split boundaries and artifact writing. Synthetic metrics are not product accuracy evidence. No genuine outcome dataset was supplied and no customer model has been evaluated.

Before predictive serving: agree the target with operations, validate event provenance and censoring, use larger representative data, inspect subgroup performance and missingness shift, select thresholds on a separate validation set, evaluate a final untouched temporal cohort, calibrate probabilities if needed, review outcomes with leads, version the feature contract and add runtime parity tests. The current single holdout is a starting evaluation, not a complete model-governance process.
