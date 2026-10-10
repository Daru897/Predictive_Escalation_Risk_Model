# Original repository audit — 10 October 2026

The original experiment was preserved, but is not imported or executed by SignalDesk.

| Finding | Evidence | Product response |
| --- | --- | --- |
| Only 40 training rows (25 positive, 15 negative) | `etl/data/training_dataset.csv` | No generalization claims; no legacy model serving |
| “Escalation” can mean simply not closed | `models/train_baseline.py`, fallback labels from status | Require a genuine escalation event and explicit prediction horizon |
| Time split sorts ticket IDs | `time_based_split` in the same script | New research uses observed timestamps, unique tickets and a 24h embargo |
| Reported ROC-AUC 0.4167 | `models/artifacts/model_comparison.csv` and metadata | Do not promote accuracy or ROI figures from old README |
| “Transfers” count generic Updated events | `etl/feature_engineering.py` | Do not infer transfers from Zendesk updates |
| Response latency is average comment-event spacing | Same feature script | Distinguish supplied response delay from inactivity |
| Ticket age computed relative to current processing time | Same feature script | Research requires timestamped snapshots before outcomes |
| Sample data randomly generates customers, tickets, events | `etl/load_sample_data.py` | Label demo data fictional; never treat it as real trial evidence |
| Hundreds of pinned environment packages in misspelled UTF-16 dependency file | `requirments.txt` | Separate minimal Node runtime and optional research dependencies |

No original model binary was deserialized. Historical model artifacts and explainability files are retained solely as experiment records. The archived README contains historical claims that do not describe the current product.
