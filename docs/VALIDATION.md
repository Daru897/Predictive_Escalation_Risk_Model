# Validation record — 10 October 2026

| Check | Result |
| --- | --- |
| Node 24 production standalone build | Passed |
| TypeScript and ESLint | Passed |
| Core domain/storage/import/connector/policy tests | 18 passed |
| Browser workflow/security/accessibility tests | 5 passed |
| Standalone restart persistence scenario | Passed |
| Python 3.12 candidate pipeline contract tests | 3 passed; synthetic smoke fixture only |
| Production Node dependency audit | Zero known advisories |
| Live Zendesk credentials and account permissions | Not available; not verified |
| Cloud deployment | Not performed |
| Docker execution | Not run; Docker not installed |
| Customer model accuracy or operational ROI | Not evaluated; no genuine labeled cohort supplied |

Core checks cover deterministic scoring, missing values, UTC normalization, CSV mapping and errors, spreadsheet formula escaping, atomic upserts, stale-update rejection, isolation, encrypted credential binding, cursor checkpoints, duplicate pages, source deletions, 429/403 failures, disconnected in-flight requests and viewer revocation.

Browser checks cover demo onboarding, inspect → assign → save → reload, CSV preview/commit, private workspace creation, viewer restrictions, revocation, cross-origin rejection, tenant separation, invalid import atomicity, desktop/mobile layouts and automated WCAG A/AA checks. Automated accessibility scans do not replace a manual assistive-technology audit. Screenshots are included in this directory.

The research tests verify labels, incomplete observation rejection, temporal boundaries with embargo, duplicate ticket rejection and JSON candidate/report output on synthetic data. Their synthetic model results are not retained or advertised as accuracy evidence.

The five development-only lint dependency advisories and deployment limitations are explained in the deployment guide. Legacy Python code and binary artifacts are preserved but were not executed or certified.

Business-rule validation covers default-score equivalence, typed configuration rejection, queue scopes, disabled/missing signals, tier aggregation, preview binding and expiry, concurrent revision checks, cross-workspace isolation, baseline migration and historical attribution. The browser workflow exercises JSON import, preview, publish, default restoration, loading an earlier version, export attribution and viewer denial.
