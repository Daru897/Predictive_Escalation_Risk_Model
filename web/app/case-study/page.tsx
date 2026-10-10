import Link from "next/link";
import { Activity } from "lucide-react";
export default function CaseStudy() {
  return (
    <>
      <header className="public-header">
        <Link href="/" className="brand">
          <Activity />
          SignalDesk
        </Link>
        <Link href="/monitor?demo=1" className="button dark">
          Try the demo
        </Link>
      </header>
      <main className="case-study">
        <p className="eyebrow">PRODUCT CASE STUDY / OCTOBER 2026</p>
        <h1>From a prediction experiment to an operations workflow.</h1>
        <p className="lead">
          Rebuilding a support escalation prototype for the people who run BPO
          queues.
        </p>
        <div className="case-grid">
          <aside>
            <strong>Audience</strong>
            <p>BPO team leads and operations managers</p>
            <strong>Scope</strong>
            <p>
              CSV ingestion, Zendesk monitoring, explainable prioritization and
              intervention tracking
            </p>
            <strong>Status</strong>
            <p>Portfolio application and customer-hosted pilot foundation</p>
          </aside>
          <article>
            <h2>The problem</h2>
            <p>
              A long ticket queue is not a plan. Leads need to know which
              tickets deserve attention, why they were flagged, and who is
              following up. A score alone does not answer those questions.
            </p>
            <h2>What the audit changed</h2>
            <p>
              The original repository had 40 training rows and a reported
              ROC-AUC of 0.42. A fallback training path treated non-closed
              tickets as escalations, while the “time-based” split used ticket
              IDs. Those results do not support a reliable escalation predictor.
            </p>
            <p>
              The rebuilt app uses a disclosed risk index instead. It preserves
              the historical experiment for inspection, introduces validated
              inputs, and keeps model evaluation separate from the operational
              workflow.
            </p>
            <h2>Three product decisions</h2>
            <ol>
              <li>
                <strong>Start with the data people already have.</strong> CSV
                column mapping and row-level errors make onboarding possible
                before a system integration.
              </li>
              <li>
                <strong>Make freshness part of the experience.</strong> Saved
                Zendesk cursors, scheduled polling and visible failures help
                leads distinguish a quiet queue from a broken connection.
              </li>
              <li>
                <strong>End with an action.</strong> Each ticket has a review
                state, owner and note. Recorded interventions stay separate from
                the ticket’s source status.
              </li>
            </ol>
            <h2>A two-minute walkthrough</h2>
            <ol>
              <li>Open the demo and select a critical ticket.</li>
              <li>Inspect its score contributions and missing data.</li>
              <li>Set an intervention owner and save a follow-up note.</li>
              <li>
                Open Data sources, download the CSV template and preview an
                import.
              </li>
              <li>
                Open Trends to inspect queue snapshots and export the review
                queue.
              </li>
            </ol>
            <h2>What is measured</h2>
            <p>
              Validation covers input rejection, workspace isolation, viewer
              permissions, deterministic scoring, sync retries and checkpoint
              safety. No customer trial, SLA reduction, production model
              accuracy or business ROI is claimed.
            </p>
            <h2>The path to production</h2>
            <p>
              Customer-hosted pilots need verified Zendesk permissions, tested
              backups and an access-key distribution process. Broader rollout
              needs individual SSO identities, normalized managed Postgres
              storage, retention controls and a Zendesk-approved integration
              strategy. Genuine outcome labels and temporal holdouts are
              prerequisites for predictive claims.
            </p>
            <Link href="/monitor?demo=1" className="button dark">
              Explore the workflow →
            </Link>
          </article>
        </div>
      </main>
      <footer>
        <Link href="/">← Back to SignalDesk</Link>
        <a href="https://github.com/Daru897/Predictive_Escalation_Risk_Model">
          Source repository ↗
        </a>
      </footer>
    </>
  );
}
