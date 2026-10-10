import Link from "next/link";
import {
  ArrowUpRight,
  Activity,
  FileSpreadsheet,
  Radio,
  ShieldCheck,
  ArrowRight,
} from "lucide-react";
export default function Home() {
  return (
    <>
      <header className="public-header">
        <Link className="brand" href="/">
          <Activity aria-hidden="true" /> SignalDesk<span>by Daru897</span>
        </Link>
        <nav aria-label="Main">
          <Link href="/case-study">The case study</Link>
          <Link className="button dark" href="/monitor">
            Open workspace <ArrowUpRight size={16} />
          </Link>
        </nav>
      </header>
      <main>
        <section className="hero">
          <div>
            <p className="eyebrow">BUILT FOR BPO OPERATIONS</p>
            <h1>
              Catch the signal.
              <br />
              <em>Before the escalation.</em>
            </h1>
            <p className="hero-copy">
              Give every shift a clear starting point. Bring your support
              tickets together, see what needs attention, and turn risk signals
              into accountable action.
            </p>
            <div className="button-row">
              <Link className="button dark" href="/monitor?demo=1">
                Explore the live demo <ArrowRight size={17} />
              </Link>
              <Link className="button outline" href="/monitor">
                Bring your own tickets
              </Link>
            </div>
            <p className="muted small">
              Fictional demo data · No sign-up · No model API costs
            </p>
            <div className="hero-proof">
              <span>
                <FileSpreadsheet size={18} /> CSV import
              </span>
              <span>
                <Radio size={18} /> Zendesk sync
              </span>
              <span>
                <ShieldCheck size={18} /> Explainable signals
              </span>
            </div>
          </div>
          <div className="hero-art">
            <div className="art-top">
              <span className="dot" /> OPERATIONS PULSE{" "}
              <span>Northstar · Demo</span>
            </div>
            <h2>Your queue, in focus.</h2>
            <div className="art-number">
              04 <span>tickets need a closer look</span>
            </div>
            <div className="spark-bars" aria-hidden="true">
              {[25, 40, 30, 55, 38, 70, 60, 90, 65, 44, 30, 20].map((v, i) => (
                <i key={i} style={{ height: v + "%" }} />
              ))}
            </div>
            <div className="art-ticket">
              <span className="badge critical">Critical · 85</span>
              <strong>Payment reversed but refund missing</strong>
              <p>SLA deadline approaching · Multiple transfers</p>
              <div>
                <span className="avatar">M</span> Maya · Investigating{" "}
                <ArrowUpRight size={18} />
              </div>
            </div>
            <p className="art-foot">
              Illustrative risk index, not an escalation probability.
            </p>
          </div>
        </section>
        <section className="steps section">
          <p className="eyebrow">FROM BACKLOG TO NEXT ACTION</p>
          <h2>A calmer way to lead the queue.</h2>
          <div className="three-columns">
            {[
              [
                "01",
                "Connect your reality",
                "Upload a CSV with column mapping, or connect a customer-owned Zendesk account for scheduled updates.",
              ],
              [
                "02",
                "Understand the urgency",
                "Review priority, age, inactivity, transfers and supplied SLA deadlines. See the evidence and missing data for every score.",
              ],
              [
                "03",
                "Close the loop",
                "Assign an intervention owner, record the next step and track which tickets still need attention.",
              ],
            ].map(([n, t, d]) => (
              <article key={n}>
                <span className="step-number">{n}</span>
                <h3>{t}</h3>
                <p>{d}</p>
              </article>
            ))}
          </div>
        </section>
        <section className="honesty section">
          <div>
            <p className="eyebrow">BUILT TO BE EXPLAINED</p>
            <h2>
              Useful signals.
              <br />
              Honest limitations.
            </h2>
          </div>
          <div>
            <p>
              SignalDesk starts with a transparent, rules-based risk index. It
              does not claim to predict a customer’s behavior or replace a team
              lead’s judgment.
            </p>
            <p>
              Zendesk sync is near-real-time polling. Data freshness, missing
              signals and connection errors are visible in the workspace. A
              separate training workflow supports evaluation on genuine,
              time-stamped escalation outcomes.
            </p>
            <Link href="/case-study">
              Read the product decisions <ArrowUpRight size={16} />
            </Link>
          </div>
        </section>
      </main>
      <footer>
        <span>SignalDesk / A portfolio product by Daru897</span>
        <a href="https://github.com/Daru897/Predictive_Escalation_Risk_Model">
          View source on GitHub ↗
        </a>
      </footer>
    </>
  );
}
