import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "SignalDesk · Escalation intelligence for BPOs",
  description:
    "Turn support queues into focused action. Import CSV tickets or monitor Zendesk with explainable escalation risk signals.",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
