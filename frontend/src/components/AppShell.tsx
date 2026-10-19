import type { Advisor } from "../hooks/useAdvisor";
import { AdvisorPanel } from "./AdvisorPanel";
import { DecisionPanel } from "./DecisionPanel";
import { Header } from "./Header";

/** Desktop: conversation on the right, decision on the left (RTL order). Narrow screens stack them. */
export function AppShell({ advisor }: { advisor: Advisor }) {
  return (
    <div className="app" dir="rtl" lang="he">
      <Header onNewConversation={() => void advisor.newConversation()} disabled={advisor.status === "starting"} />
      <main className="layout">
        <AdvisorPanel advisor={advisor} />
        <DecisionPanel decision={advisor.decision} pending={advisor.pending} canSend={advisor.canSend} onSend={advisor.send} />
      </main>
    </div>
  );
}
