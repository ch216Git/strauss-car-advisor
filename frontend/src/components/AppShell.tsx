import { useState } from "react";
import type { Advisor } from "../hooks/useAdvisor";
import { useDocumentUpload } from "../hooks/useDocumentUpload";
import { useRepresentativeAnswers } from "../hooks/useRepresentativeAnswers";
import { AdvisorPanel } from "./AdvisorPanel";
import { DecisionPanel } from "./DecisionPanel";
import { Header } from "./Header";
import { HistoryDrawer } from "./HistoryDrawer";

/** Desktop: conversation on the right, decision on the left (RTL order). Narrow screens stack them. */
export function AppShell({ advisor }: { advisor: Advisor }) {
  const [historyOpen, setHistoryOpen] = useState(false);
  const documents = useDocumentUpload(advisor.sessionId);
  const sent = advisor.entries.filter((entry) => entry.role === "employee").length;
  const answers = useRepresentativeAnswers(advisor.sessionId, advisor.decision, advisor.pending || advisor.readOnly, sent);

  const open = (id: string) => {
    advisor.openConversation(id);
    setHistoryOpen(false);
  };

  return (
    <div className="app" dir="rtl" lang="he">
      <Header
        onNewConversation={() => void advisor.newConversation()}
        onOpenHistory={() => setHistoryOpen(true)}
        disabled={advisor.status === "starting" || advisor.busy}
      />
      <main className="layout">
        <AdvisorPanel advisor={advisor} onAttach={(file) => void documents.upload(file)} />
        <DecisionPanel
          decision={advisor.decision}
          pending={advisor.pending}
          canSend={advisor.canSend}
          onSend={advisor.send}
          answers={answers}
          document={documents.state}
          onDocumentConfirm={(message) => {
            advisor.send(message);
            documents.clear();
          }}
          onDocumentClose={documents.clear}
        />
      </main>
      {historyOpen && (
        <HistoryDrawer
          conversations={advisor.conversations}
          activeId={advisor.activeId}
          enabled={advisor.historyEnabled}
          busy={advisor.busy}
          onClose={() => setHistoryOpen(false)}
          onOpen={open}
          onDelete={advisor.removeConversation}
          onDeleteAll={advisor.removeAll}
          onToggleEnabled={advisor.toggleHistory}
        />
      )}
    </div>
  );
}
