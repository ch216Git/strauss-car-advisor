import { AppShell } from "./components/AppShell";
import { useAdvisor } from "./hooks/useAdvisor";

export default function App() {
  const advisor = useAdvisor();
  return <AppShell advisor={advisor} />;
}
