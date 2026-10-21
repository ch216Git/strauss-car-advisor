import { useEffect, useState } from "react";
import { AppShell } from "./components/AppShell";
import { RepresentativeScreen } from "./components/RepresentativeScreen";
import { useAdvisor } from "./hooks/useAdvisor";

export const REPRESENTATIVE_HASH = "#/representative";

function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

export default function App() {
  return useHash() === REPRESENTATIVE_HASH ? <RepresentativeScreen /> : <Employee />;
}

// A component of its own: the representative's screen never starts an employee session.
function Employee() {
  const advisor = useAdvisor();
  return <AppShell advisor={advisor} />;
}
