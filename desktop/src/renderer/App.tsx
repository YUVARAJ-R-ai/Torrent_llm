import {
  Activity,
  Boxes,
  Home as HomeIcon,
  type LucideIcon,
  MessageSquare,
  Settings as SettingsIcon,
  ShieldCheck,
  Stethoscope,
  Users,
  Waypoints,
} from "lucide-react";
import { useEffect, useState } from "react";

import type { BackendStatus } from "../shared/types";
import { useBackend } from "./lib/hooks";
import ActivityScreen from "./screens/Activity";
import ChatScreen from "./screens/Chat";
import DiagnosticsScreen from "./screens/Diagnostics";
import HomeScreen from "./screens/Home";
import ModelScreen from "./screens/Model";
import PeersScreen from "./screens/Peers";
import SettingsScreen from "./screens/Settings";
import SwarmScreen from "./screens/Swarm";
import TrustScreen from "./screens/Trust";

export type ScreenId =
  | "home"
  | "swarm"
  | "peers"
  | "model"
  | "chat"
  | "activity"
  | "trust"
  | "settings"
  | "diagnostics";

/** The screen named in the URL hash (``#peers``), so a screen can be opened directly. */
function initialScreen(): ScreenId {
  const named = window.location.hash.replace(/^#/, "");
  return SCREENS.find((s) => s.id === named)?.id ?? "home";
}

export interface ScreenProps {
  status: BackendStatus;
  go: (screen: ScreenId) => void;
}

const SCREENS: { id: ScreenId; label: string; icon: LucideIcon; needsRunning: boolean }[] = [
  { id: "home", label: "Home", icon: HomeIcon, needsRunning: false },
  { id: "swarm", label: "Swarm", icon: Waypoints, needsRunning: false },
  { id: "peers", label: "Peers", icon: Users, needsRunning: true },
  { id: "model", label: "Model", icon: Boxes, needsRunning: true },
  { id: "chat", label: "Chat", icon: MessageSquare, needsRunning: true },
  { id: "activity", label: "Activity", icon: Activity, needsRunning: true },
  { id: "trust", label: "Trust", icon: ShieldCheck, needsRunning: true },
  { id: "settings", label: "Settings", icon: SettingsIcon, needsRunning: false },
  { id: "diagnostics", label: "Diagnostics", icon: Stethoscope, needsRunning: false },
];

const COMPONENTS: Record<ScreenId, (props: ScreenProps) => React.ReactNode> = {
  home: HomeScreen,
  swarm: SwarmScreen,
  peers: PeersScreen,
  model: ModelScreen,
  chat: ChatScreen,
  activity: ActivityScreen,
  trust: TrustScreen,
  settings: SettingsScreen,
  diagnostics: DiagnosticsScreen,
};

function useTheme(): void {
  useEffect(() => {
    void window.tllm.settings.get().then((s) => {
      if (s.theme === "system") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", s.theme);
    });
  }, []);
}

export default function App() {
  const status = useBackend();
  const [screen, setScreen] = useState<ScreenId>(initialScreen);
  useTheme();

  if (!status) {
    return <div className="p-6 text-sm" style={{ color: "var(--text-muted)" }}>Starting…</div>;
  }
  const running = status.phase === "running";
  const Current = COMPONENTS[screen];

  return (
    <div className="flex h-full">
      <nav
        aria-label="Sections"
        className="flex w-48 shrink-0 flex-col border-r px-2 py-4"
        style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}
      >
        <div className="mb-4 px-2 text-sm font-semibold">Torrent-LLM</div>
        {SCREENS.map(({ id, label, icon: Icon, needsRunning }) => {
          const active = id === screen;
          const dim = needsRunning && !running;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setScreen(id)}
              aria-current={active ? "page" : undefined}
              className="flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm"
              style={{
                background: active ? "var(--surface-2)" : "transparent",
                color: dim ? "var(--text-muted)" : "var(--text-primary)",
              }}
            >
              <Icon size={16} aria-hidden />
              {label}
            </button>
          );
        })}
      </nav>
      <main className="min-w-0 flex-1 overflow-y-auto px-6 py-5">
        <Current status={status} go={setScreen} />
      </main>
    </div>
  );
}
