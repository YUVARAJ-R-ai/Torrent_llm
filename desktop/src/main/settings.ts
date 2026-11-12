/** The app's own settings, kept as JSON in Electron's userData folder. */

import fs from "node:fs";
import path from "node:path";

import { type AppSettings, DEFAULT_SETTINGS } from "../shared/types";

const PORT_MIN = 1024;
const PORT_MAX = 65535;

/** Keeps only known keys with values of the right type and range. */
export function sanitize(raw: unknown, base: AppSettings = DEFAULT_SETTINGS): AppSettings {
  const out: AppSettings = { ...base, bootstrap: [...base.bootstrap] };
  if (!raw || typeof raw !== "object") return out;
  const r = raw as Record<string, unknown>;
  if (typeof r.label === "string") out.label = r.label.slice(0, 64);
  if (typeof r.memoryMb === "number" && Number.isFinite(r.memoryMb) && r.memoryMb >= 0)
    out.memoryMb = Math.min(Math.floor(r.memoryMb), 1_000_000);
  for (const key of ["controlPort", "listenPort"] as const) {
    const v = r[key];
    if (typeof v === "number" && Number.isInteger(v) && v >= PORT_MIN && v <= PORT_MAX) out[key] = v;
  }
  if (typeof r.executable === "string") out.executable = r.executable;
  if (Array.isArray(r.bootstrap))
    out.bootstrap = r.bootstrap.filter((a): a is string => typeof a === "string" && a.length < 300).slice(0, 8);
  if (typeof r.upnp === "boolean") out.upnp = r.upnp;
  if (typeof r.relay === "boolean") out.relay = r.relay;
  if (typeof r.lastSwarmName === "string") out.lastSwarmName = r.lastSwarmName.slice(0, 80);
  if (r.theme === "system" || r.theme === "light" || r.theme === "dark") out.theme = r.theme;
  return out;
}

export class SettingsStore {
  private value: AppSettings;

  constructor(private readonly file: string) {
    let raw: unknown = null;
    try {
      raw = JSON.parse(fs.readFileSync(file, "utf8"));
    } catch {
      raw = null; // first run or damaged: start from defaults
    }
    this.value = sanitize(raw);
  }

  get(): AppSettings {
    return { ...this.value, bootstrap: [...this.value.bootstrap] };
  }

  set(patch: Partial<AppSettings>): AppSettings {
    this.value = sanitize({ ...this.value, ...patch }, this.value);
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = `${this.file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(this.value, null, 1));
    fs.renameSync(tmp, this.file);
    return this.get();
  }
}
