/**
 * Where the device keeps its files: the same rules as src/torrent_llm/net/paths.py,
 * so the app finds the admin token and the installed backend the device created.
 */

import os from "node:os";
import path from "node:path";

export interface PathEnv {
  platform?: NodeJS.Platform;
  env?: NodeJS.ProcessEnv;
  home?: string;
}

export function dataDir({ platform = process.platform, env = process.env, home = os.homedir() }: PathEnv = {}): string {
  const override = env.TORRENT_LLM_HOME;
  if (override) return override;
  if (platform === "win32") {
    const local = env.LOCALAPPDATA;
    return local ? path.win32.join(local, "TorrentLLM") : path.win32.join(home, "AppData", "Local", "TorrentLLM");
  }
  if (platform === "darwin") return path.posix.join(home, "Library", "Application Support", "TorrentLLM");
  const xdg = env.XDG_DATA_HOME;
  if (xdg && path.posix.isAbsolute(xdg)) return path.posix.join(xdg, "torrent-llm");
  return path.posix.join(home, ".local", "share", "torrent-llm");
}

function join(platform: NodeJS.Platform, ...parts: string[]): string {
  return platform === "win32" ? path.win32.join(...parts) : path.posix.join(...parts);
}

export function adminTokenPath(p: PathEnv = {}): string {
  return join(p.platform ?? process.platform, dataDir(p), "admin-token");
}

/** Where scripts/install.sh / install.ps1 put the code and the venv. */
export function installedAppDir(p: PathEnv = {}): string {
  return (p.env ?? process.env).TORRENT_LLM_APP || join(p.platform ?? process.platform, dataDir(p), "app");
}

export function installedExecutable(p: PathEnv = {}): string {
  const platform = p.platform ?? process.platform;
  const env = p.env ?? process.env;
  const venv = env.TORRENT_VENV || join(platform, dataDir(p), "venv");
  return platform === "win32"
    ? path.win32.join(venv, "Scripts", "torrent-peer.exe")
    : path.posix.join(venv, "bin", "torrent-peer");
}
