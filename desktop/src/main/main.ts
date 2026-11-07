/**
 * The Electron main process: one window, the backend manager, and the IPC handlers
 * behind window.tllm. The renderer is sandboxed (no Node), so everything that touches
 * the filesystem, processes or the network happens here.
 */

import path from "node:path";

import { BrowserWindow, app, clipboard, ipcMain, shell } from "electron";

import type { AppSettings, GenerateRequest, StartOptions } from "../shared/types";
import { apiGenerate, apiGet, apiPost } from "./api";
import { BackendManager, realDeps } from "./backend";
import { SettingsStore } from "./settings";

let window: BrowserWindow | null = null;

const settings = new SettingsStore(path.join(app.getPath("userData"), "settings.json"));
const backend = new BackendManager(realDeps(process.resourcesPath), () => settings.get());

function base(): string {
  return `http://127.0.0.1:${settings.get().controlPort}`;
}

function send(channel: string, payload: unknown): void {
  if (window && !window.isDestroyed()) window.webContents.send(channel, payload);
}

backend.on("status", (status) => send("backend:status", status));
backend.on("log", (line) => send("backend:log", line));

function createWindow(): void {
  window = new BrowserWindow({
    width: 1180,
    height: 780,
    minWidth: 900,
    minHeight: 600,
    title: "Torrent-LLM",
    backgroundColor: "#f9f9f7",
    webPreferences: {
      preload: path.join(__dirname, "..", "preload", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  // Links open in the system browser; the app window never navigates away.
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) void window.loadURL(devUrl);
  else void window.loadFile(path.join(__dirname, "..", "renderer", "index.html"));

  // For checking the build without a person: TLLM_SCREENSHOT=out.png saves what the
  // window shows a moment after it loads, then quits.
  const shot = process.env.TLLM_SCREENSHOT;
  if (shot) {
    window.webContents.once("did-finish-load", () => {
      setTimeout(async () => {
        const image = await window?.webContents.capturePage();
        if (image) (await import("node:fs")).writeFileSync(shot, image.toPNG());
        app.exit(0);
      }, Number(process.env.TLLM_SCREENSHOT_DELAY_MS ?? 1500));
    });
  }
}

ipcMain.handle("backend:status", () => backend.current());
ipcMain.handle("backend:start", (_e, options: StartOptions) => backend.start(options));
ipcMain.handle("backend:stop", () => backend.stop());
ipcMain.handle("backend:install", () => backend.install());
ipcMain.handle("backend:logs", () => backend.recentLogs());

ipcMain.handle("api:get", (_e, p: string, query?: Record<string, string | number>) =>
  apiGet(base(), p, backend.adminToken(), query),
);
ipcMain.handle("api:post", (_e, p: string, body?: unknown) => apiPost(base(), p, backend.adminToken(), body));
ipcMain.handle("api:generate", (event, request: GenerateRequest, streamId: string) =>
  apiGenerate(base(), backend.adminToken(), request, (text, index) =>
    event.sender.send(`api:token:${streamId}`, { text, index }),
  ),
);

ipcMain.handle("settings:get", () => settings.get());
ipcMain.handle("settings:set", (_e, patch: Partial<AppSettings>) => settings.set(patch));

ipcMain.handle("system:openExternal", async (_e, url: string) => {
  if (!/^https?:\/\//.test(url)) throw new Error("only web links can be opened");
  await shell.openExternal(url);
});
ipcMain.handle("system:copy", (_e, text: string) => clipboard.writeText(String(text)));

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// Leave the swarm properly when the app closes, instead of vanishing on everyone.
let quitting = false;
app.on("before-quit", (event) => {
  if (quitting || backend.current().phase === "stopped" || backend.current().phase === "missing") return;
  event.preventDefault();
  quitting = true;
  void backend.stop().finally(() => {
    backend.dispose();
    app.quit();
  });
});
