/**
 * The only bridge between the sandboxed renderer and the main process: a small,
 * typed API (window.tllm) and nothing else. No Node, no ipcRenderer, no fs.
 */

import { contextBridge, ipcRenderer } from "electron";

import type {
  AppSettings,
  BackendStatus,
  GenerateRequest,
  GenerateResult,
  ReadPath,
  StartOptions,
  TllmBridge,
  WritePath,
} from "../shared/types";

function subscribe<T>(channel: string, callback: (value: T) => void): () => void {
  const listener = (_event: unknown, value: T) => callback(value);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

let streams = 0;

const bridge: TllmBridge = {
  backend: {
    status: () => ipcRenderer.invoke("backend:status") as Promise<BackendStatus>,
    start: (options: StartOptions) => ipcRenderer.invoke("backend:start", options) as Promise<BackendStatus>,
    stop: () => ipcRenderer.invoke("backend:stop") as Promise<BackendStatus>,
    install: () => ipcRenderer.invoke("backend:install") as Promise<BackendStatus>,
    logs: () => ipcRenderer.invoke("backend:logs") as Promise<string[]>,
    onStatus: (callback) => subscribe<BackendStatus>("backend:status", callback),
    onLog: (callback) => subscribe<string>("backend:log", callback),
  },
  api: {
    get: <T>(path: ReadPath, query?: Record<string, string | number>) =>
      ipcRenderer.invoke("api:get", path, query) as Promise<T>,
    post: <T>(path: WritePath, body?: unknown) => ipcRenderer.invoke("api:post", path, body) as Promise<T>,
    generate: async (request: GenerateRequest, onToken: (text: string, index: number) => void) => {
      const id = `${Date.now()}-${streams++}`;
      const stop = subscribe<{ text: string; index: number }>(`api:token:${id}`, (t) => onToken(t.text, t.index));
      try {
        return (await ipcRenderer.invoke("api:generate", request, id)) as GenerateResult;
      } finally {
        stop();
      }
    },
  },
  settings: {
    get: () => ipcRenderer.invoke("settings:get") as Promise<AppSettings>,
    set: (patch: Partial<AppSettings>) => ipcRenderer.invoke("settings:set", patch) as Promise<AppSettings>,
  },
  system: {
    openExternal: (url: string) => ipcRenderer.invoke("system:openExternal", url) as Promise<void>,
    copy: (text: string) => ipcRenderer.invoke("system:copy", text) as Promise<void>,
    platform: process.platform,
  },
};

contextBridge.exposeInMainWorld("tllm", bridge);
