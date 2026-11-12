// @vitest-environment node
import { describe, expect, it } from "vitest";

import { adminTokenPath, dataDir, installedAppDir, installedExecutable } from "./paths";

describe("paths", () => {
  describe("dataDir", () => {
    it("TORRENT_LLM_HOME override always wins regardless of platform", () => {
      expect(dataDir({ platform: "linux", env: { TORRENT_LLM_HOME: "/custom/data" } })).toBe("/custom/data");
      expect(dataDir({ platform: "win32", env: { TORRENT_LLM_HOME: "D:\\custom\\data" } })).toBe("D:\\custom\\data");
      expect(dataDir({ platform: "darwin", env: { TORRENT_LLM_HOME: "/Users/custom/data" } })).toBe("/Users/custom/data");
    });

    it("uses LOCALAPPDATA on win32 when set", () => {
      const result = dataDir({
        platform: "win32",
        env: { LOCALAPPDATA: "C:\\Users\\test\\AppData\\Local" },
        home: "C:\\Users\\test",
      });
      expect(result).toBe("C:\\Users\\test\\AppData\\Local\\TorrentLLM");
    });

    it("falls back to AppData/Local on win32 when LOCALAPPDATA is unset", () => {
      const result = dataDir({
        platform: "win32",
        env: {},
        home: "C:\\Users\\test",
      });
      expect(result).toBe("C:\\Users\\test\\AppData\\Local\\TorrentLLM");
    });

    it("uses Application Support on darwin", () => {
      const result = dataDir({
        platform: "darwin",
        env: { XDG_DATA_HOME: "/some/xdg" },
        home: "/Users/test",
      });
      expect(result).toBe("/Users/test/Library/Application Support/TorrentLLM");
    });

    it("uses absolute XDG_DATA_HOME on linux", () => {
      const result = dataDir({
        platform: "linux",
        env: { XDG_DATA_HOME: "/opt/shared/data" },
        home: "/home/test",
      });
      expect(result).toBe("/opt/shared/data/torrent-llm");
    });

    it("ignores relative XDG_DATA_HOME on linux and falls back to home/.local/share", () => {
      const result = dataDir({
        platform: "linux",
        env: { XDG_DATA_HOME: "relative/path" },
        home: "/home/test",
      });
      expect(result).toBe("/home/test/.local/share/torrent-llm");
    });

    it("falls back to home/.local/share on linux when XDG_DATA_HOME is unset", () => {
      const result = dataDir({
        platform: "linux",
        env: {},
        home: "/home/test",
      });
      expect(result).toBe("/home/test/.local/share/torrent-llm");
    });
  });

  describe("adminTokenPath", () => {
    it("returns admin-token path inside dataDir on posix", () => {
      const p = { platform: "linux" as const, env: { TORRENT_LLM_HOME: "/data" } };
      expect(adminTokenPath(p)).toBe("/data/admin-token");
    });

    it("returns admin-token path inside dataDir on win32", () => {
      const p = { platform: "win32" as const, env: { TORRENT_LLM_HOME: "C:\\data" } };
      expect(adminTokenPath(p)).toBe("C:\\data\\admin-token");
    });
  });

  describe("installedAppDir", () => {
    it("respects TORRENT_LLM_APP override", () => {
      const p = { platform: "linux" as const, env: { TORRENT_LLM_APP: "/opt/tllm/app" } };
      expect(installedAppDir(p)).toBe("/opt/tllm/app");
    });

    it("defaults to app directory inside dataDir on posix", () => {
      const p = { platform: "linux" as const, env: { TORRENT_LLM_HOME: "/data" } };
      expect(installedAppDir(p)).toBe("/data/app");
    });

    it("defaults to app directory inside dataDir on win32", () => {
      const p = { platform: "win32" as const, env: { TORRENT_LLM_HOME: "C:\\data" } };
      expect(installedAppDir(p)).toBe("C:\\data\\app");
    });
  });

  describe("installedExecutable", () => {
    it("returns bin/torrent-peer inside default venv on posix", () => {
      const p = { platform: "linux" as const, env: { TORRENT_LLM_HOME: "/data" } };
      expect(installedExecutable(p)).toBe("/data/venv/bin/torrent-peer");
    });

    it("returns Scripts/torrent-peer.exe inside default venv on win32", () => {
      const p = { platform: "win32" as const, env: { TORRENT_LLM_HOME: "C:\\data" } };
      expect(installedExecutable(p)).toBe("C:\\data\\venv\\Scripts\\torrent-peer.exe");
    });

    it("respects TORRENT_VENV override on posix", () => {
      const p = { platform: "linux" as const, env: { TORRENT_VENV: "/opt/custom-venv" } };
      expect(installedExecutable(p)).toBe("/opt/custom-venv/bin/torrent-peer");
    });

    it("respects TORRENT_VENV override on win32", () => {
      const p = { platform: "win32" as const, env: { TORRENT_VENV: "D:\\custom-venv" } };
      expect(installedExecutable(p)).toBe("D:\\custom-venv\\Scripts\\torrent-peer.exe");
    });
  });
});
