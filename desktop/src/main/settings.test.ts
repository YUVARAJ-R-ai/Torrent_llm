// @vitest-environment node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { DEFAULT_SETTINGS } from "../shared/types";
import { sanitize, SettingsStore } from "./settings";

describe("settings", () => {
  describe("sanitize", () => {
    it("returns base when raw is null, undefined, or non-object", () => {
      expect(sanitize(null)).toEqual(DEFAULT_SETTINGS);
      expect(sanitize(undefined)).toEqual(DEFAULT_SETTINGS);
      expect(sanitize("string")).toEqual(DEFAULT_SETTINGS);
      expect(sanitize(123)).toEqual(DEFAULT_SETTINGS);
      expect(sanitize(true)).toEqual(DEFAULT_SETTINGS);
    });

    it("keeps valid label and slices if longer than 64 characters", () => {
      const valid = sanitize({ label: "valid-device" });
      expect(valid.label).toBe("valid-device");

      const longLabel = "a".repeat(100);
      const sanitized = sanitize({ label: longLabel });
      expect(sanitized.label).toBe("a".repeat(64));

      const nonString = sanitize({ label: 123 }, { ...DEFAULT_SETTINGS, label: "fallback" });
      expect(nonString.label).toBe("fallback");
    });

    it("keeps valid memoryMb, floors floats, caps at 1,000,000, and drops negative/NaN/non-number", () => {
      expect(sanitize({ memoryMb: 4096 }).memoryMb).toBe(4096);
      expect(sanitize({ memoryMb: 2048.8 }).memoryMb).toBe(2048);
      expect(sanitize({ memoryMb: 0 }).memoryMb).toBe(0);
      expect(sanitize({ memoryMb: 2_000_000 }).memoryMb).toBe(1_000_000);

      // Dropped values should retain base value
      const base = { ...DEFAULT_SETTINGS, memoryMb: 1024 };
      expect(sanitize({ memoryMb: -1 }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: NaN }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: Infinity }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: -Infinity }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: "4096" }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: true }, base).memoryMb).toBe(1024);
      expect(sanitize({ memoryMb: null }, base).memoryMb).toBe(1024);
    });

    it("validates controlPort and listenPort between 1024 and 65535, dropping out-of-range or non-integers", () => {
      const base = { ...DEFAULT_SETTINGS, controlPort: 9000, listenPort: 9100 };

      // Valid boundary values
      expect(sanitize({ controlPort: 1024, listenPort: 65535 }, base)).toMatchObject({
        controlPort: 1024,
        listenPort: 65535,
      });

      // Below 1024
      expect(sanitize({ controlPort: 1023, listenPort: 80 }, base)).toMatchObject({
        controlPort: 9000,
        listenPort: 9100,
      });
      expect(sanitize({ controlPort: 0, listenPort: -1 }, base)).toMatchObject({
        controlPort: 9000,
        listenPort: 9100,
      });

      // Above 65535
      expect(sanitize({ controlPort: 65536, listenPort: 70000 }, base)).toMatchObject({
        controlPort: 9000,
        listenPort: 9100,
      });

      // Non-integers and non-numbers
      expect(sanitize({ controlPort: 8080.5, listenPort: 9000.1 }, base)).toMatchObject({
        controlPort: 9000,
        listenPort: 9100,
      });
      expect(sanitize({ controlPort: "9000", listenPort: true }, base)).toMatchObject({
        controlPort: 9000,
        listenPort: 9100,
      });
    });

    it("keeps executable if string, drops non-string", () => {
      expect(sanitize({ executable: "/usr/local/bin/torrent-peer" }).executable).toBe("/usr/local/bin/torrent-peer");
      const base = { ...DEFAULT_SETTINGS, executable: "/default/bin" };
      expect(sanitize({ executable: 12345 }, base).executable).toBe("/default/bin");
    });

    it("filters non-string bootstrap entries, drops entries >= 300 chars, and keeps at most 8 entries", () => {
      const longEntry = "x".repeat(300);
      const validEntries = Array.from({ length: 10 }, (_, i) => `tcp://10.0.0.${i}:9100`);
      const raw = {
        bootstrap: [
          ...validEntries,
          123,
          null,
          undefined,
          { foo: "bar" },
          longEntry,
        ],
      };
      const sanitized = sanitize(raw);
      expect(sanitized.bootstrap).toHaveLength(8);
      expect(sanitized.bootstrap).toEqual(validEntries.slice(0, 8));

      // Non-array bootstrap drops and keeps base
      const base = { ...DEFAULT_SETTINGS, bootstrap: ["tcp://base:9100"] };
      expect(sanitize({ bootstrap: "not-an-array" }, base).bootstrap).toEqual(["tcp://base:9100"]);
    });

    it("keeps boolean upnp and relay, drops non-booleans", () => {
      expect(sanitize({ upnp: false, relay: true })).toMatchObject({
        upnp: false,
        relay: true,
      });
      const base = { ...DEFAULT_SETTINGS, upnp: true, relay: false };
      expect(sanitize({ upnp: "false", relay: 1 }, base)).toMatchObject({
        upnp: true,
        relay: false,
      });
    });

    it("keeps lastSwarmName up to 80 chars", () => {
      expect(sanitize({ lastSwarmName: "sw-1" }).lastSwarmName).toBe("sw-1");
      const longName = "s".repeat(120);
      expect(sanitize({ lastSwarmName: longName }).lastSwarmName).toBe("s".repeat(80));
      const base = { ...DEFAULT_SETTINGS, lastSwarmName: "prev" };
      expect(sanitize({ lastSwarmName: null }, base).lastSwarmName).toBe("prev");
    });

    it("validates theme (system, light, dark) and drops unknown theme", () => {
      expect(sanitize({ theme: "system" }).theme).toBe("system");
      expect(sanitize({ theme: "light" }).theme).toBe("light");
      expect(sanitize({ theme: "dark" }).theme).toBe("dark");

      const base = { ...DEFAULT_SETTINGS, theme: "light" as const };
      expect(sanitize({ theme: "neon" }, base).theme).toBe("light");
      expect(sanitize({ theme: 123 }, base).theme).toBe("light");
      expect(sanitize({ theme: null }, base).theme).toBe("light");
    });
  });

  describe("SettingsStore", () => {
    let tmpDir: string;
    let settingsFile: string;

    beforeEach(() => {
      tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "tllm-settings-test-"));
      settingsFile = path.join(tmpDir, "settings.json");
    });

    afterEach(() => {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    });

    it("starts from defaults when file does not exist", () => {
      const store = new SettingsStore(settingsFile);
      expect(store.get()).toEqual(DEFAULT_SETTINGS);
    });

    it("starts from defaults when JSON file is damaged", () => {
      fs.writeFileSync(settingsFile, "{ damaged json content ...", "utf8");
      const store = new SettingsStore(settingsFile);
      expect(store.get()).toEqual(DEFAULT_SETTINGS);
    });

    it("round trips settings and preserves mutations through persistence", () => {
      const store = new SettingsStore(settingsFile);
      const updated = store.set({
        label: "worker-node",
        memoryMb: 8192,
        controlPort: 9005,
        listenPort: 9105,
        theme: "dark",
        bootstrap: ["tcp://node1:9100"],
      });

      expect(updated.label).toBe("worker-node");
      expect(updated.memoryMb).toBe(8192);
      expect(updated.controlPort).toBe(9005);
      expect(updated.theme).toBe("dark");
      expect(updated.bootstrap).toEqual(["tcp://node1:9100"]);

      // Verify file contains valid JSON
      const content = fs.readFileSync(settingsFile, "utf8");
      const parsed = JSON.parse(content);
      expect(parsed.label).toBe("worker-node");
      expect(parsed.memoryMb).toBe(8192);

      // Re-read with a new store instance
      const secondStore = new SettingsStore(settingsFile);
      expect(secondStore.get()).toEqual(updated);
    });

    it("writes atomically without leaving leftover tmp files", () => {
      const nestedFile = path.join(tmpDir, "sub", "dir", "settings.json");
      const store = new SettingsStore(nestedFile);
      store.set({ label: "atomic-test" });

      expect(fs.existsSync(nestedFile)).toBe(true);
      expect(fs.existsSync(`${nestedFile}.tmp`)).toBe(false);

      const parsed = JSON.parse(fs.readFileSync(nestedFile, "utf8"));
      expect(parsed.label).toBe("atomic-test");
    });

    it("returns cloned bootstrap array so mutating get() does not affect store", () => {
      const store = new SettingsStore(settingsFile);
      store.set({ bootstrap: ["tcp://original:9100"] });

      const state1 = store.get();
      state1.bootstrap.push("tcp://mutated:9100");

      const state2 = store.get();
      expect(state2.bootstrap).toEqual(["tcp://original:9100"]);
    });
  });
});
