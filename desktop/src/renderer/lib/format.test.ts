import { describe, expect, it } from "vitest";

import { ago, bandwidth, bytes, ms, ratio, shortId } from "./format";

describe("format", () => {
  it("bytes", () => {
    expect(bytes(0)).toBe("0 B");
    expect(bytes(1536)).toBe("1.5 KB");
    expect(bytes(581 * 1024)).toBe("581 KB");
    expect(bytes(null)).toBe("—");
    expect(bytes(Number.NaN)).toBe("—");
  });
  it("ms", () => {
    expect(ms(12.4)).toBe("12 ms");
    expect(ms(4583)).toBe("4.58 s");
    expect(ms(undefined)).toBe("—");
  });
  it("bandwidth", () => {
    expect(bandwidth(125_000)).toBe("1.0 Mbit/s");
    expect(bandwidth(2.5e8)).toBe("2.0 Gbit/s");
    expect(bandwidth(null)).toBe("—");
  });
  it("misc", () => {
    expect(shortId("abcdefghijklmnop")).toBe("abcdefgh");
    expect(ago(100, 130)).toBe("30s ago");
    expect(ago(0, 7300)).toBe("2h ago");
    expect(ratio(595080, 75400)).toBe("7.9x");
    expect(ratio(0, 0)).toBe("—");
  });
});
