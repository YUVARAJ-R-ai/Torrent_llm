import type { NextConfig } from "next";

/**
 * Two ways of building the same dashboard.
 *
 * The normal one (`npm run dev`) is for working on it. Setting NEXT_EXPORT=1 turns
 * it into plain static files that every device in a pool serves itself, so someone
 * with only Python installed can open a device in a browser and see the pool. See
 * scripts/build_ui.sh.
 */
const exporting = process.env.NEXT_EXPORT === "1";

const nextConfig: NextConfig = {
  ...(exporting ? { output: "export" as const, trailingSlash: true } : {}),
};

export default nextConfig;
