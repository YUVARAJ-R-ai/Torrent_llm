import type { Metadata } from "next";

import "./globals.css";

/*
 * No webfont. The data-viz reference specifies the system sans throughout, and
 * dropping next/font/google also means `npm run build` needs no network — worth
 * having on a tool that should build from a clean clone on a rig machine.
 */

export const metadata: Metadata = {
  title: "Torrent-LLM pool",
  description:
    "Devices lending what they have to run one model together, and surviving the loss of any of them.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col">{children}</body>
    </html>
  );
}
