import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mavic · Video Agent Buddy",
  description: "Camera, voice, memory, and location agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
