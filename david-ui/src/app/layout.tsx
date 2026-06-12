import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/nav";
import { ProvenanceFooter } from "@/components/provenance-footer";

export const metadata: Metadata = {
  title: "DAVID / M0.1",
  description: "Capture-resistant prediction engine for tobacco industry interference",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased flex flex-col min-h-screen">
        <Nav />
        <main className="ml-56 flex-1 p-8">
          {children}
        </main>
        <div className="ml-56">
          <ProvenanceFooter />
        </div>
      </body>
    </html>
  );
}
