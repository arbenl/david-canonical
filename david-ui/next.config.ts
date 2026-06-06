import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Proxy /api/* → FastAPI on 8080 so browser fetches stay same-origin
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.DAVID_API_URL ?? "http://localhost:8080"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
