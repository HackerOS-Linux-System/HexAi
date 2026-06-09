/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  distDir: 'out',
  // When embedded in Axum under /gui, set basePath so assets load correctly
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
  images: { unoptimized: true },
  transpilePackages: ['react-syntax-highlighter'],
};
module.exports = nextConfig;
