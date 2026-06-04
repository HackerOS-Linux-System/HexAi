/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  distDir: 'out',
  images: { unoptimized: true },
  transpilePackages: ['react-syntax-highlighter'],
};
module.exports = nextConfig;
