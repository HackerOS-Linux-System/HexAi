/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  distDir: 'out',
  images: {
    unoptimized: true,
  },
  // Transpile react-syntax-highlighter so Next.js handles its CJS/ESM mix
  transpilePackages: ['react-syntax-highlighter'],
};
module.exports = nextConfig;
