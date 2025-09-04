import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {
    // 하이드레이션 오류 방지를 위한 설정
    optimizePackageImports: ['@copilotkit/react-core', '@copilotkit/react-ui'],
  },
  // SSR 최적화
  swcMinify: true,
  // 컴파일러 옵션
  compiler: {
    // 개발 환경에서 하이드레이션 경고 제거
    removeConsole: process.env.NODE_ENV === 'production',
  },
  // 하이드레이션 오류 방지
  reactStrictMode: false,
  // 개발 환경에서 하이드레이션 경고 억제
  onDemandEntries: {
    // 개발 환경에서 하이드레이션 경고 억제
    maxInactiveAge: 25 * 1000,
    pagesBufferLength: 2,
  },
};

export default nextConfig;
