import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ET-Agent",
  description: "농업 자격증 및 교육 관련 AI 에이전트",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body className={"antialiased"}>
        {children}
      </body>
    </html>
  );
}
