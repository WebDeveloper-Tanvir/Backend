import type React from "react"
import type { Metadata, Viewport } from "next"
import { Analytics } from "@vercel/analytics/next"
import { ClerkClientProvider } from "@/components/clerk-provider"
import { ConvexClientProvider } from "@/components/convex-provider"
import "./globals.css"

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#030e09",
}

export const metadata: Metadata = {
  title: "Errorless - AI-Powered Code Error Solver | Real-Time Debugging",
  description:
    "Errorless is an AI-powered code error solver that provides context-aware live error detection and fixes. Collaborate with your team, share fixes, and build better code faster.",
  keywords: [
    "code error solver", "AI debugging", "code assistant",
    "error detection", "live error solver", "collaborative coding", "developer tools",
  ],
  applicationName: "Errorless",
  authors: [{ name: "Errorless Team" }],
  creator: "Errorless",
  publisher: "Errorless",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/errorless-logo.jpg", type: "image/jpeg" },
    ],
    apple: [{ url: "/errorless-logo.jpg" }],
    shortcut: "/favicon.ico",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-video-preview": -1, "max-image-preview": "large", "max-snippet": -1 },
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://errorless.dev",
    siteName: "Errorless",
    title: "Errorless - AI-Powered Code Error Solver",
    description: "Real-time error detection and AI-powered fixes for developers",
    images: [{ url: "/errorless-logo.jpg", width: 1024, height: 1024, alt: "Errorless Logo" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Errorless - AI-Powered Code Error Solver",
    description: "Real-time error detection and AI-powered fixes for developers",
    images: ["/errorless-logo.jpg"],
  },
  alternates: { canonical: "https://errorless.dev" },
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/errorless-logo.jpg" type="image/jpeg" />
        <link rel="apple-touch-icon" href="/errorless-logo.jpg" />
        <link rel="shortcut icon" href="/favicon.ico" />

        {/* Brand theme color */}
        <meta name="theme-color" content="#030e09" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />

        {/* Structured data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              name: "Errorless",
              description: "AI-powered code error solver with real-time debugging and team collaboration",
              url: "https://errorless.dev",
              applicationCategory: "DeveloperApplication",
              offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
              aggregateRating: { "@type": "AggregateRating", ratingValue: "4.8", ratingCount: "1250" },
            }),
          }}
        />
      </head>
      <body className="antialiased">
        <ClerkClientProvider>
          <ConvexClientProvider>
            {children}
            <Analytics />
          </ConvexClientProvider>
        </ClerkClientProvider>
      </body>
    </html>
  )
}
