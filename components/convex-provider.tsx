"use client"

import { ConvexProvider, ConvexReactClient } from "convex/react"
import type { ReactNode } from "react"

// Get Convex URL from environment - in preview/local it comes from .env.local
const convexUrl = process.env.NEXT_PUBLIC_CONVEX_URL

if (!convexUrl) {
  console.error(
    "[v0] NEXT_PUBLIC_CONVEX_URL is not set. Please ensure the environment variable is configured in your Vercel project settings or .env.local file."
  )
}

// Create client only if URL is available, otherwise create a placeholder
const convex = convexUrl ? new ConvexReactClient(convexUrl) : null

export function ConvexClientProvider({ children }: { children: ReactNode }) {
  if (!convex) {
    return (
      <div className="flex items-center justify-center h-screen bg-background text-foreground">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-2">Configuration Required</h1>
          <p className="text-muted-foreground mb-4">
            Convex backend is not configured. Please set NEXT_PUBLIC_CONVEX_URL environment variable.
          </p>
          <p className="text-sm text-muted-foreground">Check the browser console for more details.</p>
        </div>
      </div>
    )
  }

  return <ConvexProvider client={convex}>{children}</ConvexProvider>
}
