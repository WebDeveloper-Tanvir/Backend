"use client"

import { IDEContainer } from "@/components/ide/ide-container"
import { useClerkConvexSync } from "@/hooks/use-clerk-convex-sync"

export default function IDEPage() {
  useClerkConvexSync()

  return <IDEContainer />
}
