"use client"

import { useEffect } from "react"
import { useUser } from "@clerk/nextjs"

// This hook syncs Clerk user data to Convex database
// It will be fully functional after Convex deployment
export function useClerkConvexSync() {
  const { user, isLoaded } = useUser()

  useEffect(() => {
    if (!isLoaded || !user) return

    // Store user info in session storage temporarily
    // This will be replaced with Convex mutation after deployment
    try {
      const userData = {
        clerkId: user.id,
        email: user.emailAddresses[0]?.emailAddress || "",
        name: user.firstName || user.username || "",
        profileImage: user.imageUrl,
      }
      sessionStorage.setItem("convex_user_sync", JSON.stringify(userData))
    } catch (error) {
      console.error("[v0] Error storing user data:", error)
    }
  }, [isLoaded, user])
}
