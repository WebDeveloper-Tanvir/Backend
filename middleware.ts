// middleware.ts  (root of project, alongside package.json)
//
// Clerk middleware — protects authenticated routes.
// Public routes (login, signup, home, sso-callback) are accessible without auth.
// Everything else requires a signed-in session.

import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server"

const isPublicRoute = createRouteMatcher([
  "/",
  "/login(.*)",
  "/signup(.*)",
  "/sso-callback(.*)",
  "/pricing(.*)",
  "/learn(.*)",
  "/api/webhooks/(.*)",   // Clerk webhook must be public
])

export default clerkMiddleware(async (auth, request) => {
  if (!isPublicRoute(request)) {
    await auth.protect()
  }
})

export const config = {
  matcher: [
    // Run on all routes except static files and Next.js internals
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
}