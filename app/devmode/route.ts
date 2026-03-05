/**
 * app/api/devmode/route.ts
 *
 * Next.js edge that proxies DevMode requests to your Python AI server.
 * 
 * Replace ERRORLESS_AI_URL with your Python server URL:
 *   - Local dev:    http://localhost:8000
 *   - Railway/Fly:  https://your-app.railway.app
 *   - Render:       https://your-app.onrender.com
 *
 * POST /api/devmode
 * Body: { mode, prompt?, code?, language, plan, context? }
 */

import { NextRequest, NextResponse } from "next/server"
import { auth } from "@clerk/nextjs/server"

const AI_URL = process.env.ERRORLESS_AI_URL ?? "http://localhost:8000"

// Map DevMode UI modes → Python server endpoints
const ENDPOINT: Record<string, string> = {
  generate: "/generate",
  analyze:  "/analyze",
  optimize: "/optimize",
  agent:    "/agent",        // emergent agent mode
}

export async function POST(req: NextRequest) {
  // Auth check
  const { userId } = await auth()
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

  const body = await req.json()
  const { mode = "generate", plan = "basic" } = body

  // Plan gate: optimize & agent require premium
  if ((mode === "optimize" || mode === "agent") && plan !== "premium") {
    return NextResponse.json(
      { error: "This feature requires a Premium plan.", upgrade: true },
      { status: 403 }
    )
  }

  const endpoint = ENDPOINT[mode]
  if (!endpoint) {
    return NextResponse.json({ error: `Unknown mode: ${mode}` }, { status: 400 })
  }

  try {
    const aiResp = await fetch(`${AI_URL}${endpoint}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ ...body, userId }),
      signal:  AbortSignal.timeout(90_000),  // 90s for agent mode
    })

    const data = await aiResp.json()
    return NextResponse.json(data, { status: aiResp.ok ? 200 : 502 })

  } catch (err) {
    console.error("[DevMode API]", err)
    return NextResponse.json({ error: "AI server unavailable" }, { status: 503 })
  }
}

// SSE streaming proxy for /generate?stream=true
export async function GET(req: NextRequest) {
  const { userId } = await auth()
  if (!userId) return NextResponse.json({ error: "Unauthorized" }, { status: 401 })

  const { searchParams } = req.nextUrl
  const prompt   = searchParams.get("prompt") ?? ""
  const language = searchParams.get("language") ?? "python"
  const context  = searchParams.get("context") ?? ""

  const aiResp = await fetch(`${AI_URL}/generate/stream`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ prompt, language, context, stream: true }),
  })

  return new NextResponse(aiResp.body, {
    headers: {
      "Content-Type":  "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection":    "keep-alive",
    },
  })
}