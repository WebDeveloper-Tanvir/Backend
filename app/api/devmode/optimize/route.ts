import { NextRequest, NextResponse } from "next/server"

const GEMINI_KEY = process.env.GEMINI_API_KEY || ""
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_KEY}`

export async function POST(req: NextRequest) {
  try {
    const { code, language, analysis } = await req.json()

    if (!code || !language) {
      return NextResponse.json({ error: "code and language are required" }, { status: 400 })
    }

    const analysisContext = analysis
      ? `\nPREVIOUS ANALYSIS FINDINGS:\n- Issues: ${analysis.issues || "none"}\n- Performance: ${analysis.performance || "none"}\n- Suggestions: ${analysis.suggestions || "none"}`
      : ""

    const prompt = `You are a world-class ${language} engineer. Optimize and refactor this code to be production-ready.
${analysisContext}

ORIGINAL CODE:
\`\`\`${language}
${code.slice(0, 8000)}
\`\`\`

Produce an optimized version that:
1. Fixes all bugs and issues identified
2. Improves time and space complexity where possible
3. Follows ${language} best practices and idioms
4. Adds proper error handling
5. Improves readability with clear naming and comments
6. Adds type hints/annotations where applicable
7. Removes dead code and redundancy
8. Makes it production-grade

Return ONLY the optimized code — no markdown fences, no before/after comparison, no explanation.
Start directly with the code.`

    if (!GEMINI_KEY) {
      return NextResponse.json({
        optimized: `# Optimized version of your code\n# Add GEMINI_API_KEY to .env.local for real AI optimization\n\n${code}`,
      })
    }

    const response = await fetch(GEMINI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.15, maxOutputTokens: 6000 },
      }),
    })

    if (!response.ok) {
      return NextResponse.json({ optimized: code, error: "Optimization service unavailable. Returning original code." })
    }

    const data = await response.json()
    let optimized = data.candidates?.[0]?.content?.parts?.[0]?.text || code

    // Strip any markdown fences the model might have added
    optimized = optimized.replace(/^```[\w]*\n?/gm, "").replace(/```$/gm, "").trim()

    return NextResponse.json({ optimized })
  } catch (err) {
    console.error("[DevMode] optimize error:", err)
    return NextResponse.json({ error: "Optimization failed. Please try again." }, { status: 500 })
  }
}