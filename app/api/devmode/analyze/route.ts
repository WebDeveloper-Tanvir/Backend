import { NextRequest, NextResponse } from "next/server"

const GEMINI_KEY = process.env.GEMINI_API_KEY || ""
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_KEY}`

export async function POST(req: NextRequest) {
  try {
    const { code, language, plan } = await req.json()

    if (!code || !language) {
      return NextResponse.json({ error: "code and language are required" }, { status: 400 })
    }

    const prompt = `Analyze this ${language} code thoroughly as a senior developer. 

CODE:
\`\`\`${language}
${code.slice(0, 8000)}
\`\`\`

Return ONLY a JSON object with these exact keys (no markdown, no extra text):
{
  "summary": "2-3 sentences: what this code does and its overall quality",
  "complexity": "Time complexity, space complexity, cyclomatic complexity. Be specific with Big-O where possible.",
  "issues": "List any bugs, logic errors, or code smells found. Use newlines to separate items. Say 'No issues found' if clean.",
  "security": "Security vulnerabilities, injection risks, unsafe operations. Say 'No security issues found' if clean.",
  "performance": "Performance bottlenecks, inefficient algorithms, memory issues. Specific line references if possible.",
  "suggestions": "Top 3-5 actionable improvements the developer should make, numbered."
}`

    if (!GEMINI_KEY) {
      return NextResponse.json(getOfflineAnalysis(code, language))
    }

    const response = await fetch(GEMINI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.1,
          maxOutputTokens: 3000,
          responseMimeType: "application/json",
        },
      }),
    })

    if (!response.ok) {
      return NextResponse.json(getOfflineAnalysis(code, language))
    }

    const data = await response.json()
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || ""

    try {
      const clean = text.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim()
      const parsed = JSON.parse(clean)
      return NextResponse.json(parsed)
    } catch {
      const match = text.match(/\{[\s\S]*\}/)
      if (match) return NextResponse.json(JSON.parse(match[0]))
      return NextResponse.json(getOfflineAnalysis(code, language))
    }
  } catch (err) {
    console.error("[DevMode] analyze error:", err)
    return NextResponse.json({ error: "Analysis failed. Please try again." }, { status: 500 })
  }
}

function getOfflineAnalysis(code: string, language: string) {
  const lines = code.split("\n").length
  const chars = code.length
  const funcs = (code.match(/def |function |const .* = \(|=>|\bfunc\b/g) || []).length
  const hasLoops = /for |while |forEach|\.map\(/.test(code)
  const hasIO = /print|console\.|System\.out|fmt\.Print/.test(code)
  const hasImports = /import |require\(|#include/.test(code)

  return {
    summary: `${language.charAt(0).toUpperCase() + language.slice(1)} file with ${lines} lines and approximately ${funcs} function(s). ${hasIO ? "Contains I/O operations." : ""} Add GEMINI_API_KEY for deep AI analysis.`,
    complexity: `File size: ${lines} lines, ${chars} characters.\nFunctions detected: ~${funcs}\nLoops detected: ${hasLoops ? "Yes" : "No"}\nAdd GEMINI_API_KEY for Big-O analysis.`,
    issues: `Static scan only — add GEMINI_API_KEY for deep bug detection.\n${lines > 200 ? "⚠️ File is long — consider breaking into modules." : "File length looks reasonable."}`,
    security: `Add GEMINI_API_KEY for security analysis.\n${code.includes("eval(") ? "⚠️ eval() detected — potential security risk." : "No obvious eval() usage detected."}`,
    performance: `${hasLoops ? "Loops detected — verify no O(n²) nesting." : "No loops detected."}\nAdd GEMINI_API_KEY for detailed performance analysis.`,
    suggestions: `1. Add GEMINI_API_KEY to .env.local for AI-powered analysis\n2. ${hasImports ? "Ensure all imports are necessary" : "Add appropriate imports"}\n3. Add unit tests for all functions\n4. Consider adding type annotations\n5. Add error handling where missing`,
  }
}