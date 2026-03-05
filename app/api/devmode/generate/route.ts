import { NextRequest, NextResponse } from "next/server"

const GEMINI_KEY = process.env.GEMINI_API_KEY || ""
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_KEY}`

export async function POST(req: NextRequest) {
  try {
    const { prompt, language, plan } = await req.json()

    if (!prompt || !language) {
      return NextResponse.json({ error: "prompt and language are required" }, { status: 400 })
    }

    const systemPrompt = `You are an expert ${language} developer on Errorless Dev Mode.
Generate clean, production-ready, well-commented ${language} code.

Rules:
- Write only the code — no explanations before or after unless it's a comment inside the code
- Include helpful inline comments
- Handle edge cases and errors properly
- Follow best practices for ${language}
- ${plan === "premium" ? "Include performance optimizations and thorough documentation" : "Write clean, readable code"}`

    const userPrompt = `Write ${language} code for: ${prompt}

Return ONLY the raw code with no markdown fences, no preamble, no explanation after.`

    if (!GEMINI_KEY) {
      return NextResponse.json({ code: getFallbackCode(language, prompt) })
    }

    const response = await fetch(GEMINI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents: [{ role: "user", parts: [{ text: userPrompt }] }],
        generationConfig: { temperature: 0.2, maxOutputTokens: 4096 },
      }),
    })

    if (!response.ok) {
      return NextResponse.json({ code: getFallbackCode(language, prompt) })
    }

    const data = await response.json()
    let code = data.candidates?.[0]?.content?.parts?.[0]?.text || ""

    // Strip markdown fences if AI added them
    code = code.replace(/^```[\w]*\n?/gm, "").replace(/```$/gm, "").trim()

    return NextResponse.json({ code })
  } catch (err) {
    console.error("[DevMode] generate error:", err)
    return NextResponse.json({ error: "Generation failed" }, { status: 500 })
  }
}

function getFallbackCode(language: string, prompt: string): string {
  const templates: Record<string, string> = {
    python: `# ${prompt}\n# Add GEMINI_API_KEY to .env.local for real AI generation\n\ndef main():\n    # Your implementation here\n    print("Hello from Errorless Dev Mode!")\n\nif __name__ == "__main__":\n    main()`,
    javascript: `// ${prompt}\n// Add GEMINI_API_KEY to .env.local for real AI generation\n\nfunction main() {\n  // Your implementation here\n  console.log("Hello from Errorless Dev Mode!");\n}\n\nmain();`,
    typescript: `// ${prompt}\n// Add GEMINI_API_KEY to .env.local for real AI generation\n\nfunction main(): void {\n  // Your implementation here\n  console.log("Hello from Errorless Dev Mode!");\n}\n\nmain();`,
  }
  return templates[language] || `// ${prompt}\n// Add GEMINI_API_KEY to .env.local for real AI generation\n// Language: ${language}`
}