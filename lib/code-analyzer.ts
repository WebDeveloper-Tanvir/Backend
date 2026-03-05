// ── Deep static code analyzer ──────────────────────────────────────────────
// Analyzes uploaded code files for errors, warnings, suggestions

export type Severity = "error" | "warning" | "info" | "success"

export interface CodeIssue {
  severity: Severity
  line:     number
  col?:     number
  rule:     string
  message:  string
  fix?:     string
}

export interface AnalysisResult {
  language:    string
  fileName:    string
  lines:       number
  issues:      CodeIssue[]
  score:       number        // 0-100 quality score
  summary:     string
  suggestions: string[]
}

// ── Language detection ─────────────────────────────────────────────────────
export function detectLanguage(fileName: string, code: string): string {
  const ext = fileName.split(".").pop()?.toLowerCase() ?? ""
  const extMap: Record<string, string> = {
    py: "python", js: "javascript", ts: "typescript", jsx: "javascript",
    tsx: "typescript", java: "java", cpp: "cpp", cc: "cpp", c: "c",
    go: "go", rs: "rust", rb: "ruby", php: "php", swift: "swift",
    kt: "kotlin", cs: "csharp", html: "html", css: "css", sql: "sql",
  }
  if (extMap[ext]) return extMap[ext]

  // Heuristic from code content
  if (/def\s+\w+\s*\(|import\s+\w+|print\s*\(/.test(code)) return "python"
  if (/function\s+\w+|const\s+\w+\s*=|console\.log/.test(code)) return "javascript"
  if (/public\s+class|System\.out\.println/.test(code)) return "java"
  if (/#include|std::cout/.test(code)) return "cpp"
  return "unknown"
}

// ── Python analyzer ────────────────────────────────────────────────────────
function analyzePython(code: string, fileName: string): AnalysisResult {
  const issues: CodeIssue[] = []
  const lines = code.split("\n")
  const defined = new Set<string>()
  const imported = new Set<string>()

  lines.forEach((raw, i) => {
    const line = raw
    const trimmed = raw.trim()
    const ln = i + 1

    // Skip blank lines and comments
    if (!trimmed || trimmed.startsWith("#")) return

    // Track imports
    const importMatch = trimmed.match(/^(?:import|from)\s+([\w.]+)/)
    if (importMatch) { imported.add(importMatch[1].split(".")[0]); return }

    // Variable assignments
    const assignMatch = trimmed.match(/^(\w+)\s*=/)
    if (assignMatch) defined.add(assignMatch[1])

    // Trailing whitespace
    if (raw !== raw.trimEnd()) {
      issues.push({ severity: "info", line: ln, rule: "trailing-whitespace",
        message: "Trailing whitespace", fix: "Remove trailing spaces" })
    }

    // Tabs vs spaces
    if (raw.startsWith("\t")) {
      issues.push({ severity: "warning", line: ln, rule: "tabs",
        message: "Tab used for indentation — Python recommends 4 spaces (PEP 8)",
        fix: "Replace tabs with 4 spaces" })
    }

    // Line too long
    if (raw.length > 79) {
      issues.push({ severity: "info", line: ln, rule: "line-length",
        message: `Line length ${raw.length} exceeds PEP 8 limit of 79`,
        fix: "Break line into multiple lines" })
    }

    // Bare except
    if (/^except\s*:/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "bare-except",
        message: "Bare 'except:' catches all exceptions including SystemExit and KeyboardInterrupt",
        fix: "Use 'except Exception as e:' or catch a specific exception type" })
    }

    // == None instead of is None
    if (/==\s*None|!=\s*None/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "none-comparison",
        message: "Comparison to None using == instead of 'is'",
        fix: "Use 'if x is None:' or 'if x is not None:'" })
    }

    // == True/False
    if (/==\s*True|==\s*False/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "bool-comparison",
        message: "Explicit comparison to True/False is redundant",
        fix: "Use 'if x:' or 'if not x:' instead" })
    }

    // Missing colon on control structures
    if (/^(if|elif|else|for|while|def|class|try|except|finally|with)\b/.test(trimmed)) {
      if (!/:\s*$/.test(trimmed) && !trimmed.endsWith("\\")) {
        issues.push({ severity: "error", line: ln, rule: "missing-colon",
          message: `'${trimmed.split(" ")[0]}' block is missing colon at end of line`,
          fix: `Add ':' at the end of the line` })
      }
    }

    // Mutable default argument
    if (/def\s+\w+\s*\(.*=\s*(\[\]|\{\}|\(\))/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "mutable-default",
        message: "Mutable default argument — shared across all calls",
        fix: "Use None as default and create the object inside the function" })
    }

    // print statement (Python 2 style)
    if (/^print\s+[^("]/.test(trimmed)) {
      issues.push({ severity: "error", line: ln, rule: "print-statement",
        message: "Python 2 print statement used — will fail in Python 3",
        fix: "Use print() function: print(...)" })
    }

    // Division that might be integer division
    if (/\d\s*\/\s*\d/.test(trimmed) && !/\/\//.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "division",
        message: "Division with / always returns float in Python 3. Use // for integer division",
        fix: "Use // for integer division if that's the intent" })
    }

    // Unused variable (simple: assigned but name starts with _ convention ignored)
    const varDecl = trimmed.match(/^([a-z]\w*)\s*=\s*[^=]/)
    if (varDecl && !varDecl[1].startsWith("_")) {
      // Very basic: check if used elsewhere in code (word boundary search)
      const varName = varDecl[1]
      const usageInCode = code.replace(raw, "").match(new RegExp(`\\b${varName}\\b`))
      if (!usageInCode && !["result", "data", "response", "output"].includes(varName)) {
        // Only flag as info to avoid false positives
        // issues.push({ severity: "info", line: ln, rule: "unused-var", ... })
      }
    }

    // f-string without format vars
    if (/f["']/.test(trimmed) && !/\{/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "f-string-no-vars",
        message: "f-string with no format variables — plain string would suffice",
        fix: "Remove f prefix: use regular string" })
    }

    // except Exception as e: but e not used
    if (/except\s+\w+\s+as\s+e\s*:/.test(trimmed)) {
      const bodyLine = lines[i + 1]?.trim()
      if (bodyLine && !bodyLine.includes("e")) {
        issues.push({ severity: "info", line: ln, rule: "unused-exception",
          message: "Exception variable 'e' caught but not used in handler",
          fix: "Use 'except ExceptionType:' without 'as e' if you don't need the exception object" })
      }
    }
  })

  return buildResult("python", fileName, lines.length, issues, code)
}

// ── JavaScript / TypeScript analyzer ──────────────────────────────────────
function analyzeJS(code: string, fileName: string, lang: string): AnalysisResult {
  const issues: CodeIssue[] = []
  const lines = code.split("\n")

  lines.forEach((raw, i) => {
    const trimmed = raw.trim()
    const ln = i + 1
    if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("*")) return

    // Trailing whitespace
    if (raw !== raw.trimEnd()) {
      issues.push({ severity: "info", line: ln, rule: "trailing-whitespace",
        message: "Trailing whitespace", fix: "Remove trailing spaces" })
    }

    // var usage
    if (/\bvar\s+/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "no-var",
        message: "'var' is function-scoped and hoisted — prefer const or let",
        fix: "Replace 'var' with 'const' or 'let'" })
    }

    // == instead of ===
    if (/[^=!<>]==[^=]|[^=]!=[^=]/.test(trimmed) && !/==\s*null/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "eqeqeq",
        message: "Loose equality (==) can cause unexpected type coercion",
        fix: "Use strict equality (===) and strict inequality (!==)" })
    }

    // console.log left in
    if (/console\.(log|debug|info)\(/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "no-console",
        message: "console.log statement left in code",
        fix: "Remove debug logs before production or use a proper logging library" })
    }

    // Long line
    if (raw.length > 120) {
      issues.push({ severity: "info", line: ln, rule: "max-len",
        message: `Line length ${raw.length} exceeds recommended 120 characters`,
        fix: "Break into multiple lines" })
    }

    // Missing semicolons (simplified heuristic)
    if (lang === "javascript" && /^(const|let|var|return|throw)\s/.test(trimmed) && !/[;{},]$/.test(trimmed) && !trimmed.endsWith("=>")) {
      issues.push({ severity: "info", line: ln, rule: "semi",
        message: "Missing semicolon",
        fix: "Add semicolon at end of statement" })
    }

    // eval() usage
    if (/\beval\s*\(/.test(trimmed)) {
      issues.push({ severity: "error", line: ln, rule: "no-eval",
        message: "eval() is dangerous — executes arbitrary code",
        fix: "Avoid eval(). Use JSON.parse for data parsing" })
    }

    // Async without await
    if (/async\s+function|async\s+\(/.test(trimmed)) {
      const body = lines.slice(i + 1, i + 20).join("\n")
      if (!body.includes("await")) {
        issues.push({ severity: "warning", line: ln, rule: "require-await",
          message: "async function has no await expression — could be a regular function",
          fix: "Add await before async operations, or remove async keyword" })
      }
    }

    // Promise without .catch()
    if (/\.then\s*\(/.test(trimmed) && !trimmed.includes(".catch") && !lines[i + 1]?.includes(".catch")) {
      issues.push({ severity: "warning", line: ln, rule: "unhandled-promise",
        message: "Promise .then() without .catch() — unhandled rejection possible",
        fix: "Add .catch(err => ...) or use try/catch with async/await" })
    }

    // Infinite loop risk
    if (/while\s*\(\s*true\s*\)/.test(trimmed)) {
      const body = lines.slice(i + 1, i + 15).join("\n")
      if (!body.includes("break")) {
        issues.push({ severity: "error", line: ln, rule: "infinite-loop",
          message: "while(true) loop with no break statement detected",
          fix: "Add a break condition or use a proper loop termination" })
      }
    }

    // == null check (ok, but flag if using ===)
    if (/=== null/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "null-check",
        message: "Checking === null won't catch undefined — use == null to catch both",
        fix: "Use 'x == null' to check for both null and undefined" })
    }

    // TypeScript: any usage
    if (lang === "typescript" && /:\s*any\b/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "no-explicit-any",
        message: "Usage of 'any' type disables TypeScript type safety",
        fix: "Replace 'any' with a specific type or 'unknown'" })
    }

    // TypeScript: non-null assertion
    if (lang === "typescript" && /\w!\.|\w!\[/.test(trimmed)) {
      issues.push({ severity: "warning", line: ln, rule: "non-null-assertion",
        message: "Non-null assertion (!) bypasses TypeScript null checks",
        fix: "Use optional chaining (?.) or proper null check instead" })
    }
  })

  return buildResult(lang, fileName, lines.length, issues, code)
}

// ── Java analyzer ──────────────────────────────────────────────────────────
function analyzeJava(code: string, fileName: string): AnalysisResult {
  const issues: CodeIssue[] = []
  const lines = code.split("\n")

  lines.forEach((raw, i) => {
    const trimmed = raw.trim()
    const ln = i + 1
    if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("*")) return

    if (raw.length > 120) {
      issues.push({ severity: "info", line: ln, rule: "max-len",
        message: `Line too long: ${raw.length} chars`, fix: "Break line" })
    }

    // Null check with ==
    if (/==\s*null|!=\s*null/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "null-check",
        message: "Consider using Optional<T> instead of null checks",
        fix: "Use Optional.ofNullable() and Optional.isPresent()" })
    }

    // Empty catch block
    if (/catch\s*\(/.test(trimmed)) {
      const nextLine = lines[i + 1]?.trim()
      if (nextLine === "}") {
        issues.push({ severity: "error", line: ln, rule: "empty-catch",
          message: "Empty catch block silently swallows exceptions",
          fix: "Log the exception or handle it properly" })
      }
    }

    // System.out.println
    if (/System\.out\.println/.test(trimmed)) {
      issues.push({ severity: "info", line: ln, rule: "no-sysout",
        message: "System.out.println left in code",
        fix: "Use a logging framework like SLF4J/Log4j" })
    }

    // Missing final on loop variable
    if (/for\s*\(int\s+/.test(trimmed) && !trimmed.includes("final")) {
      issues.push({ severity: "info", line: ln, rule: "for-loop-final",
        message: "Loop variable not declared final",
        fix: "Consider 'for (final int i = ...)'" })
    }
  })

  return buildResult("java", fileName, lines.length, issues, code)
}

// ── Generic / HTML / CSS analyzer ─────────────────────────────────────────
function analyzeGeneric(code: string, fileName: string, lang: string): AnalysisResult {
  const issues: CodeIssue[] = []
  const lines = code.split("\n")

  lines.forEach((raw, i) => {
    const ln = i + 1
    if (raw.length > 200) {
      issues.push({ severity: "info", line: ln, rule: "max-len",
        message: `Very long line: ${raw.length} chars`, fix: "Break into multiple lines" })
    }
    if (raw !== raw.trimEnd()) {
      issues.push({ severity: "info", line: ln, rule: "trailing-whitespace",
        message: "Trailing whitespace", fix: "Remove trailing spaces" })
    }
  })

  // HTML specific
  if (lang === "html") {
    if (!code.includes("<!DOCTYPE")) {
      issues.push({ severity: "warning", line: 1, rule: "doctype",
        message: "Missing DOCTYPE declaration",
        fix: "Add <!DOCTYPE html> at the top" })
    }
    if (!code.includes('<meta charset')) {
      issues.push({ severity: "warning", line: 1, rule: "charset",
        message: "Missing charset meta tag",
        fix: 'Add <meta charset="UTF-8">' })
    }
    // img without alt
    const imgNoAlt = [...code.matchAll(/<img(?![^>]*alt=)[^>]*>/g)]
    imgNoAlt.forEach(() => {
      issues.push({ severity: "warning", line: 0, rule: "img-alt",
        message: "<img> tag missing alt attribute",
        fix: 'Add alt="description" to all img tags' })
    })
  }

  return buildResult(lang, fileName, lines.length, issues, code)
}

// ── Build final result with score ──────────────────────────────────────────
function buildResult(lang: string, fileName: string, totalLines: number, issues: CodeIssue[], code: string): AnalysisResult {
  const errors   = issues.filter(i => i.severity === "error").length
  const warnings = issues.filter(i => i.severity === "warning").length
  const infos    = issues.filter(i => i.severity === "info").length

  // Score calculation
  let score = 100
  score -= errors   * 15
  score -= warnings * 5
  score -= infos    * 1
  score = Math.max(0, Math.min(100, score))

  // Suggestions
  const suggestions: string[] = []
  if (errors > 0)   suggestions.push(`Fix ${errors} critical error${errors > 1 ? "s" : ""} before deploying`)
  if (warnings > 0) suggestions.push(`Review ${warnings} warning${warnings > 1 ? "s" : ""} to improve code quality`)
  if (totalLines > 200) suggestions.push("Consider splitting this file into smaller modules")
  if (score >= 90) suggestions.push("Excellent code quality! Minor style improvements available")
  else if (score >= 70) suggestions.push("Good code quality with some improvements needed")
  else if (score >= 50) suggestions.push("Moderate issues detected — review warnings carefully")
  else suggestions.push("Significant issues found — address errors before continuing")

  const summary = errors === 0 && warnings === 0
    ? `✅ No critical issues found in ${fileName}. Code quality score: ${score}/100`
    : `Found ${errors} error${errors !== 1 ? "s" : ""}, ${warnings} warning${warnings !== 1 ? "s" : ""}, ${infos} suggestion${infos !== 1 ? "s" : ""} in ${fileName}`

  return { language: lang, fileName, lines: totalLines, issues, score, summary, suggestions }
}

// ── Main export ────────────────────────────────────────────────────────────
export function analyzeCode(code: string, fileName: string): AnalysisResult {
  const lang = detectLanguage(fileName, code)

  switch (lang) {
    case "python":     return analyzePython(code, fileName)
    case "javascript": return analyzeJS(code, fileName, "javascript")
    case "typescript": return analyzeJS(code, fileName, "typescript")
    case "java":       return analyzeJava(code, fileName)
    default:           return analyzeGeneric(code, fileName, lang)
  }
}

// ── Format analysis as chat message ───────────────────────────────────────
export function formatAnalysisAsMarkdown(result: AnalysisResult): string {
  const { language, fileName, lines, issues, score, summary, suggestions } = result
  const errors   = issues.filter(i => i.severity === "error")
  const warnings = issues.filter(i => i.severity === "warning")
  const infos    = issues.filter(i => i.severity === "info")

  const scoreBar = "█".repeat(Math.round(score / 10)) + "░".repeat(10 - Math.round(score / 10))
  const scoreColor = score >= 80 ? "🟢" : score >= 60 ? "🟡" : "🔴"

  let md = `## Code Analysis: \`${fileName}\`\n\n`
  md += `**Language:** ${language} · **Lines:** ${lines} · **Quality Score:** ${scoreColor} ${score}/100\n`
  md += `\`${scoreBar}\`\n\n`
  md += `${summary}\n\n`

  if (errors.length > 0) {
    md += `### 🔴 Errors (${errors.length})\n`
    errors.forEach(e => {
      md += `**Line ${e.line}** — \`${e.rule}\`\n${e.message}\n`
      if (e.fix) md += `> 💡 Fix: ${e.fix}\n`
      md += "\n"
    })
  }

  if (warnings.length > 0) {
    md += `### 🟡 Warnings (${warnings.length})\n`
    warnings.forEach(w => {
      md += `**Line ${w.line}** — \`${w.rule}\`\n${w.message}\n`
      if (w.fix) md += `> 💡 Fix: ${w.fix}\n`
      md += "\n"
    })
  }

  if (infos.length > 0) {
    md += `### 🔵 Suggestions (${infos.length})\n`
    infos.slice(0, 5).forEach(info => {  // cap at 5 info items
      md += `**Line ${info.line}** — \`${info.rule}\`\n${info.message}\n`
      if (info.fix) md += `> 💡 Fix: ${info.fix}\n`
      md += "\n"
    })
    if (infos.length > 5) md += `_...and ${infos.length - 5} more style suggestions_\n\n`
  }

  if (issues.length === 0) {
    md += `### ✅ No Issues Found!\nYour code looks clean. Great job! 🎉\n\n`
  }

  if (suggestions.length > 0) {
    md += `### 📋 Recommendations\n`
    suggestions.forEach(s => { md += `- ${s}\n` })
  }

  return md
}
