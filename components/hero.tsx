"use client"

import Link from "next/link"
import { ArrowRight, Zap } from "lucide-react"

/* ── Floating circuit node ─────────────────────────────────────────────── */
function Node({ x, y, delay = 0, color = "#00c9a7" }: { x: string; y: string; delay?: number; color?: string }) {
  return (
    <div
      className="absolute flex items-center justify-center"
      style={{ left: x, top: y, animationDelay: `${delay}s` }}
    >
      <div
        className="w-2 h-2 rounded-full animate-pulse-node"
        style={{ background: color, boxShadow: `0 0 10px ${color}`, animationDelay: `${delay}s` }}
      />
    </div>
  )
}

/* ── Inline mock code block ────────────────────────────────────────────── */
const codeLines = [
  { tokens: [{ t: "// AI detected 3 issues", c: "rgba(200,230,220,0.35)" }] },
  { tokens: [
    { t: "function ", c: "#00c9a7" },
    { t: "fetchData", c: "#00b4d8" },
    { t: "(url) {", c: "rgba(200,230,220,0.8)" },
  ]},
  { tokens: [
    { t: "  const res = ", c: "rgba(200,230,220,0.8)" },
    { t: "await ", c: "#7b5ea7" },
    { t: "fetch(url)", c: "#4361ee" },
  ]},
  { tokens: [
    { t: "  // ✓ ", c: "#00c9a7" },
    { t: "Error caught & fixed", c: "rgba(200,230,220,0.55)" },
  ]},
  { tokens: [
    { t: "  return ", c: "#00c9a7" },
    { t: "res.json()", c: "#00b4d8" },
  ]},
  { tokens: [{ t: "}", c: "rgba(200,230,220,0.8)" }] },
]

export function Hero() {
  return (
    <section className="relative w-full overflow-hidden bg-mesh py-20 lg:py-36">

      {/* ── Atmospheric glows ──────────────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-5%] w-[50vw] h-[50vw] max-w-[700px] max-h-[700px] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(0,201,167,0.1) 0%, transparent 70%)" }} />
        <div className="absolute bottom-[-10%] right-[-5%] w-[45vw] h-[45vw] max-w-[600px] max-h-[600px] rounded-full animate-drift"
          style={{ background: "radial-gradient(circle, rgba(67,97,238,0.12) 0%, transparent 70%)" }} />
        <div className="absolute top-[30%] right-[15%] w-[30vw] h-[30vw] max-w-[400px] max-h-[400px] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(0,180,216,0.07) 0%, transparent 70%)" }} />
      </div>

      {/* ── Circuit nodes ──────────────────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <Node x="8%"  y="18%" delay={0}   color="#00c9a7" />
        <Node x="15%" y="62%" delay={0.8} color="#00b4d8" />
        <Node x="88%" y="22%" delay={1.4} color="#4361ee" />
        <Node x="92%" y="68%" delay={0.4} color="#00c9a7" />
        <Node x="50%" y="8%"  delay={2.0} color="#7b5ea7" />

        {/* Connector lines */}
        <svg className="absolute inset-0 w-full h-full opacity-[0.07]" preserveAspectRatio="none">
          <line x1="8%" y1="18%" x2="15%" y2="62%" stroke="#00c9a7" strokeWidth="1" />
          <line x1="88%" y1="22%" x2="92%" y2="68%" stroke="#4361ee" strokeWidth="1" />
          <line x1="50%" y1="8%" x2="88%" y2="22%" stroke="#00b4d8" strokeWidth="1" />
        </svg>
      </div>

      <div className="relative container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row items-center gap-16 lg:gap-20">

          {/* ── Left: Text ─────────────────────────────────────────────── */}
          <div className="flex-1 flex flex-col items-start gap-8 max-w-2xl">

            {/* Badge */}
            <div className="flex items-center gap-2 px-4 py-1.5 rounded-full border"
              style={{ borderColor: "rgba(0,201,167,0.25)", background: "rgba(0,201,167,0.06)" }}>
              <Zap size={13} className="text-[#00c9a7]" />
              <span className="text-xs font-semibold tracking-widest uppercase text-[#00c9a7]">
                AI-Powered · Real-Time · Free
              </span>
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-[3.5rem] font-extrabold leading-[1.1] tracking-tight">
              Never Get Stuck<br />
              on <span className="brand-gradient text-glow">Code Errors</span><br />
              Again
            </h1>

            <p className="text-base sm:text-lg text-[rgba(200,230,220,0.6)] leading-relaxed max-w-lg">
              Errorless uses AI to detect, analyze, and fix code errors in real-time.
              Write cleaner code, ship faster, debug less.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Link href="/sign-up">
                <button className="relative group px-7 py-3 rounded-xl font-bold text-sm overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee] transition-opacity" />
                  <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee] opacity-0 group-hover:opacity-100 blur-md transition-opacity" />
                  <span className="relative flex items-center gap-2 text-[#030e09]">
                    Start Free Trial
                    <ArrowRight size={16} />
                  </span>
                </button>
              </Link>
              <Link href="/ide">
                <button className="px-7 py-3 rounded-xl font-semibold text-sm border text-[rgba(200,230,220,0.8)] hover:text-white hover:border-[rgba(0,201,167,0.4)] transition-all"
                  style={{ borderColor: "rgba(0,201,167,0.2)", background: "rgba(0,201,167,0.04)" }}>
                  Try the IDE →
                </button>
              </Link>
            </div>

            {/* Stats row */}
            <div className="flex items-center gap-8 pt-2 border-t w-full" style={{ borderColor: "rgba(0,201,167,0.1)" }}>
              {[["10K+", "Developers"], ["50M+", "Errors Fixed"], ["99.9%", "Uptime"]].map(([val, label]) => (
                <div key={label} className="flex flex-col gap-0.5">
                  <span className="text-xl font-extrabold brand-gradient">{val}</span>
                  <span className="text-xs text-[rgba(200,230,220,0.45)] tracking-wide">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Right: Code card ───────────────────────────────────────── */}
          <div className="flex-1 w-full max-w-md">
            <div className="relative rounded-2xl overflow-hidden glass glow-teal">
              {/* Scan line animation */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none rounded-2xl">
                <div className="absolute left-0 right-0 h-px"
                  style={{
                    background: "linear-gradient(90deg, transparent, rgba(0,201,167,0.5), transparent)",
                    animation: "scan-line 4s linear infinite",
                  }} />
              </div>

              {/* Window bar */}
              <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: "rgba(0,201,167,0.1)" }}>
                <div className="w-3 h-3 rounded-full bg-[#ff6b6b]" />
                <div className="w-3 h-3 rounded-full bg-[#ffd166]" />
                <div className="w-3 h-3 rounded-full bg-[#00c9a7]" />
                <span className="ml-3 font-mono text-xs text-[rgba(200,230,220,0.4)]">errorless.ai — live fix</span>
                <div className="ml-auto flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#00c9a7] animate-pulse" />
                  <span className="font-mono text-[10px] text-[#00c9a7]">ACTIVE</span>
                </div>
              </div>

              {/* Code body */}
              <div className="p-5 font-mono text-sm leading-7">
                {codeLines.map((line, i) => (
                  <div key={i} className="flex">
                    <span className="w-8 text-right mr-4 select-none text-[rgba(200,230,220,0.2)] text-xs leading-7">
                      {i + 1}
                    </span>
                    <span>
                      {line.tokens.map((tok, j) => (
                        <span key={j} style={{ color: tok.c }}>{tok.t}</span>
                      ))}
                    </span>
                  </div>
                ))}
                {/* Blinking cursor */}
                <div className="flex mt-1">
                  <span className="w-8 mr-4 text-[rgba(200,230,220,0.2)] text-xs leading-7 text-right">7</span>
                  <span className="w-2 h-5 rounded-sm bg-[#00c9a7] cursor-blink inline-block" />
                </div>
              </div>

              {/* AI panel */}
              <div className="mx-4 mb-4 p-3 rounded-xl" style={{ background: "rgba(0,201,167,0.06)", border: "1px solid rgba(0,201,167,0.15)" }}>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#00c9a7] animate-pulse" />
                  <span className="font-mono text-[10px] font-semibold text-[#00c9a7] tracking-widest uppercase">Errorless AI</span>
                </div>
                <p className="font-mono text-xs text-[rgba(200,230,220,0.65)] leading-5">
                  ✓ Missing error handling detected<br />
                  ✓ Async/await optimized<br />
                  ✓ Applied fix in <span className="text-[#00c9a7]">12ms</span>
                </p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
