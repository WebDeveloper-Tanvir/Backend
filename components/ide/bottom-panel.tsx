"use client"

import React, { useState, useRef, useEffect, useCallback } from "react"
import {
  Play, Trash2, Copy, AlertCircle, AlertTriangle, Info,
  Terminal as TerminalIcon, Bug, MonitorDot, Network,
  ChevronDown, ChevronUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { CodeExecutor } from "@/lib/code-executor"
// import type { DetectedProblem } from "@/lib/code-executor"

// ── Types ──────────────────────────────────────────────────────────────────
export type PanelTab = "problems" | "output" | "debug" | "terminal" | "ports"

export interface Problem {
  id: string
  severity: "error" | "warning" | "info"
  message: string
  file: string
  line: number
  column: number
  source?: string
}

interface LogLine {
  id: string          // guaranteed unique — never use index alone
  text: string
  color: string       // tailwind text class
}

interface Port {
  port: number
  protocol: "http" | "https"
  label: string
  origin: string
  running: boolean
}

interface BottomPanelProps {
  currentLanguage: string
  currentCode: string
  currentFile?: string
  problems?: Problem[]
}

// ── Unique ID helper — uses counter + random to avoid timestamp collisions ──
let _seq = 0
function nextId(): string {
  return `${++_seq}-${Math.random().toString(36).slice(2, 6)}`
}

function mkLine(text: string, color = "text-foreground"): LogLine {
  return { id: nextId(), text, color }
}

// ── Static colour helpers ──────────────────────────────────────────────────
function termColor(text: string): string {
  if (text.startsWith("$"))  return "text-green-400"
  if (text.startsWith("✓"))  return "text-green-300"
  if (text.startsWith("✗") || /error/i.test(text)) return "text-red-400"
  if (text.startsWith(">"))  return "text-blue-300"
  return "text-gray-300"
}

function debugColor(text: string): string {
  if (text.startsWith("[")) return "text-muted-foreground"
  if (/error/i.test(text))  return "text-red-400"
  if (/succeed|code 0/i.test(text)) return "text-green-400"
  return "text-foreground"
}

function outputColor(type: "info" | "success" | "error" | "warn" | "system"): string {
  switch (type) {
    case "error":  return "text-red-400"
    case "warn":   return "text-yellow-400"
    case "success":return "text-green-400"
    case "system": return "text-muted-foreground"
    default:       return "text-foreground"
  }
}

// ── Static data ────────────────────────────────────────────────────────────
const PORTS: Port[] = [
  { port: 3000, protocol: "http",  label: "Next.js Dev", origin: "localhost", running: true  },
  { port: 8080, protocol: "http",  label: "API Server",  origin: "localhost", running: false },
]

const TABS: { id: PanelTab; label: string; icon: React.ReactNode }[] = [
  { id: "problems", label: "Problems",      icon: <AlertCircle  size={13} /> },
  { id: "output",   label: "Output",        icon: <MonitorDot   size={13} /> },
  { id: "debug",    label: "Debug Console", icon: <Bug          size={13} /> },
  { id: "terminal", label: "Terminal",      icon: <TerminalIcon size={13} /> },
  { id: "ports",    label: "Ports",         icon: <Network      size={13} /> },
]

function ts(): string {
  return new Date().toLocaleTimeString("en-US", { hour12: false })
}

// ── Component ──────────────────────────────────────────────────────────────
export function BottomPanel({
  currentLanguage,
  currentCode,
  currentFile = "untitled",
  problems = [],
}: BottomPanelProps) {
  const [activeTab,    setActiveTab]    = useState<PanelTab>("terminal")
  const [collapsed,    setCollapsed]    = useState(false)
  const [height,       setHeight]       = useState(220)
  const [isRunning,    setIsRunning]    = useState(false)
  const [termInput,    setTermInput]    = useState("")

  // Each tab has its own line array (LogLine[]) for safe keying
  const [outputLines,  setOutputLines]  = useState<LogLine[]>([
    mkLine("Errorless IDE — Output ready", "text-muted-foreground"),
  ])
  const [debugLines,   setDebugLines]   = useState<LogLine[]>([
    mkLine("Debug console ready. Run your code to see evaluation.", "text-muted-foreground"),
  ])
  const [termLines,    setTermLines]    = useState<LogLine[]>([
    mkLine("Welcome to Errorless Terminal", "text-green-400"),
    mkLine("Type or press Run Code to execute the active file.", "text-gray-400"),
  ])
  // Problems detected during last run
  const [runProblems,  setRunProblems]  = useState<Problem[]>([])

  const outputRef  = useRef<HTMLDivElement>(null)
  const debugRef   = useRef<HTMLDivElement>(null)
  const termRef    = useRef<HTMLDivElement>(null)
  const inputRef   = useRef<HTMLInputElement>(null)
  const [executor] = useState(() => new CodeExecutor())

  // Drag-to-resize state
  const dragging   = useRef(false)
  const dragStartY = useRef(0)
  const dragStartH = useRef(0)

  // Register global drag listeners once
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return
      const delta = dragStartY.current - e.clientY
      setHeight(h => Math.max(120, Math.min(640, dragStartH.current + delta)))
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseup",   onUp)
    return () => {
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("mouseup",   onUp)
    }
  }, [])

  // Auto-scroll all panels
  useEffect(() => { outputRef.current && (outputRef.current.scrollTop = outputRef.current.scrollHeight) }, [outputLines])
  useEffect(() => { debugRef.current  && (debugRef.current.scrollTop  = debugRef.current.scrollHeight)  }, [debugLines])
  useEffect(() => { termRef.current   && (termRef.current.scrollTop   = termRef.current.scrollHeight)   }, [termLines])

  // ── Add helpers ────────────────────────────────────────────────────────
  const addOutput = useCallback((text: string, type: "info"|"success"|"error"|"warn"|"system" = "info") => {
    setOutputLines(p => [...p, mkLine(text, outputColor(type))])
  }, [])

  const addDebug = useCallback((text: string) => {
    setDebugLines(p => [...p, mkLine(text, debugColor(text))])
  }, [])

  const addTerm = useCallback((text: string) => {
    setTermLines(p => [...p, mkLine(text, termColor(text))])
  }, [])

  // ── Run code ───────────────────────────────────────────────────────────
  const runCode = useCallback(async () => {
    if (!currentCode.trim()) {
      addOutput("No code to execute.", "warn")
      addTerm("✗ No code to execute.")
      return
    }
    setIsRunning(true)

    // Switch to terminal and show activity immediately
    setActiveTab("terminal")

    const lang = currentLanguage
    addOutput(`[${ts()}] Running ${lang}  ›  ${currentFile}`, "system")
    addDebug(`[${ts()}] Starting execution: ${currentFile}`)
    addTerm(`$ run ${currentFile}`)
    addTerm(`> Executing ${lang} code…`)

    try {
      const result = await executor.executeCode(currentCode, lang)

      if (result.success) {
        const lines = result.output.split("\n")

        // ── Terminal: show every output line ──
        lines.forEach(l => addTerm(l || " "))
        addTerm(`✓ Done (${result.executionTime}ms)`)

        // ── Output tab ──
        lines.forEach(l => addOutput(l, "success"))
        addOutput(`✓ Execution completed in ${result.executionTime}ms`, "system")

        // ── Debug tab ──
        addDebug(`[${ts()}] Execution succeeded`)
        lines.forEach(l => addDebug(`  › ${l}`))
        addDebug(`[${ts()}] Process exited with code 0  (${result.executionTime}ms)`)

        // ── Problems from executor ──
        if (result.problems?.length) {
          const mapped: Problem[] = result.problems.map((p: { severity: any; message: any; line: any; column: any; source: any }) => ({
            id: nextId(),
            severity: p.severity,
            message: p.message,
            file: currentFile,
            line: p.line,
            column: p.column,
            source: p.source,
          }))
          setRunProblems(mapped)
        } else {
          setRunProblems([])
        }
      } else {
        const err = result.error ?? "Unknown error"
        addTerm(`✗ Error: ${err}`)
        addOutput(`✗ ${err}`, "error")
        addOutput(`✗ Failed after ${result.executionTime}ms`, "system")
        addDebug(`[${ts()}] Runtime Error: ${err}`)
        addDebug(`[${ts()}] Process exited with code 1`)
        setRunProblems([{
          id: nextId(),
          severity: "error",
          message: err,
          file: currentFile,
          line: 0, column: 0,
          source: "runtime",
        }])
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      addTerm(`✗ ${msg}`)
      addOutput(`✗ ${msg}`, "error")
      setRunProblems([{ id: nextId(), severity: "error", message: msg, file: currentFile, line: 0, column: 0, source: "executor" }])
    } finally {
      setIsRunning(false)
    }
  }, [currentCode, currentLanguage, currentFile, executor, addOutput, addDebug, addTerm])

  // ── Terminal keyboard input ────────────────────────────────────────────
  const handleTermKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== "Enter") return
    const cmd = termInput.trim()
    setTermInput("")
    if (!cmd) return
    addTerm(`$ ${cmd}`)
    if (cmd === "clear" || cmd === "cls") { setTermLines([]); return }
    if (cmd === "run")  { runCode(); return }
    if (cmd === "help") {
      addTerm("Available commands:")
      addTerm("  run    — execute current file")
      addTerm("  clear  — clear terminal")
      addTerm("  help   — show this message")
      return
    }
    addTerm(`bash: ${cmd}: command not found`)
  }

  // ── Badge counts ───────────────────────────────────────────────────────
  const allProblems  = [...problems, ...runProblems]
  const errorCount   = allProblems.filter(p => p.severity === "error").length
  const warnCount    = allProblems.filter(p => p.severity === "warning").length

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div
      className="relative flex flex-col border-t border-border bg-background shrink-0"
      style={{ height: collapsed ? 36 : height }}
    >
      {/* Drag handle */}
      {!collapsed && (
        <div
          className="absolute top-0 left-0 right-0 h-1 cursor-row-resize z-10 hover:bg-primary/30 transition-colors"
          onMouseDown={e => {
            dragging.current   = true
            dragStartY.current = e.clientY
            dragStartH.current = height
            e.preventDefault()
          }}
        />
      )}

      {/* ── Tab bar ──────────────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-1 bg-card border-b border-border shrink-0 select-none"
        style={{ height: 36 }}
      >
        {/* Tabs */}
        <div className="flex items-center h-full overflow-x-auto">
          {TABS.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => { setActiveTab(id); if (collapsed) setCollapsed(false) }}
              className={`flex items-center gap-1.5 px-3 h-full text-xs font-medium border-b-2 whitespace-nowrap transition-colors ${
                activeTab === id && !collapsed
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {icon}
              {label}
              {id === "problems" && (errorCount + warnCount) > 0 && (
                <span className="flex items-center gap-0.5 ml-1">
                  {errorCount > 0 && (
                    <span className="bg-red-500 text-white text-[10px] px-1 min-w-[16px] text-center rounded-sm">
                      {errorCount}
                    </span>
                  )}
                  {warnCount > 0 && (
                    <span className="bg-yellow-500 text-black text-[10px] px-1 min-w-[16px] text-center rounded-sm">
                      {warnCount}
                    </span>
                  )}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 ml-2 shrink-0">
          <Button
            size="sm" variant="ghost"
            onClick={runCode}
            disabled={isRunning}
            className="h-6 px-2 gap-1 text-xs text-green-400 hover:text-green-300 hover:bg-green-400/10"
          >
            <Play size={12} />
            {isRunning ? "Running…" : "Run Code"}
          </Button>

          {/* Clear active tab */}
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0" title="Clear"
            onClick={() => {
              if (activeTab === "output")   setOutputLines([mkLine("Output cleared", "text-muted-foreground")])
              if (activeTab === "debug")    setDebugLines([mkLine("Debug console cleared.", "text-muted-foreground")])
              if (activeTab === "terminal") setTermLines([])
              if (activeTab === "problems") setRunProblems([])
            }}
          >
            <Trash2 size={12} />
          </Button>

          {/* Copy active tab */}
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0" title="Copy"
            onClick={() => {
              const txt =
                activeTab === "output"   ? outputLines.map(l => l.text).join("\n") :
                activeTab === "debug"    ? debugLines.map(l => l.text).join("\n") :
                activeTab === "terminal" ? termLines.map(l => l.text).join("\n") : ""
              if (txt) navigator.clipboard.writeText(txt)
            }}
          >
            <Copy size={12} />
          </Button>

          {/* Collapse */}
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0"
            onClick={() => setCollapsed(c => !c)}
          >
            {collapsed ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </Button>
        </div>
      </div>

      {/* ── Tab content ───────────────────────────────────────────────── */}
      {!collapsed && (
        <div className="flex-1 overflow-hidden min-h-0">

          {/* PROBLEMS */}
          {activeTab === "problems" && (
            <div className="h-full overflow-y-auto text-xs">
              {allProblems.length === 0 ? (
                <p className="text-muted-foreground p-3 font-mono">No problems detected.</p>
              ) : (
                <table className="w-full">
                  <thead className="sticky top-0 bg-card z-10">
                    <tr className="text-muted-foreground border-b border-border">
                      <th className="text-left py-1.5 px-2 font-normal w-6" />
                      <th className="text-left py-1.5 px-2 font-normal">Message</th>
                      <th className="text-left py-1.5 px-2 font-normal w-32">File</th>
                      <th className="text-left py-1.5 px-2 font-normal w-16">Line</th>
                      <th className="text-left py-1.5 px-2 font-normal w-20">Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allProblems.map(p => (
                      <tr key={p.id} className="hover:bg-muted/40 cursor-pointer border-b border-border/30 font-mono">
                        <td className="py-1 px-2">
                          {p.severity === "error"   && <AlertCircle  size={13} className="text-red-500"    />}
                          {p.severity === "warning" && <AlertTriangle size={13} className="text-yellow-500" />}
                          {p.severity === "info"    && <Info          size={13} className="text-blue-400"   />}
                        </td>
                        <td className="py-1 px-2 text-foreground">{p.message}</td>
                        <td className="py-1 px-2 text-muted-foreground truncate max-w-[8rem]">{p.file}</td>
                        <td className="py-1 px-2 text-muted-foreground">{p.line}:{p.column}</td>
                        <td className="py-1 px-2 text-muted-foreground">{p.source ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* OUTPUT */}
          {activeTab === "output" && (
            <div ref={outputRef} className="h-full overflow-y-auto p-3 font-mono text-xs space-y-px">
              {outputLines.map(line => (
                <div key={line.id} className={`whitespace-pre-wrap break-all leading-5 ${line.color}`}>
                  {line.text}
                </div>
              ))}
            </div>
          )}

          {/* DEBUG CONSOLE */}
          {activeTab === "debug" && (
            <div className="h-full flex flex-col">
              <div ref={debugRef} className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-px">
                {debugLines.map(line => (
                  <div key={line.id} className={`whitespace-pre-wrap break-all leading-5 ${line.color}`}>
                    {line.text}
                  </div>
                ))}
              </div>
              <div className="border-t border-border flex items-center px-3 py-1 gap-2 shrink-0">
                <span className="text-muted-foreground text-xs font-mono">&gt;</span>
                <input
                  type="text"
                  placeholder="Evaluate expression…"
                  className="flex-1 bg-transparent text-xs font-mono outline-none text-foreground placeholder:text-muted-foreground"
                  onKeyDown={e => {
                    if (e.key !== "Enter") return
                    const expr = e.currentTarget.value.trim()
                    if (!expr) return
                    addDebug(`> ${expr}`)
                    addDebug(`← [evaluated in browser context]`)
                    e.currentTarget.value = ""
                  }}
                />
              </div>
            </div>
          )}

          {/* TERMINAL */}
          {activeTab === "terminal" && (
            <div className="h-full flex flex-col bg-[#0d0d0d]">
              <div ref={termRef} className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-px">
                {termLines.map(line => (
                  <div key={line.id} className={`whitespace-pre-wrap break-all leading-5 ${line.color}`}>
                    {line.text}
                  </div>
                ))}
              </div>
              <div className="border-t border-[#1a1a1a] flex items-center px-3 py-1.5 gap-2 shrink-0 bg-[#0d0d0d]">
                <span className="text-green-400 text-xs font-mono">$</span>
                <input
                  ref={inputRef}
                  type="text"
                  value={termInput}
                  onChange={e => setTermInput(e.target.value)}
                  onKeyDown={handleTermKey}
                  placeholder="Type a command (run, clear, help)…"
                  className="flex-1 bg-transparent text-xs font-mono outline-none text-gray-200 placeholder:text-gray-600"
                  autoFocus
                />
              </div>
            </div>
          )}

          {/* PORTS */}
          {activeTab === "ports" && (
            <div className="h-full overflow-y-auto text-xs">
              <table className="w-full">
                <thead className="sticky top-0 bg-card z-10">
                  <tr className="text-muted-foreground border-b border-border">
                    <th className="text-left py-1.5 px-3 font-normal">Port</th>
                    <th className="text-left py-1.5 px-3 font-normal">Protocol</th>
                    <th className="text-left py-1.5 px-3 font-normal">Label</th>
                    <th className="text-left py-1.5 px-3 font-normal">Local Address</th>
                    <th className="text-left py-1.5 px-3 font-normal">Status</th>
                    <th className="text-left py-1.5 px-3 font-normal" />
                  </tr>
                </thead>
                <tbody>
                  {PORTS.map(p => (
                    <tr key={p.port} className="hover:bg-muted/40 border-b border-border/30 font-mono">
                      <td className="py-1.5 px-3 font-semibold text-foreground">{p.port}</td>
                      <td className="py-1.5 px-3 text-muted-foreground uppercase">{p.protocol}</td>
                      <td className="py-1.5 px-3 text-muted-foreground">{p.label}</td>
                      <td
                        className="py-1.5 px-3 text-blue-400 underline cursor-pointer hover:text-blue-300"
                        onClick={() => window.open(`${p.protocol}://${p.origin}:${p.port}`, "_blank")}
                      >
                        {p.protocol}://{p.origin}:{p.port}
                      </td>
                      <td className="py-1.5 px-3">
                        <span className={`flex items-center gap-1.5 ${p.running ? "text-green-400" : "text-muted-foreground"}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${p.running ? "bg-green-400" : "bg-gray-500"}`} />
                          {p.running ? "Running" : "Stopped"}
                        </span>
                      </td>
                      <td className="py-1.5 px-3">
                        <Button size="sm" variant="ghost" className="h-5 px-2 text-[10px]"
                          onClick={() => window.open(`${p.protocol}://${p.origin}:${p.port}`, "_blank")}>
                          Open
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

        </div>
      )}
    </div>
  )
}