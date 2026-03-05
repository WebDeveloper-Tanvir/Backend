"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import {
  Send, Trash2, Copy, ChevronDown, Bot, Sparkles, Zap, Bug, BookOpen,
  Paperclip, X, FileCode, FileText, AlertCircle, CheckCircle2, AlertTriangle,
  Upload, BarChart3,
} from "lucide-react"
import { ErrorlessChatbot, type ChatMessage } from "@/lib/errorless-chatbot"
import { analyzeCode, formatAnalysisAsMarkdown, type AnalysisResult } from "@/lib/code-analyzer"
import ReactMarkdown from "react-markdown"

interface ErrorlessChatbotPanelProps {
  selectedCode?: string
  selectedLanguage?: string
}

interface AttachedFile {
  id:       string
  name:     string
  content:  string
  size:     number
  language: string
  result?:  AnalysisResult
}

let _id = 200
const nid = () => String(++_id)

// ── Helpers ─────────────────────────────────────────────────────────────────
function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function fileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase()
  if (["js","jsx","ts","tsx","py","java","cpp","c","go","rs","rb"].includes(ext ?? ""))
    return <FileCode size={12} className="text-[#00c9a7]" />
  return <FileText size={12} className="text-[rgba(200,230,220,0.5)]" />
}

function scoreColor(s: number) {
  if (s >= 80) return "#00c9a7"
  if (s >= 60) return "#ffd166"
  return "#ff4d6d"
}

const SUPPORTED_EXTS = [".py",".js",".ts",".jsx",".tsx",".java",".cpp",".c",".go",".rs",".rb",".php",".swift",".html",".css",".sql",".txt",".md"]

// ── Component ─────────────────────────────────────────────────────────────
export function ErrorlessChatbotPanel({ selectedCode = "", selectedLanguage = "python" }: ErrorlessChatbotPanelProps) {
  const [messages, setMessages]     = useState<ChatMessage[]>([{
    id: "1", role: "assistant", timestamp: new Date(),
    content: "Hello! I'm Errorless, your AI coding assistant.\n\nI can help you with:\n\n- **Code Generation** - Write code in any language\n- **Debugging** - Fix errors and bugs\n- **Explanations** - Learn programming concepts\n- **Optimization** - Improve code performance\n- **Best Practices** - Follow industry standards\n\n📎 **Upload a code file** using the paperclip button to get a full static analysis with error detection!\n\nWhat would you like help with?",
  }])
  const [input, setInput]           = useState("")
  const [isLoading, setIsLoading]   = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showNLP, setShowNLP]       = useState(false)
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([])
  const [dragOver, setDragOver]     = useState(false)
  const [chatbot]                   = useState(() => new ErrorlessChatbot())

  const endRef     = useRef<HTMLDivElement>(null)
  const fileRef    = useRef<HTMLInputElement>(null)
  const inputRef   = useRef<HTMLInputElement>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  // ── Read & analyze file ────────────────────────────────────────────────
  const processFile = useCallback(async (file: File) => {
    if (file.size > 500 * 1024) {
      addMessage("assistant", `⚠️ File **${file.name}** is too large (max 500 KB). Please upload a smaller file.`)
      return
    }

    const ext = "." + (file.name.split(".").pop()?.toLowerCase() ?? "")
    if (!SUPPORTED_EXTS.includes(ext)) {
      addMessage("assistant", `⚠️ Unsupported file type **${ext}**.\n\nSupported: ${SUPPORTED_EXTS.join(", ")}`)
      return
    }

    setIsAnalyzing(true)
    addMessage("user", `📎 Uploaded \`${file.name}\` (${fmtSize(file.size)}) for analysis`)

    const content = await file.text()
    const result  = analyzeCode(content, file.name)

    const attached: AttachedFile = {
      id: nid(), name: file.name, content,
      size: file.size, language: result.language, result,
    }
    setAttachedFiles(p => [...p, attached])

    // Format and add analysis as assistant message
    await new Promise(r => setTimeout(r, 600))  // simulate thinking
    const analysisMarkdown = formatAnalysisAsMarkdown(result)
    addMessage("assistant", analysisMarkdown, attached)
    setIsAnalyzing(false)
  }, [])

  const addMessage = (role: "user" | "assistant", content: string, file?: AttachedFile) => {
    const msg: ChatMessage & { attachedFile?: AttachedFile } = {
      id: nid(), role, content, timestamp: new Date(),
      ...(file ? { attachedFile: file } : {})
    }
    setMessages(p => [...p, msg as ChatMessage])
  }

  // ── File input handler ─────────────────────────────────────────────────
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    files.forEach(processFile)
    e.target.value = ""
  }

  // ── Drag & drop ────────────────────────────────────────────────────────
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    files.forEach(processFile)
  }, [processFile])

  // ── Chat send ──────────────────────────────────────────────────────────
  const send = async () => {
    if (!input.trim()) return
    const text = input.trim()
    addMessage("user", text)
    setInput("")
    setIsLoading(true)
    try {
      const res = await chatbot.processMessage(text)
      setMessages(p => [...p, res])
    } catch (err) {
      addMessage("assistant", `Error: ${err instanceof Error ? err.message : "Unknown error"}`)
    } finally { setIsLoading(false) }
  }

  // ── Quick actions ──────────────────────────────────────────────────────
  const quickAction = (action: string) => {
    const prompts: Record<string, string> = {
      debug:    `Debug this ${selectedLanguage} code:\n\`\`\`${selectedLanguage}\n${selectedCode}\n\`\`\``,
      explain:  `Explain this ${selectedLanguage} code:\n\`\`\`${selectedLanguage}\n${selectedCode}\n\`\`\``,
      optimize: `Optimize this ${selectedLanguage} code:\n\`\`\`${selectedLanguage}\n${selectedCode}\n\`\`\``,
    }
    if (prompts[action]) { setInput(prompts[action]); inputRef.current?.focus() }
  }

  const clearChat = () => {
    setMessages([{ id: nid(), role: "assistant", timestamp: new Date(),
      content: "Chat cleared. Upload a code file 📎 or ask me anything!" }])
    setAttachedFiles([])
    chatbot.clearHistory()
  }

  const lastMsg = messages[messages.length - 1] as any

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: "#030e09", borderLeft: "1px solid rgba(0,201,167,0.1)" }}
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Drag-over overlay */}
      {dragOver && (
        <div className="absolute inset-0 z-50 flex flex-col items-center justify-center pointer-events-none"
          style={{ background: "rgba(0,201,167,0.06)", border: "2px dashed rgba(0,201,167,0.5)", borderRadius: 12 }}>
          <Upload size={32} className="text-[#00c9a7] mb-2" />
          <p className="text-sm font-bold text-[#00c9a7]">Drop to analyze</p>
        </div>
      )}

      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="px-4 py-3 shrink-0"
        style={{ borderBottom: "1px solid rgba(0,201,167,0.1)", background: "rgba(2,8,5,0.9)" }}>
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, rgba(0,201,167,0.2), rgba(67,97,238,0.2))", border: "1px solid rgba(0,201,167,0.25)" }}>
              <Bot size={14} className="text-[#00c9a7]" />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-[#00c9a7]"
              style={{ boxShadow: "0 0 6px rgba(0,201,167,0.8)" }} />
          </div>
          <div className="flex-1">
            <p className="text-xs font-bold text-white tracking-wide">Errorless AI Assistant</p>
            <p className="text-[10px] text-[rgba(200,230,220,0.4)]">
              {isAnalyzing ? "Analyzing code…" : "Your coding companion"}
            </p>
          </div>
          {/* Upload button in header */}
          <button
            onClick={() => fileRef.current?.click()}
            title="Upload code file for analysis"
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-semibold transition-all"
            style={{ background: "rgba(0,201,167,0.08)", border: "1px solid rgba(0,201,167,0.2)", color: "#00c9a7" }}
            onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.15)"}
            onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.08)"}
          >
            <Paperclip size={11} /> Analyze
          </button>
        </div>

        {/* Quick actions */}
        {selectedCode && (
          <div className="flex gap-1.5 mt-3 flex-wrap">
            {[
              { id: "debug",    label: "Debug",    icon: <Bug size={10} /> },
              { id: "explain",  label: "Explain",  icon: <BookOpen size={10} /> },
              { id: "optimize", label: "Optimize", icon: <Zap size={10} /> },
            ].map(({ id, label, icon }) => (
              <button key={id} onClick={() => quickAction(id)}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-semibold tracking-wide transition-all"
                style={{ border: "1px solid rgba(0,201,167,0.2)", background: "rgba(0,201,167,0.06)", color: "rgba(200,230,220,0.7)" }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.12)"; (e.currentTarget as HTMLButtonElement).style.color = "#00c9a7" }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.06)"; (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.7)" }}
              >
                {icon}{label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Messages ──────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3 min-h-0">
        {messages.map(msg => {
          const msgAny = msg as any
          return (
            <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "assistant" && (
                <div className="w-6 h-6 rounded-md flex items-center justify-center mr-2 mt-0.5 shrink-0"
                  style={{ background: "linear-gradient(135deg, rgba(0,201,167,0.2), rgba(67,97,238,0.2))", border: "1px solid rgba(0,201,167,0.2)" }}>
                  <Sparkles size={10} className="text-[#00c9a7]" />
                </div>
              )}
              <div className="max-w-[88%] group relative">
                {/* Attached file badge */}
                {msgAny.attachedFile && (
                  <FileAnalysisBadge file={msgAny.attachedFile as AttachedFile} />
                )}

                <div
                  className="px-3 py-2.5 rounded-xl text-xs leading-relaxed"
                  style={msg.role === "user" ? {
                    background: "linear-gradient(135deg, rgba(0,201,167,0.18), rgba(67,97,238,0.12))",
                    border: "1px solid rgba(0,201,167,0.22)",
                    color: "rgba(200,230,220,0.9)",
                    borderBottomRightRadius: 4,
                  } : {
                    background: "rgba(4,18,12,0.9)",
                    border: "1px solid rgba(0,201,167,0.08)",
                    color: "rgba(200,230,220,0.8)",
                    borderBottomLeftRadius: 4,
                  }}
                >
                  <ReactMarkdown
                    components={{
                      h2: ({ children }) => <h2 className="text-sm font-bold mb-2 mt-1" style={{ color: "#00c9a7" }}>{children}</h2>,
                      h3: ({ children }) => <h3 className="text-xs font-bold mb-1.5 mt-2" style={{ color: "rgba(200,230,220,0.9)" }}>{children}</h3>,
                      strong: ({ children }) => <strong className="font-bold" style={{ color: "#00c9a7" }}>{children}</strong>,
                      code: ({ node, inline, children, ...props }: any) =>
                        inline ? (
                          <code className="px-1 py-0.5 rounded text-[11px] font-mono"
                            style={{ background: "rgba(0,201,167,0.1)", color: "#00c9a7" }} {...props}>
                            {children}
                          </code>
                        ) : (
                          <div className="mt-2 mb-1">
                            <pre className="p-2.5 rounded-lg overflow-x-auto text-[11px] font-mono"
                              style={{ background: "rgba(0,0,0,0.5)", border: "1px solid rgba(0,201,167,0.1)", color: "#00c9a7" }}>
                              <code {...props}>{children}</code>
                            </pre>
                          </div>
                        ),
                      blockquote: ({ children }) => (
                        <div className="pl-3 mt-1 mb-1 text-[11px]"
                          style={{ borderLeft: "2px solid rgba(0,201,167,0.3)", color: "rgba(200,230,220,0.55)" }}>
                          {children}
                        </div>
                      ),
                      li: ({ children }) => <li className="ml-3 list-disc mb-0.5">{children}</li>,
                      ul: ({ children }) => <ul className="mb-2 space-y-0.5">{children}</ul>,
                      p: ({ children }) => <div className="mb-1.5 last:mb-0">{children}</div>,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>

                {msg.role === "assistant" && (
                  <button
                    onClick={() => navigator.clipboard.writeText(msg.content)}
                    className="absolute -bottom-1 right-1 w-5 h-5 rounded flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: "rgba(0,201,167,0.1)", border: "1px solid rgba(0,201,167,0.2)" }}
                  >
                    <Copy size={9} className="text-[#00c9a7]" />
                  </button>
                )}
              </div>
            </div>
          )
        })}

        {/* Thinking / analyzing indicator */}
        {(isLoading || isAnalyzing) && (
          <div className="flex justify-start">
            <div className="w-6 h-6 rounded-md flex items-center justify-center mr-2 shrink-0"
              style={{ background: "linear-gradient(135deg, rgba(0,201,167,0.2), rgba(67,97,238,0.2))", border: "1px solid rgba(0,201,167,0.2)" }}>
              <Sparkles size={10} className="text-[#00c9a7]" />
            </div>
            <div className="px-3 py-2.5 rounded-xl rounded-bl-sm"
              style={{ background: "rgba(4,18,12,0.9)", border: "1px solid rgba(0,201,167,0.08)" }}>
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  {[0, 0.2, 0.4].map(delay => (
                    <div key={delay} className="w-1.5 h-1.5 rounded-full animate-bounce"
                      style={{ background: "#00c9a7", animationDelay: `${delay}s`, boxShadow: "0 0 6px rgba(0,201,167,0.5)" }} />
                  ))}
                </div>
                <span className="text-[10px] font-mono" style={{ color: "rgba(200,230,220,0.4)" }}>
                  {isAnalyzing ? "Analyzing code…" : "Thinking…"}
                </span>
              </div>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* ── Upload drop zone hint (only when no files) ─────────────── */}
      {attachedFiles.length === 0 && (
        <div
          className="mx-3 mb-2 rounded-xl border-dashed flex items-center justify-center gap-2 cursor-pointer transition-all shrink-0"
          style={{ border: "1px dashed rgba(0,201,167,0.2)", background: "rgba(0,201,167,0.03)", padding: "10px 12px" }}
          onClick={() => fileRef.current?.click()}
          onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(0,201,167,0.4)"; (e.currentTarget as HTMLDivElement).style.background = "rgba(0,201,167,0.06)" }}
          onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(0,201,167,0.2)"; (e.currentTarget as HTMLDivElement).style.background = "rgba(0,201,167,0.03)" }}
        >
          <Upload size={14} className="text-[#00c9a7] shrink-0" />
          <div>
            <p className="text-[10px] font-semibold text-[rgba(200,230,220,0.6)]">Drop or click to upload code</p>
            <p className="text-[9px] text-[rgba(200,230,220,0.3)]">.py .js .ts .java .cpp .go .rs + more</p>
          </div>
        </div>
      )}

      {/* ── NLP Analysis ──────────────────────────────────────────── */}
      <div className="shrink-0 px-3 py-1"
        style={{ borderTop: "1px solid rgba(0,201,167,0.08)" }}>
        <button
          onClick={() => setShowNLP(s => !s)}
          className="w-full flex items-center justify-between px-2 py-1 rounded-lg text-[10px] font-semibold tracking-widest uppercase transition-colors"
          style={{ color: "rgba(200,230,220,0.35)" }}
          onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.6)"}
          onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.35)"}
        >
          <span className="flex items-center gap-1.5"><BarChart3 size={10} /> NLP Analysis</span>
          <ChevronDown size={11} className={`transition-transform duration-200 ${showNLP ? "rotate-180" : ""}`} />
        </button>
        {showNLP && lastMsg?.nlpAnalysis && (
          <div className="mt-1 mx-1 p-2 rounded-lg space-y-1 text-[10px] font-mono"
            style={{ background: "rgba(0,201,167,0.05)", border: "1px solid rgba(0,201,167,0.1)" }}>
            {[["Intent", lastMsg.nlpAnalysis.intent], ["Language", lastMsg.nlpAnalysis.detectedLanguage], ["Sentiment", lastMsg.nlpAnalysis.sentiment]].map(([k, v]) => (
              <div key={k} className="flex gap-1">
                <span style={{ color: "rgba(200,230,220,0.4)" }}>{k}:</span>
                <span style={{ color: "#00c9a7" }}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Input area ────────────────────────────────────────────── */}
      <div className="px-3 pb-3 pt-2 shrink-0" style={{ borderTop: "1px solid rgba(0,201,167,0.1)" }}>
        <div className="flex gap-2 items-center">
          {/* Attach button */}
          <button
            onClick={() => fileRef.current?.click()}
            title="Upload code file"
            className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-all"
            style={{ background: "rgba(0,201,167,0.06)", border: "1px solid rgba(0,201,167,0.18)" }}
            onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.14)"}
            onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.06)"}
          >
            <Paperclip size={13} className="text-[#00c9a7]" />
          </button>

          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Ask Errorless for help…"
            disabled={isLoading || isAnalyzing}
            className="flex-1 px-3 py-2 rounded-xl text-xs outline-none transition-all font-medium placeholder:font-normal"
            style={{ background: "rgba(0,201,167,0.06)", border: "1px solid rgba(0,201,167,0.18)", color: "rgba(200,230,220,0.9)" }}
            onFocus={e => (e.target as HTMLInputElement).style.borderColor = "rgba(0,201,167,0.4)"}
            onBlur={e => (e.target as HTMLInputElement).style.borderColor = "rgba(0,201,167,0.18)"}
          />

          {/* Send */}
          <button
            onClick={send}
            disabled={isLoading || isAnalyzing || !input.trim()}
            className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-all"
            style={{
              background: input.trim() && !isLoading && !isAnalyzing
                ? "linear-gradient(135deg, #00c9a7, #4361ee)"
                : "rgba(0,201,167,0.08)",
              border: "1px solid rgba(0,201,167,0.2)",
            }}
          >
            <Send size={13} className={input.trim() && !isLoading ? "text-[#030e09]" : "text-[rgba(200,230,220,0.3)]"} />
          </button>
        </div>

        <div className="flex justify-between items-center mt-1.5">
          <p className="text-[9px]" style={{ color: "rgba(200,230,220,0.2)" }}>
            📎 Drag & drop code files to analyze
          </p>
          <button onClick={clearChat}
            className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded transition-colors"
            style={{ color: "rgba(200,230,220,0.3)" }}
            onMouseEnter={e => (e.currentTarget as HTMLButtonElement).style.color = "#ff4d6d"}
            onMouseLeave={e => (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.3)"}
          >
            <Trash2 size={9} /> Clear
          </button>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileRef}
        type="file"
        multiple
        accept={SUPPORTED_EXTS.join(",")}
        onChange={handleFileInput}
        className="hidden"
      />
    </div>
  )
}

// ── File analysis summary badge shown above message ────────────────────────
function FileAnalysisBadge({ file }: { file: AttachedFile }) {
  const r = file.result
  if (!r) return null
  const errors   = r.issues.filter(i => i.severity === "error").length
  const warnings = r.issues.filter(i => i.severity === "warning").length
  const infos    = r.issues.filter(i => i.severity === "info").length
  const color    = scoreColor(r.score)

  return (
    <div className="mb-1.5 p-2 rounded-lg"
      style={{ background: "rgba(0,201,167,0.05)", border: "1px solid rgba(0,201,167,0.12)" }}>
      <div className="flex items-center gap-2 mb-1.5">
        {fileIcon(file.name)}
        <span className="text-[11px] font-mono font-semibold text-[rgba(200,230,220,0.8)] flex-1 truncate">{file.name}</span>
        <span className="text-[10px] font-bold" style={{ color }}>{r.score}/100</span>
      </div>

      {/* Score bar */}
      <div className="w-full rounded-full overflow-hidden mb-1.5" style={{ height: 3, background: "rgba(255,255,255,0.06)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${r.score}%`, background: color }} />
      </div>

      {/* Issue counts */}
      <div className="flex gap-2 text-[10px] font-mono">
        {errors > 0 && (
          <span className="flex items-center gap-0.5" style={{ color: "#ff4d6d" }}>
            <AlertCircle size={9} /> {errors} error{errors !== 1 ? "s" : ""}
          </span>
        )}
        {warnings > 0 && (
          <span className="flex items-center gap-0.5" style={{ color: "#ffd166" }}>
            <AlertTriangle size={9} /> {warnings} warn{warnings !== 1 ? "s" : ""}
          </span>
        )}
        {infos > 0 && (
          <span className="flex items-center gap-0.5" style={{ color: "#00b4d8" }}>
            <CheckCircle2 size={9} /> {infos} hint{infos !== 1 ? "s" : ""}
          </span>
        )}
        {errors === 0 && warnings === 0 && (
          <span className="flex items-center gap-0.5" style={{ color: "#00c9a7" }}>
            <CheckCircle2 size={9} /> Clean!
          </span>
        )}
        <span className="ml-auto" style={{ color: "rgba(200,230,220,0.3)" }}>
          {r.language} · {r.lines} lines · {fmtSize(file.size)}
        </span>
      </div>
    </div>
  )
}
