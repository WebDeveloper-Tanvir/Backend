"use client"

import type React from "react"

import { useState, useEffect, useCallback, useRef } from "react"
import dynamic from "next/dynamic"
import { X } from "lucide-react"

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-muted-foreground bg-[#1e1e1e]">
      Loading editor...
    </div>
  ),
})

interface CodeEditorIDEProps {
  file: {
    id: string
    name: string
    content?: string
    language?: string
  } | null
  onContentChange: (content: string) => void
  onClose: () => void
}

const languageExtensions: Record<string, string> = {
  py: "python",
  js: "javascript",
  ts: "typescript",
  jsx: "javascript",
  tsx: "typescript",
  java: "java",
  cpp: "cpp",
  c: "c",
  swift: "swift",
  go: "go",
  rs: "rust",
  php: "php",
  rb: "ruby",
  sql: "sql",
  html: "html",
  css: "css",
  json: "json",
  xml: "xml",
  yaml: "yaml",
  md: "markdown",
}

function getLanguageFromFilename(filename: string, fallback = "python"): string {
  const ext = filename.split(".").pop()?.toLowerCase()
  return (ext && languageExtensions[ext]) || fallback
}

export function CodeEditorIDE({ file, onContentChange, onClose }: CodeEditorIDEProps) {
  const [fontSize, setFontSize] = useState(14)
  const [theme, setTheme] = useState("vs-dark")

  // Detect system theme — always use vs-dark variant for brand consistency
  useEffect(() => {
    setTheme("vs-dark")
  }, [])

  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      if (value !== undefined) {
        onContentChange(value)
      }
    },
    [onContentChange]
  )

  const handleFontSizeChange = (delta: number) => {
    setFontSize((prev) => Math.max(10, Math.min(32, prev + delta)))
  }

  if (!file) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#1e1e1e] text-muted-foreground">
        <p>Select a file to start editing</p>
      </div>
    )
  }

  const language = getLanguageFromFilename(file.name, file.language || "python")

  return (
    <div className="flex-1 flex flex-col bg-background h-full">
      <div className="flex items-center justify-between px-4 py-1.5 shrink-0"
        style={{ borderBottom: "1px solid rgba(0,201,167,0.1)", background: "rgba(2,8,5,0.8)" }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium" style={{ color: "rgba(200,230,220,0.6)" }}>{file.name}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold tracking-widest uppercase"
            style={{ background: "rgba(0,201,167,0.08)", color: "#00c9a7", border: "1px solid rgba(0,201,167,0.15)" }}>
            {language}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <div className="flex items-center gap-0.5 text-xs">
            <button onClick={() => handleFontSizeChange(-2)}
              className="w-6 h-6 flex items-center justify-center rounded transition-all"
              style={{ color: "rgba(200,230,220,0.4)" }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.1)"; (e.currentTarget as HTMLButtonElement).style.color = "#00c9a7" }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.4)" }}
            >−</button>
            <span className="w-8 text-center font-mono text-[10px]" style={{ color: "rgba(200,230,220,0.35)" }}>{fontSize}px</span>
            <button onClick={() => handleFontSizeChange(2)}
              className="w-6 h-6 flex items-center justify-center rounded transition-all"
              style={{ color: "rgba(200,230,220,0.4)" }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(0,201,167,0.1)"; (e.currentTarget as HTMLButtonElement).style.color = "#00c9a7" }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.4)" }}
            >+</button>
          </div>
          <button onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded transition-all"
            style={{ color: "rgba(200,230,220,0.35)" }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,77,109,0.12)"; (e.currentTarget as HTMLButtonElement).style.color = "#ff4d6d" }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; (e.currentTarget as HTMLButtonElement).style.color = "rgba(200,230,220,0.35)" }}
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {/* key={file.id} forces Monaco to fully remount when switching files,
          ensuring the correct content is always displayed */}
      <div className="flex-1 overflow-hidden min-h-0">
        <MonacoEditor
          key={file.id}
          height="100%"
          language={language}
          value={file.content ?? ""}
          onChange={handleEditorChange}
          theme={theme}
          options={{
            minimap: { enabled: true },
            fontSize: fontSize,
            fontFamily: "'Fira Code', 'Cascadia Code', monospace",
            fontLigatures: true,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            wordWrap: "on",
            folding: true,
            foldingStrategy: "indentation",
            lineDecorationsWidth: 20,
            lineNumbersMinChars: 3,
            renderLineHighlight: "line",
            roundedSelection: false,
            smoothScrolling: true,
            bracketPairColorization: { enabled: true },
            padding: { top: 12 },
          }}
        />
      </div>
    </div>
  )
}

