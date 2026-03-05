"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Play, Trash2, Copy } from "lucide-react"
import { CodeExecutor, type ExecutionResult } from "@/lib/code-executor"

interface TerminalProps {
  onRunCode?: (code: string, language: string) => Promise<string>
  currentLanguage: string
  currentCode: string
}

export function Terminal({ currentLanguage, currentCode }: TerminalProps) {
  const [output, setOutput] = useState<string[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const terminalRef = useRef<HTMLDivElement>(null)
  const [executor] = useState(() => new CodeExecutor())

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [output])

  const handleRunCode = async () => {
    if (!currentCode.trim()) {
      setOutput(["Error: No code to run"])
      return
    }

    setIsRunning(true)
    setOutput([`Running ${currentLanguage} code...`])

    try {
      const result: ExecutionResult = await executor.executeCode(currentCode, currentLanguage)

      if (result.success) {
        setOutput((prev) => [...prev, result.output, `\n✓ Execution completed in ${result.executionTime}ms`])
      } else {
        setOutput((prev) => [...prev, `Error: ${result.error}`, `\n✗ Execution failed after ${result.executionTime}ms`])
      }
    } catch (error) {
      setOutput((prev) => [...prev, `Error: ${error instanceof Error ? error.message : "Unknown error"}`])
    } finally {
      setIsRunning(false)
    }
  }

  const handleClear = () => {
    setOutput([])
  }

  const handleCopyOutput = () => {
    const text = output.join("\n")
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="flex flex-col h-full bg-background border-t border-border">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card">
        <span className="text-sm font-medium">Terminal</span>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleRunCode}
            disabled={isRunning}
            className="gap-2 bg-transparent"
          >
            <Play size={14} />
            Run Code
          </Button>
          <Button size="sm" variant="ghost" onClick={handleCopyOutput}>
            <Copy size={14} />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleClear}>
            <Trash2 size={14} />
          </Button>
        </div>
      </div>
      <div ref={terminalRef} className="flex-1 overflow-y-auto p-4 font-mono text-sm bg-background text-foreground">
        {output.length === 0 ? (
          <p className="text-muted-foreground">Terminal output will appear here...</p>
        ) : (
          output.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-words">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
