"use client"

import { useState } from "react"
import { AlertCircle, CheckCircle, Info, X } from "lucide-react"

interface Error {
  id: string
  type: "error" | "warning" | "info"
  line: number
  message: string
  suggestion: string
}

interface ErrorPanelProps {
  onClose?: () => void
}

export function ErrorPanel({ onClose }: ErrorPanelProps) {
  const [errors, setErrors] = useState<Error[]>([
    {
      id: "1",
      type: "error",
      line: 3,
      message: "Missing semicolon",
      suggestion: "Add a semicolon at the end of the line",
    },
    {
      id: "2",
      type: "warning",
      line: 1,
      message: "Unused variable",
      suggestion: "Remove or use the variable 'x'",
    },
  ])

  const getIcon = (type: string) => {
    switch (type) {
      case "error":
        return <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0" />
      case "warning":
        return <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
      case "info":
        return <Info className="w-5 h-5 text-blue-500 flex-shrink-0" />
      default:
        return null
    }
  }

  return (
    <div className="w-full md:w-80 flex flex-col border-t md:border-t-0 md:border-l border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between p-3 md:p-4 border-b border-border">
        <h2 className="text-base md:text-lg font-semibold text-foreground">Issues ({errors.length})</h2>
        <button onClick={onClose} className="md:hidden p-1 hover:bg-muted rounded">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Error List */}
      <div className="flex-1 overflow-y-auto">
        {errors.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground p-4">
            <CheckCircle className="w-8 h-8" />
            <p className="text-sm">No issues found</p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {errors.map((error) => (
              <div key={error.id} className="p-3 md:p-4 hover:bg-muted/50 cursor-pointer transition-colors">
                <div className="flex items-start gap-3">
                  {getIcon(error.type)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-mono text-muted-foreground">Line {error.line}</span>
                      <span
                        className={`text-xs font-semibold uppercase ${
                          error.type === "error"
                            ? "text-destructive"
                            : error.type === "warning"
                              ? "text-yellow-500"
                              : "text-blue-500"
                        }`}
                      >
                        {error.type}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-foreground mt-1">{error.message}</p>
                    <p className="text-xs text-muted-foreground mt-2">{error.suggestion}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 md:p-4 border-t border-border bg-muted/30">
        <button className="w-full px-3 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors">
          Fix All Issues
        </button>
      </div>
    </div>
  )
}
