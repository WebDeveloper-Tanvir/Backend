"use client"

import type React from "react"
import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Copy, Download, Play, RotateCcw } from "lucide-react"

const LANGUAGE_TEMPLATES: Record<string, { code: string; extension: string }> = {
  python: {
    code: `# Python Example
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()`,
    extension: "py",
  },
  javascript: {
    code: `// JavaScript Example
function helloWorld() {
    console.log("Hello, World!");
}

helloWorld();`,
    extension: "js",
  },
  cpp: {
    code: `#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}`,
    extension: "cpp",
  },
  c: {
    code: `#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}`,
    extension: "c",
  },
  java: {
    code: `public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}`,
    extension: "java",
  },
  swift: {
    code: `import Foundation

func helloWorld() {
    print("Hello, World!")
}

helloWorld()`,
    extension: "swift",
  },
}

interface CodeEditorProps {
  onCodeChange?: (code: string) => void
  onLanguageChange?: (language: string) => void
  initialLanguage?: string
  initialCode?: string
  readOnly?: boolean
}

export function CodeEditorAdvanced({
  onCodeChange,
  onLanguageChange,
  initialLanguage = "python",
  initialCode,
  readOnly = false,
}: CodeEditorProps) {
  const [language, setLanguage] = useState(initialLanguage)
  const [code, setCode] = useState(initialCode || LANGUAGE_TEMPLATES[initialLanguage].code)
  const [output, setOutput] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage)
    setCode(LANGUAGE_TEMPLATES[newLanguage].code)
    setOutput("")
    setError("")
    onLanguageChange?.(newLanguage)
  }

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newCode = e.target.value
    setCode(newCode)
    onCodeChange?.(newCode)
  }

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code)
  }

  const handleDownloadCode = () => {
    const element = document.createElement("a")
    const file = new Blob([code], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = `code.${LANGUAGE_TEMPLATES[language].extension}`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const handleResetCode = () => {
    setCode(LANGUAGE_TEMPLATES[language].code)
    setOutput("")
    setError("")
  }

  const handleRunCode = async () => {
    setIsRunning(true)
    setOutput("")
    setError("")

    try {
      // Simulate code execution (in production, this would call a backend API)
      // For now, we'll just show a mock output
      await new Promise((resolve) => setTimeout(resolve, 1000))

      if (language === "python" && code.includes("print")) {
        setOutput("Hello, World!\n[Code executed successfully]")
      } else if (language === "javascript" && code.includes("console.log")) {
        setOutput("Hello, World!\n[Code executed successfully]")
      } else {
        setOutput("[Code executed successfully]\n[Output would appear here]")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred while running the code")
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="w-full space-y-4">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-col gap-2 md:flex-row md:items-center">
          <label className="text-sm font-medium">Language:</label>
          <Select value={language} onValueChange={handleLanguageChange} disabled={readOnly}>
            <SelectTrigger className="w-full md:w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="python">Python</SelectItem>
              <SelectItem value="javascript">JavaScript</SelectItem>
              <SelectItem value="cpp">C++</SelectItem>
              <SelectItem value="c">C</SelectItem>
              <SelectItem value="java">Java</SelectItem>
              <SelectItem value="swift">Swift</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleCopyCode}
            className="flex items-center gap-2 bg-transparent"
            disabled={readOnly}
          >
            <Copy className="h-4 w-4" />
            Copy
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleDownloadCode}
            className="flex items-center gap-2 bg-transparent"
            disabled={readOnly}
          >
            <Download className="h-4 w-4" />
            Download
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleResetCode}
            className="flex items-center gap-2 bg-transparent"
            disabled={readOnly}
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
          <Button
            size="sm"
            onClick={handleRunCode}
            disabled={isRunning || readOnly}
            className="flex items-center gap-2 bg-primary"
          >
            <Play className="h-4 w-4" />
            {isRunning ? "Running..." : "Run"}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="overflow-hidden border-border/50">
          <div className="bg-muted/50 px-4 py-2 border-b">
            <p className="text-sm font-medium">Code Editor</p>
          </div>
          <textarea
            ref={textareaRef}
            value={code}
            onChange={handleCodeChange}
            readOnly={readOnly}
            className="w-full h-96 p-4 font-mono text-sm bg-background text-foreground resize-none focus:outline-none"
            spellCheck="false"
          />
        </Card>

        <Card className="overflow-hidden border-border/50">
          <div className="bg-muted/50 px-4 py-2 border-b">
            <p className="text-sm font-medium">Output</p>
          </div>
          <div className="w-full h-96 p-4 font-mono text-sm bg-background text-foreground overflow-auto">
            {error ? (
              <div className="text-red-500">
                <p className="font-semibold">Error:</p>
                <p>{error}</p>
              </div>
            ) : output ? (
              <pre className="whitespace-pre-wrap break-words text-green-500">{output}</pre>
            ) : (
              <p className="text-muted-foreground">Output will appear here...</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
