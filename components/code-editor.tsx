"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Play, Copy, Download, Menu, X } from "lucide-react"

export function CodeEditor() {
  const [code, setCode] = useState(`function helloWorld() {
  console.log("Hello, World!")
  return 42
}

helloWorld()`)

  const [language, setLanguage] = useState("javascript")
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
  }

  const handleDownload = () => {
    const element = document.createElement("a")
    const file = new Blob([code], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = `code.${language === "javascript" ? "js" : "py"}`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <div className="flex-1 flex flex-col border-r border-border">
      {/* Header */}
      <div className="flex items-center justify-between p-3 md:p-4 border-b border-border bg-card gap-2 flex-wrap md:flex-nowrap">
        <div className="flex items-center gap-2">
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="px-2 md:px-3 py-2 bg-input border border-border rounded-md text-xs md:text-sm text-foreground"
          >
            <option value="javascript">JavaScript</option>
            <option value="python">Python</option>
            <option value="typescript">TypeScript</option>
            <option value="jsx">JSX</option>
          </select>
        </div>

        {/* Desktop Actions */}
        <div className="hidden md:flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy} className="gap-2 bg-transparent">
            <Copy className="w-4 h-4" />
            Copy
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload} className="gap-2 bg-transparent">
            <Download className="w-4 h-4" />
            Download
          </Button>
          <Button size="sm" className="gap-2 bg-primary hover:bg-primary/90">
            <Play className="w-4 h-4" />
            Run
          </Button>
        </div>

        {/* Mobile Menu Button */}
        <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="md:hidden p-2 hover:bg-muted rounded-md">
          {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
        </button>
      </div>

      {/* Mobile Actions */}
      {mobileMenuOpen && (
        <div className="md:hidden flex gap-2 p-3 border-b border-border bg-muted/50 flex-wrap">
          <Button variant="outline" size="sm" onClick={handleCopy} className="gap-2 bg-transparent flex-1 min-w-fit">
            <Copy className="w-4 h-4" />
            Copy
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleDownload}
            className="gap-2 bg-transparent flex-1 min-w-fit"
          >
            <Download className="w-4 h-4" />
            Download
          </Button>
          <Button size="sm" className="gap-2 bg-primary hover:bg-primary/90 flex-1 min-w-fit">
            <Play className="w-4 h-4" />
            Run
          </Button>
        </div>
      )}

      {/* Editor */}
      <div className="flex-1 overflow-hidden">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="w-full h-full p-3 md:p-4 bg-background text-foreground font-mono text-xs md:text-sm resize-none focus:outline-none border-none"
          placeholder="Enter your code here..."
          spellCheck="false"
        />
      </div>
    </div>
  )
}
