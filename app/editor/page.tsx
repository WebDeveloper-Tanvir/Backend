"use client"

import { useState } from "react"
import { CodeEditor } from "@/components/code-editor"
import { ErrorPanel } from "@/components/error-panel"
import { Header } from "@/components/header"

export default function EditorPage() {
  const [showErrorPanel, setShowErrorPanel] = useState(true)

  return (
    <main className="min-h-screen bg-background">
      <Header />
      <div className="flex flex-col md:flex-row h-[calc(100vh-80px)]">
        <CodeEditor />
        {showErrorPanel && <ErrorPanel onClose={() => setShowErrorPanel(false)} />}
        {!showErrorPanel && (
          <button onClick={() => setShowErrorPanel(true)} className="md:hidden p-4 text-primary hover:bg-muted">
            Show Issues
          </button>
        )}
      </div>
    </main>
  )
}
