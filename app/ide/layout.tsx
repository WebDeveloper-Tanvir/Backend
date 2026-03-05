import type React from "react"

export const metadata = {
  title: "Errorless IDE - Code Editor with AI Assistant",
  description: "Full-featured VS Code-like IDE with Errorless AI chatbot for coding assistance",
}

export default function IDELayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}
