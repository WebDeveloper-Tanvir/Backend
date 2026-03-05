"use client"

import type React from "react"
import { useEffect, useState, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertCircle, Lock } from "lucide-react"

interface QuizContainerProps {
  children: React.ReactNode
  quizTitle: string
  timeLimit?: number
  onTimeUp?: () => void
}

export function QuizContainer({ children, quizTitle, timeLimit = 30, onTimeUp }: QuizContainerProps) {
  const [timeRemaining, setTimeRemaining] = useState(timeLimit * 60)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [cheatingAttempts, setCheatingAttempts] = useState(0)
  const [showWarning, setShowWarning] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Disable right-click
  useEffect(() => {
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault()
      setCheatingAttempts((prev) => prev + 1)
      setShowWarning(true)
      setTimeout(() => setShowWarning(false), 3000)
      return false
    }

    document.addEventListener("contextmenu", handleContextMenu)
    return () => document.removeEventListener("contextmenu", handleContextMenu)
  }, [])

  // Disable copy/paste
  useEffect(() => {
    const handleCopy = (e: ClipboardEvent) => {
      e.preventDefault()
      setCheatingAttempts((prev) => prev + 1)
      setShowWarning(true)
      setTimeout(() => setShowWarning(false), 3000)
    }

    const handlePaste = (e: ClipboardEvent) => {
      e.preventDefault()
      setCheatingAttempts((prev) => prev + 1)
      setShowWarning(true)
      setTimeout(() => setShowWarning(false), 3000)
    }

    document.addEventListener("copy", handleCopy)
    document.addEventListener("paste", handlePaste)

    return () => {
      document.removeEventListener("copy", handleCopy)
      document.removeEventListener("paste", handlePaste)
    }
  }, [])

  // Disable keyboard shortcuts for screenshot
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Disable Print Screen
      if (e.key === "PrintScreen") {
        e.preventDefault()
        setCheatingAttempts((prev) => prev + 1)
        setShowWarning(true)
        setTimeout(() => setShowWarning(false), 3000)
      }

      // Disable Ctrl+C, Ctrl+X, Ctrl+V
      if ((e.ctrlKey || e.metaKey) && ["c", "x", "v"].includes(e.key.toLowerCase())) {
        e.preventDefault()
        setCheatingAttempts((prev) => prev + 1)
        setShowWarning(true)
        setTimeout(() => setShowWarning(false), 3000)
      }

      // Disable F12 (Developer Tools)
      if (e.key === "F12") {
        e.preventDefault()
        setCheatingAttempts((prev) => prev + 1)
        setShowWarning(true)
        setTimeout(() => setShowWarning(false), 3000)
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [])

  // Timer countdown
  useEffect(() => {
    if (timeRemaining <= 0) {
      onTimeUp?.()
      return
    }

    const timer = setInterval(() => {
      setTimeRemaining((prev) => prev - 1)
    }, 1000)

    return () => clearInterval(timer)
  }, [timeRemaining, onTimeUp])

  // Fullscreen mode
  const handleFullscreen = async () => {
    if (!document.fullscreenElement) {
      try {
        await containerRef.current?.requestFullscreen()
        setIsFullscreen(true)
      } catch (err) {
        console.error("Error requesting fullscreen:", err)
      }
    } else {
      await document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  // Warn on page leave
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ""
    }

    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => window.removeEventListener("beforeunload", handleBeforeUnload)
  }, [])

  const minutes = Math.floor(timeRemaining / 60)
  const seconds = timeRemaining % 60
  const isTimeWarning = timeRemaining < 300 // 5 minutes

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Header with timer and security info */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold">{quizTitle}</h2>
          <p className="text-sm text-muted-foreground">Answer all questions carefully</p>
        </div>

        <div className="flex flex-col gap-2 md:flex-row md:items-center">
          <div
            className={`px-4 py-2 rounded-lg font-mono font-bold text-lg ${
              isTimeWarning ? "bg-red-500/20 text-red-500" : "bg-primary/20 text-primary"
            }`}
          >
            {String(minutes).padStart(2, "0")}:{String(seconds).padStart(2, "0")}
          </div>
          <Button size="sm" variant="outline" onClick={handleFullscreen} className="bg-transparent">
            {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          </Button>
        </div>
      </div>

      {/* Security warning */}
      {showWarning && (
        <Card className="border-red-500/50 bg-red-500/10">
          <CardContent className="pt-6 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
            <div>
              <p className="font-semibold text-red-500">Suspicious Activity Detected</p>
              <p className="text-sm text-red-500/80">
                Screenshots, copying, and external tools are not allowed. Attempts: {cheatingAttempts}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Security info banner */}
      <Card className="border-blue-500/50 bg-blue-500/10">
        <CardContent className="pt-6 flex items-center gap-3">
          <Lock className="h-5 w-5 text-blue-500 flex-shrink-0" />
          <div>
            <p className="text-sm text-blue-500">
              This quiz is protected. Screenshots, copying, and developer tools are disabled. Your activity is being
              monitored.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Quiz content */}
      <div className="bg-background rounded-lg border border-border/50">{children}</div>
    </div>
  )
}
