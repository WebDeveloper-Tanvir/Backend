"use client"

import { useState } from "react"
import { ViralGrowthGate } from "@/components/viral-growth-gate"
import { Header } from "@/components/header"
import { Button } from "@/components/ui/button"

export default function DemoPage() {
  const [showGate, setShowGate] = useState(false)

  return (
    <main className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-4xl font-bold text-foreground mb-4">Viral Growth Gate Demo</h1>
          <p className="text-muted-foreground mb-8">
            Click the button below to see the viral growth gate in action. Users can share to unlock premium features.
          </p>
          <Button onClick={() => setShowGate(true)} className="bg-primary hover:bg-primary/90">
            Try Unlocking a Feature
          </Button>
        </div>
      </div>

      {showGate && (
        <ViralGrowthGate
          feature="Advanced Error Analysis"
          onUnlock={() => {
            setShowGate(false)
            alert("Feature unlocked!")
          }}
        />
      )}
    </main>
  )
}
