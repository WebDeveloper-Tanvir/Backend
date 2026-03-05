"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Share2, Users, Gift, Copy, Check, X } from "lucide-react"

interface GateProps {
  feature: string
  onUnlock?: () => void
  onClose?: () => void
}

export function ViralGrowthGate({ feature, onUnlock, onClose }: GateProps) {
  const [copied, setCopied] = useState(false)
  const [invitesSent, setInvitesSent] = useState(0)
  const referralLink = `https://errorless.dev?ref=${Math.random().toString(36).substr(2, 9)}`

  const handleCopyLink = () => {
    navigator.clipboard.writeText(referralLink)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleShareTwitter = () => {
    const text = `I'm using Errorless - an AI-powered code error solver that's helping me debug faster! Check it out: ${referralLink}`
    window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`, "_blank")
  }

  const handleShareLinkedIn = () => {
    window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(referralLink)}`, "_blank")
  }

  const invitesNeeded = 3
  const progress = (invitesSent / invitesNeeded) * 100

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-lg max-w-md w-full p-6 md:p-8 shadow-xl max-h-[90vh] overflow-y-auto">
        {/* Close Button */}
        <button onClick={onClose} className="absolute top-4 right-4 p-2 hover:bg-muted rounded-lg">
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 bg-primary/20 rounded-lg mb-4">
            <Gift className="w-6 h-6 text-primary" />
          </div>
          <h2 className="text-xl md:text-2xl font-bold text-foreground mb-2">Unlock {feature}</h2>
          <p className="text-sm md:text-base text-muted-foreground">
            Invite friends and unlock premium features for free
          </p>
        </div>

        {/* Progress */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-foreground">Invites sent</span>
            <span className="text-sm font-semibold text-primary">
              {invitesSent}/{invitesNeeded}
            </span>
          </div>
          <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary h-full transition-all duration-300"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>
        </div>

        {/* Share Options */}
        <div className="space-y-3 mb-6">
          <button
            onClick={handleCopyLink}
            className="w-full flex items-center justify-between p-3 bg-muted hover:bg-muted/80 rounded-lg transition-colors text-foreground"
          >
            <div className="flex items-center gap-3">
              {copied ? <Check className="w-5 h-5 text-green-500" /> : <Copy className="w-5 h-5" />}
              <span className="text-sm font-medium">{copied ? "Copied!" : "Copy Link"}</span>
            </div>
            <span className="text-xs text-muted-foreground">+1</span>
          </button>

          <button
            onClick={handleShareTwitter}
            className="w-full flex items-center justify-between p-3 bg-muted hover:bg-muted/80 rounded-lg transition-colors text-foreground"
          >
            <div className="flex items-center gap-3">
              <Share2 className="w-5 h-5" />
              <span className="text-sm font-medium">Share on Twitter</span>
            </div>
            <span className="text-xs text-muted-foreground">+1</span>
          </button>

          <button
            onClick={handleShareLinkedIn}
            className="w-full flex items-center justify-between p-3 bg-muted hover:bg-muted/80 rounded-lg transition-colors text-foreground"
          >
            <div className="flex items-center gap-3">
              <Users className="w-5 h-5" />
              <span className="text-sm font-medium">Share on LinkedIn</span>
            </div>
            <span className="text-xs text-muted-foreground">+1</span>
          </button>
        </div>

        {/* Benefits */}
        <div className="bg-muted/50 rounded-lg p-4 mb-6">
          <h3 className="text-sm font-semibold text-foreground mb-3">What you'll unlock:</h3>
          <ul className="space-y-2 text-xs md:text-sm text-muted-foreground">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-primary rounded-full flex-shrink-0" />
              Advanced error analysis
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-primary rounded-full flex-shrink-0" />
              Team collaboration features
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-primary rounded-full flex-shrink-0" />
              Priority support
            </li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 flex-col md:flex-row">
          <Button variant="outline" onClick={onClose} className="flex-1 bg-transparent">
            Maybe later
          </Button>
          <Button
            onClick={onUnlock}
            disabled={invitesSent < invitesNeeded}
            className="flex-1 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {invitesSent >= invitesNeeded ? "Unlock Now" : "Keep Inviting"}
          </Button>
        </div>
      </div>
    </div>
  )
}
