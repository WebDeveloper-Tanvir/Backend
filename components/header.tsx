"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Menu, X } from "lucide-react"
import { SignInButton, SignUpButton, SignedIn, SignedOut, UserButton, useUser } from "@clerk/nextjs"
import { classifyEmail } from "@/lib/emailClassifier"

export function Header() {
  const [isOpen, setIsOpen]       = useState(false)
  const [scrolled, setScrolled]   = useState(false)

  // ── Smart workspace routing based on email type ───────────────────────────
  const { user, isSignedIn } = useUser()
  const _email   = user?.primaryEmailAddress?.emailAddress ?? ""
  const _type    = isSignedIn && _email ? classifyEmail(_email) : null
  const _showDev = _type === "personal" || _type === "org"   // gmail/company → Dev Mode
  const _showIDE = _type === "student"                        // .edu/.ac.in   → IDE
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24)
    window.addEventListener("scroll", onScroll)
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <header
      className={`sticky top-0 z-50 w-full transition-all duration-300 ${
        scrolled
          ? "bg-[#030e09]/90 backdrop-blur-xl border-b border-[rgba(0,201,167,0.12)]"
          : "bg-transparent border-b border-transparent"
      }`}
    >
      <nav className="container mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">

        {/* Logo → Home */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="relative w-9 h-9 shrink-0">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-[#00c9a7] to-[#4361ee] opacity-0 group-hover:opacity-40 blur-[8px] transition-opacity duration-300" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/errorless-logo.jpg"
              alt="Errorless"
              className="relative w-9 h-9 rounded-xl object-cover object-center"
              style={{ boxShadow: "0 0 0 1px rgba(0,201,167,0.2)" }}
            />
          </div>
          <span className="font-bold text-[1.05rem] tracking-tight">
            <span className="wordmark-error">error</span>
            <span className="wordmark-less">less</span>
          </span>
        </Link>

        {/* Desktop nav links */}
        <div className="hidden md:flex items-center gap-8">
          {["Features", "Pricing", "FAQ"].map(label => (
            <Link
              key={label}
              href={`#${label.toLowerCase()}`}
              className="text-sm text-[rgba(200,230,220,0.6)] hover:text-[#00c9a7] transition-colors tracking-wide font-medium"
            >
              {label}
            </Link>
          ))}
          {/* ── Smart workspace link — changes based on logged-in email ── */}
          {/* Not signed in → show IDE (default behaviour) */}
          {!isSignedIn && (
            <Link href="/ide" className="text-sm font-semibold text-[#00c9a7] hover:text-[#00e5c5] transition-colors tracking-wide">
              IDE
            </Link>
          )}
          {/* Personal / Org email (gmail, company) → Dev Mode */}
          {_showDev && (
            <Link
              href="/devmode"
              className="flex items-center gap-1.5 text-sm font-bold tracking-wide transition-all hover:-translate-y-px"
              style={{
                color: "#f59e0b",
                background: "rgba(245,158,11,0.1)",
                padding: "4px 12px",
                borderRadius: "8px",
                border: "1px solid rgba(245,158,11,0.25)",
                textDecoration: "none",
              }}
            >
              ⚙ Dev Mode
              <span style={{ fontSize: "0.55rem", fontWeight: 800, background: "#f59e0b", color: "#04070e", padding: "1px 5px", borderRadius: "3px" }}>
                NEW
              </span>
            </Link>
          )}
          {/* Student email (.edu / .ac.in) → IDE */}
          {_showIDE && (
            <Link
              href="/ide"
              className="flex items-center gap-1.5 text-sm font-bold tracking-wide transition-all hover:-translate-y-px"
              style={{
                color: "#58a6ff",
                background: "rgba(88,166,255,0.1)",
                padding: "4px 12px",
                borderRadius: "8px",
                border: "1px solid rgba(88,166,255,0.25)",
                textDecoration: "none",
              }}
            >
              💻 IDE
            </Link>
          )}
        </div>

        {/* Desktop CTA */}
        <div className="hidden md:flex items-center gap-3">
          <SignedOut>
            <SignInButton mode="modal">
              <button className="text-sm font-medium text-[rgba(200,230,220,0.7)] hover:text-white transition-colors px-4 py-1.5">
                Sign In
              </button>
            </SignInButton>
            <SignUpButton mode="modal">
              <button className="relative text-sm font-semibold px-5 py-2 rounded-lg overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee]" />
                <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee] opacity-0 group-hover:opacity-100 blur-sm transition-opacity" />
                <span className="relative text-[#030e09] font-bold">Get Started</span>
              </button>
            </SignUpButton>
          </SignedOut>
          <SignedIn>
            <Link href="/dashboard">
              <button className="text-sm font-medium text-[rgba(200,230,220,0.7)] hover:text-[#00c9a7] transition-colors px-4 py-1.5">
                Dashboard
              </button>
            </Link>
            <UserButton afterSignOutUrl="/" />
          </SignedIn>
        </div>

        {/* Mobile toggle */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="md:hidden p-2 text-[rgba(200,230,220,0.7)] hover:text-white transition-colors"
        >
          {isOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </nav>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden border-t border-[rgba(0,201,167,0.1)] bg-[#030e09]/95 backdrop-blur-xl">
          <div className="flex flex-col gap-1 p-4">
            {["Features", "Pricing", "FAQ"].map(label => (
              <Link
                key={label}
                href={`#${label.toLowerCase()}`}
                onClick={() => setIsOpen(false)}
                className="text-sm text-[rgba(200,230,220,0.7)] hover:text-[#00c9a7] transition-colors px-3 py-2.5 rounded-lg hover:bg-[rgba(0,201,167,0.06)]"
              >
                {label}
              </Link>
            ))}
            {/* Smart workspace link in mobile menu */}
            {!isSignedIn && (
              <Link href="/ide" onClick={() => setIsOpen(false)}
                className="text-sm text-[rgba(200,230,220,0.7)] hover:text-[#00c9a7] transition-colors px-3 py-2.5 rounded-lg hover:bg-[rgba(0,201,167,0.06)]">
                IDE
              </Link>
            )}
            {_showDev && (
              <Link href="/devmode" onClick={() => setIsOpen(false)}
                className="flex items-center gap-2 text-sm font-bold px-3 py-2.5 rounded-lg"
                style={{ color: "#f59e0b", background: "rgba(245,158,11,0.08)" }}>
                ⚙ Dev Mode
                <span style={{ fontSize: "0.55rem", fontWeight: 800, background: "#f59e0b", color: "#04070e", padding: "1px 5px", borderRadius: "3px" }}>NEW</span>
              </Link>
            )}
            {_showIDE && (
              <Link href="/ide" onClick={() => setIsOpen(false)}
                className="flex items-center gap-2 text-sm font-bold px-3 py-2.5 rounded-lg"
                style={{ color: "#58a6ff", background: "rgba(88,166,255,0.08)" }}>
                💻 IDE
              </Link>
            )}
            <div className="border-t border-[rgba(0,201,167,0.08)] mt-2 pt-3 flex flex-col gap-2">
              <SignedOut>
                <SignInButton mode="modal">
                  <button className="w-full text-left text-sm text-[rgba(200,230,220,0.7)] px-3 py-2.5 rounded-lg hover:bg-[rgba(0,201,167,0.06)] transition-colors">
                    Sign In
                  </button>
                </SignInButton>
                <SignUpButton mode="modal">
                  <button className="w-full text-sm font-semibold py-2.5 rounded-lg bg-gradient-to-r from-[#00c9a7] to-[#4361ee] text-[#030e09]">
                    Get Started
                  </button>
                </SignUpButton>
              </SignedOut>
              <SignedIn>
                <Link href="/dashboard" onClick={() => setIsOpen(false)}
                  className="text-sm text-[rgba(200,230,220,0.7)] px-3 py-2.5 rounded-lg hover:bg-[rgba(0,201,167,0.06)]">
                  Dashboard
                </Link>
                <div className="px-3"><UserButton afterSignOutUrl="/" /></div>
              </SignedIn>
            </div>
          </div>
        </div>
      )}
    </header>
  )
}
