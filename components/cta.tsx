"use client"

import Link from "next/link"
import { ArrowRight, Sparkles } from "lucide-react"

export function CTA() {
  return (
    <section className="w-full py-20 lg:py-32 bg-mesh relative overflow-hidden">

      {/* Large central glow */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-[70vw] h-[50vh] max-w-[900px] rounded-full"
          style={{ background: "radial-gradient(ellipse, rgba(0,201,167,0.08) 0%, rgba(67,97,238,0.05) 50%, transparent 80%)" }} />
      </div>

      <div className="relative container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="relative rounded-3xl overflow-hidden p-[1px]"
          style={{ background: "linear-gradient(135deg, rgba(0,201,167,0.4), rgba(0,180,216,0.2), rgba(67,97,238,0.4))" }}>
          <div className="rounded-3xl px-8 sm:px-14 py-16 text-center relative overflow-hidden"
            style={{ background: "linear-gradient(160deg, rgba(4,18,11,0.97) 0%, rgba(5,12,25,0.97) 100%)" }}>

            {/* Inner glows */}
            <div className="absolute top-0 left-0 w-64 h-64 rounded-full"
              style={{ background: "radial-gradient(circle, rgba(0,201,167,0.1) 0%, transparent 70%)", transform: "translate(-30%,-30%)" }} />
            <div className="absolute bottom-0 right-0 w-64 h-64 rounded-full"
              style={{ background: "radial-gradient(circle, rgba(67,97,238,0.12) 0%, transparent 70%)", transform: "translate(30%,30%)" }} />

            <div className="relative flex flex-col items-center gap-8">
              {/* Icon badge */}
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, rgba(0,201,167,0.15), rgba(67,97,238,0.15))", border: "1px solid rgba(0,201,167,0.25)" }}>
                <Sparkles size={24} className="text-[#00c9a7]" />
              </div>

              <div className="space-y-4 max-w-2xl">
                <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-balance leading-tight">
                  Ready to Code Without<br />
                  <span className="brand-gradient">Errors Holding You Back?</span>
                </h2>
                <p className="text-base text-[rgba(200,230,220,0.55)] leading-relaxed">
                  Join thousands of developers already using Errorless to write better code, ship faster, and debug less.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/sign-up">
                  <button className="relative group px-8 py-3.5 rounded-xl font-bold text-sm overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee]" />
                    <div className="absolute inset-0 bg-gradient-to-r from-[#00c9a7] via-[#00b4d8] to-[#4361ee] opacity-0 group-hover:opacity-100 blur-md transition-opacity" />
                    <span className="relative flex items-center gap-2 text-[#030e09]">
                      Start Free — No Card Needed
                      <ArrowRight size={16} />
                    </span>
                  </button>
                </Link>
                <Link href="/demo">
                  <button className="px-8 py-3.5 rounded-xl font-semibold text-sm border text-[rgba(200,230,220,0.75)] hover:text-white hover:border-[rgba(0,201,167,0.35)] transition-all"
                    style={{ borderColor: "rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.04)" }}>
                    Watch Demo
                  </button>
                </Link>
              </div>

              <p className="text-xs text-[rgba(200,230,220,0.3)] tracking-wide">
                Free forever plan · No credit card required · Setup in 30 seconds
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
