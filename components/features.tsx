"use client"

import { Brain, Zap, Users, Lock, BarChart3, Rocket } from "lucide-react"

const features = [
  { icon: Brain,    title: "AI-Powered Analysis",   gradient: "from-[#00c9a7] to-[#00b4d8]",
    desc: "Context-aware error detection that understands your codebase and provides intelligent, precise solutions." },
  { icon: Zap,      title: "Real-Time Fixes",        gradient: "from-[#00b4d8] to-[#4361ee]",
    desc: "Get instant error fixes without leaving your editor. Apply AI suggestions with a single click." },
  { icon: Users,    title: "Team Collaboration",     gradient: "from-[#4361ee] to-[#7b5ea7]",
    desc: "Share fixes with your team, build a knowledge library of solutions, and learn together." },
  { icon: Lock,     title: "Secure & Private",       gradient: "from-[#7b5ea7] to-[#00c9a7]",
    desc: "Your code stays private. Enterprise-grade security with end-to-end encryption at every layer." },
  { icon: BarChart3,title: "Analytics Dashboard",    gradient: "from-[#00c9a7] to-[#4361ee]",
    desc: "Track error patterns, team productivity, and code quality metrics in real-time dashboards." },
  { icon: Rocket,   title: "Lightning Fast",         gradient: "from-[#00b4d8] to-[#00c9a7]",
    desc: "Optimized for speed at every level. Results in milliseconds, not seconds. Never blocks your flow." },
]

export function Features() {
  return (
    <section id="features" className="w-full py-20 lg:py-32 relative overflow-hidden">

      {/* Subtle bg glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80vw] h-[60vh] rounded-full"
          style={{ background: "radial-gradient(ellipse, rgba(0,180,216,0.04) 0%, transparent 70%)" }} />
      </div>

      <div className="relative container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="flex flex-col items-center text-center gap-5 mb-16">
          <div className="px-4 py-1.5 rounded-full text-xs font-semibold tracking-widest uppercase"
            style={{ border: "1px solid rgba(0,201,167,0.2)", background: "rgba(0,201,167,0.05)", color: "#00c9a7" }}>
            Features
          </div>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-balance leading-tight">
            Powerful Tools for<br />
            <span className="brand-gradient">Modern Developers</span>
          </h2>
          <p className="text-base text-[rgba(200,230,220,0.55)] max-w-xl leading-relaxed">
            Everything you need to write better code faster — from AI error detection to full team collaboration.
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => {
            const Icon = f.icon
            return (
              <div
                key={i}
                className="group relative rounded-2xl p-6 overflow-hidden transition-all duration-300 cursor-default"
                style={{
                  background: "rgba(5,25,16,0.5)",
                  border: "1px solid rgba(0,201,167,0.1)",
                  backdropFilter: "blur(12px)",
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLDivElement).style.border = "1px solid rgba(0,201,167,0.28)"
                  ;(e.currentTarget as HTMLDivElement).style.background = "rgba(5,30,18,0.75)"
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLDivElement).style.border = "1px solid rgba(0,201,167,0.1)"
                  ;(e.currentTarget as HTMLDivElement).style.background = "rgba(5,25,16,0.5)"
                }}
              >
                {/* Corner glow on hover */}
                <div className="absolute top-0 right-0 w-24 h-24 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ background: "radial-gradient(circle, rgba(0,201,167,0.12) 0%, transparent 70%)", transform: "translate(30%,-30%)" }} />

                {/* Icon */}
                <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${f.gradient} p-px mb-5`}>
                  <div className="w-full h-full rounded-xl bg-[#030e09] flex items-center justify-center">
                    <Icon size={20} className="text-[#00c9a7] group-hover:scale-110 transition-transform duration-300" />
                  </div>
                </div>

                <h3 className="font-bold text-base text-white mb-2 tracking-tight">{f.title}</h3>
                <p className="text-sm text-[rgba(200,230,220,0.5)] leading-relaxed">{f.desc}</p>

                {/* Bottom gradient line */}
                <div className={`absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r ${f.gradient} opacity-0 group-hover:opacity-60 transition-opacity duration-300`} />
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
