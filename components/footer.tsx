"use client"

import Link from "next/link"
import { Github, Twitter, Linkedin } from "lucide-react"

const links = {
  Product: ["Features", "Pricing", "Security", "Changelog"],
  Company:  ["About", "Blog", "Careers", "Press"],
  Legal:    ["Privacy", "Terms", "Contact", "Cookies"],
}

export function Footer() {
  return (
    <footer className="w-full border-t relative overflow-hidden" style={{ borderColor: "rgba(0,201,167,0.1)", background: "#020b06" }}>

      {/* Subtle top glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[60vw] h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(0,201,167,0.3), rgba(67,97,238,0.3), transparent)" }} />

      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-14">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10 mb-12">

          {/* Brand col */}
          <div className="col-span-2 md:col-span-2 flex flex-col gap-4">
            <Link href="/" className="flex items-center gap-2.5 w-fit group">
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
              <span className="font-bold text-[1.05rem]">
                <span className="wordmark-error">error</span>
                <span className="wordmark-less">less</span>
              </span>
            </Link>
            <p className="text-sm text-[rgba(200,230,220,0.45)] leading-relaxed max-w-[220px]">
              AI-powered code error solver for modern developers. Ship faster, debug less.
            </p>
            <div className="flex items-center gap-3 mt-1">
              {[Github, Twitter, Linkedin].map((Icon, i) => (
                <Link key={i} href="#"
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-[rgba(200,230,220,0.4)] hover:text-[#00c9a7] transition-colors"
                  style={{ border: "1px solid rgba(0,201,167,0.12)", background: "rgba(0,201,167,0.04)" }}>
                  <Icon size={15} />
                </Link>
              ))}
            </div>
          </div>

          {/* Link cols */}
          {Object.entries(links).map(([group, items]) => (
            <div key={group} className="flex flex-col gap-3">
              <h4 className="text-xs font-semibold tracking-widest uppercase text-[rgba(200,230,220,0.35)]">{group}</h4>
              {items.map(item => (
                <Link key={item} href="#"
                  className="text-sm text-[rgba(200,230,220,0.5)] hover:text-[#00c9a7] transition-colors">
                  {item}
                </Link>
              ))}
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-8 border-t"
          style={{ borderColor: "rgba(0,201,167,0.07)" }}>
          <p className="text-xs text-[rgba(200,230,220,0.3)]">
            © {new Date().getFullYear()} Errorless. All rights reserved.
          </p>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00c9a7] animate-pulse" />
            <span className="text-xs font-mono text-[rgba(200,230,220,0.3)]">All systems operational</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
