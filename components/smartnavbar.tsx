"use client"

/**
 * components/SmartNavbar.tsx
 *
 * ✅ FIXED: Now reads the logged-in user directly from Clerk's useUser()
 * instead of the custom UserContext (which was never being populated).
 *
 * After login with a personal email → ⚙ Dev Mode appears in navbar
 * After login with a student email  → 💻 IDE appears in navbar
 * Not logged in                     → "Your Workspace" greyed out
 */

import Link from "next/link"
import { useState } from "react"
import { useUser, useClerk } from "@clerk/nextjs"
import { classifyEmail, accountLabel, accountEmoji } from "@/lib/emailClassifier"

const navItems: Record<string, { href: string; label: string; icon: string; description: string; color: string; badge?: string }> = {
  student: { href: "/ide", label: "IDE", icon: "💻", description: "Student IDE", color: "#58a6ff" },
  personal: { href: "/dev-mode", label: "Dev Mode", icon: "⚙", description: "Dev Mode", color: "#f59e0b" },
  organization: { href: "/dev-mode", label: "Dev Mode", icon: "⚙", description: "Dev Mode", color: "#f59e0b" },
}

export function SmartNavbar({ activePage }: { activePage?: string }) {
  const { user, isLoaded, isSignedIn } = useUser()  // ← Clerk, NOT UserContext
  const { signOut } = useClerk()
  const [mob, setMob]         = useState(false)
  const [dropdown, setDropdown] = useState(false)

  // ── THE CORE: classify the Clerk email ─────────────────────────────────────
  const email    = user?.primaryEmailAddress?.emailAddress ?? ""
  const userType = isSignedIn && email ? classifyEmail(email) : null
  const navItem  = userType ? navItems[userType] : null
  // ──────────────────────────────────────────────────────────────────────────

  return (
    <>
      <nav className="snav">
        <Link href="/" className="logo" onClick={() => setMob(false)}>
          <span className="lt">&lt;</span>Errorless<span className="lt">/&gt;</span>
        </Link>

        {/* ── Desktop centre links ── */}
        <div className="links">
          <Link href="/learn" className={`lnk ${activePage === "learn" ? "lnk-on" : ""}`}>
            Learn
          </Link>

          {!isLoaded ? (
            <div className="lnk-skel" />
          ) : navItem ? (
            // ✅ Shows "💻 IDE" for students, "⚙ Dev Mode" for personal/org
            <Link
              href={navItem.href}
              className={`lnk lnk-smart ${userType === "student" ? "lnk-ide" : "lnk-dev"}`}
              title={navItem.description}
            >
              <span className="ni">{navItem.icon}</span>
              {navItem.label}
              {navItem.badge && <span className="nbadge">{navItem.badge}</span>}
            </Link>
          ) : (
            <span className="lnk lnk-locked">Your Workspace</span>
          )}

          <Link href="/pricing" className={`lnk ${activePage === "pricing" ? "lnk-on" : ""}`}>
            Pricing
          </Link>
        </div>

        {/* ── Desktop right ── */}
        <div className="nav-right">
          {!isLoaded ? (
            <div className="lnk-skel" style={{ width: 90 }} />
          ) : isSignedIn && user ? (
            <div className="prof" onMouseLeave={() => setDropdown(false)}>
              <button className="prof-btn" onClick={() => setDropdown(d => !d)}>
                {user.imageUrl
                  ? <img src={user.imageUrl} className="av-img" alt="" />
                  : <div className="av-init">{(user.firstName || email).charAt(0).toUpperCase()}</div>}
                <div className="av-text">
                  <span className="av-name">{user.firstName || email.split("@")[0]}</span>
                  {userType && (
                    <span className="av-role" style={{ color: navItem?.color }}>
                      {accountEmoji(userType)} {accountLabel(userType)}
                    </span>
                  )}
                </div>
                <span className="caret">{dropdown ? "▴" : "▾"}</span>
              </button>

              {dropdown && (
                <div className="dd">
                  <div className="dd-head">
                    <div className="dd-email">{email}</div>
                    {userType && (
                      <div className="dd-type" style={{ color: navItem?.color }}>
                        {accountLabel(userType)}
                      </div>
                    )}
                  </div>
                  <div className="dd-sep" />
                  <Link href="/dashboard" className="dd-item" onClick={() => setDropdown(false)}>📊 Dashboard</Link>
                  {navItem && (
                    <Link href={navItem.href} className="dd-item" onClick={() => setDropdown(false)}>
                      {navItem.icon} {navItem.label}
                    </Link>
                  )}
                  <Link href="/settings" className="dd-item" onClick={() => setDropdown(false)}>⚙ Settings</Link>
                  <div className="dd-sep" />
                  <button className="dd-out" onClick={() => signOut({ redirectUrl: "/" })}>↩ Sign Out</button>
                </div>
              )}
            </div>
          ) : (
            <>
              <Link href="/login" className="btn-ghost">Sign In</Link>
              <Link href="/signup" className="btn-solid">Get Started →</Link>
            </>
          )}

          <button className="ham" onClick={() => setMob(m => !m)} aria-label="Menu">
            <span className={`hb ${mob ? "hb1" : ""}`} />
            <span className={`hb ${mob ? "hb2" : ""}`} />
            <span className={`hb ${mob ? "hb3" : ""}`} />
          </button>
        </div>
      </nav>

      {/* Mobile menu */}
      <div className={`mob ${mob ? "mob-open" : ""}`}>
        <Link href="/learn" className="mi" onClick={() => setMob(false)}>Learn</Link>

        {navItem ? (
          <Link
            href={navItem.href}
            className={`mi mi-smart ${userType === "student" ? "mi-ide" : "mi-dev"}`}
            onClick={() => setMob(false)}
          >
            {navItem.icon} {navItem.label}
            {navItem.badge
              ? <span className="mchip mchip-a">{navItem.badge}</span>
              : <span className="mchip mchip-b">Student</span>}
          </Link>
        ) : (
          <span className="mi mi-locked">Sign in to access workspace</span>
        )}

        <Link href="/pricing" className="mi" onClick={() => setMob(false)}>Pricing</Link>
        <div className="msep" />

        {isSignedIn && user ? (
          <>
            <div className="m-user">
              <div className="m-name">{user.fullName || email.split("@")[0]}</div>
              <div className="m-email">{email}</div>
              {userType && navItem && (
                <div className="m-ws" style={{ color: navItem.color }}>
                  {navItem.icon} {navItem.label}
                </div>
              )}
            </div>
            <button className="m-out" onClick={() => signOut({ redirectUrl: "/" })}>↩ Sign Out</button>
          </>
        ) : (
          <>
            <Link href="/login" className="mi" onClick={() => setMob(false)}>Sign In</Link>
            <Link href="/signup" className="m-cta" onClick={() => setMob(false)}>Get Started →</Link>
          </>
        )}
      </div>

      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800;900&family=Outfit:wght@400;500;600;700&display=swap');

        .snav {
          position: sticky; top: 0; z-index: 500;
          display: flex; align-items: center; justify-content: space-between;
          height: 54px; padding: 0 2rem;
          background: rgba(4,7,14,0.93); backdrop-filter: blur(28px);
          border-bottom: 1px solid rgba(255,255,255,0.07);
          font-family: 'Outfit', sans-serif;
        }
        .logo { font-family: 'Syne', sans-serif; font-weight: 900; font-size: 1rem; text-decoration: none; color: #dde4f5; letter-spacing: .05em; }
        .lt { color: #00ff88; }

        .links { display: flex; align-items: center; gap: 3px; }
        .lnk { color: #68788f; text-decoration: none; font-size: .875rem; font-weight: 500; padding: .36rem .78rem; border-radius: 8px; transition: all .15s; border: 1px solid transparent; display: flex; align-items: center; gap: .32rem; }
        .lnk:hover { color: #dde4f5; background: rgba(255,255,255,.05); }
        .lnk-on { color: #dde4f5; background: rgba(255,255,255,.06); }
        .lnk-locked { opacity: .32; cursor: default; font-size: .82rem; font-style: italic; pointer-events: none; }

        /* ✅ Dev Mode — amber pill (personal/org users) */
        .lnk-dev { color: #f59e0b; background: rgba(245,158,11,.08); border-color: rgba(245,158,11,.22); }
        .lnk-dev:hover { color: #fbbf24; background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.4); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(245,158,11,.18); }

        /* ✅ IDE — blue pill (student users) */
        .lnk-ide { color: #58a6ff; background: rgba(88,166,255,.08); border-color: rgba(88,166,255,.22); }
        .lnk-ide:hover { color: #7ec8ff; background: rgba(88,166,255,.14); border-color: rgba(88,166,255,.4); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(88,166,255,.18); }

        .ni { font-size: .82rem; }
        .nbadge { font-size: .58rem; font-weight: 800; background: #f59e0b; color: #050810; padding: 1px 5px; border-radius: 3px; }

        .lnk-skel { width: 90px; height: 30px; border-radius: 8px; background: rgba(255,255,255,.06); animation: sh 1.5s infinite; }
        @keyframes sh { 0%,100%{opacity:.4} 50%{opacity:.9} }

        .nav-right { display: flex; align-items: center; gap: .55rem; }
        .btn-ghost { color: #68788f; text-decoration: none; font-size: .85rem; padding: .36rem .78rem; border-radius: 8px; transition: color .15s; }
        .btn-ghost:hover { color: #dde4f5; }
        .btn-solid { background: #00ff88; color: #04070e; text-decoration: none; padding: .4rem 1.1rem; border-radius: 8px; font-size: .85rem; font-weight: 700; transition: all .2s; }
        .btn-solid:hover { transform: translateY(-1px); box-shadow: 0 4px 18px rgba(0,255,136,.3); }

        /* Profile */
        .prof { position: relative; }
        .prof-btn { display: flex; align-items: center; gap: .45rem; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); border-radius: 10px; padding: .3rem .6rem .3rem .38rem; cursor: pointer; transition: all .15s; }
        .prof-btn:hover { background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.16); }
        .av-img { width: 26px; height: 26px; border-radius: 50%; object-fit: cover; }
        .av-init { width: 26px; height: 26px; border-radius: 50%; background: linear-gradient(135deg,#00ff88,#00ccff); display: flex; align-items: center; justify-content: center; font-size: .8rem; font-weight: 800; color: #04070e; flex-shrink: 0; }
        .av-text { display: flex; flex-direction: column; text-align: left; }
        .av-name { font-size: .8rem; font-weight: 600; color: #dde4f5; line-height: 1.15; }
        .av-role { font-size: .62rem; font-weight: 600; line-height: 1.15; }
        .caret { font-size: .6rem; color: #68788f; }

        .dd { position: absolute; top: calc(100% + 8px); right: 0; width: 230px; background: #0b1022; border: 1px solid rgba(255,255,255,.1); border-radius: 14px; padding: .45rem; box-shadow: 0 20px 60px rgba(0,0,0,.65); z-index: 600; animation: ddin .15s ease; }
        @keyframes ddin { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
        .dd-head { padding: .5rem .65rem .4rem; }
        .dd-email { font-size: .7rem; color: #68788f; word-break: break-all; margin-bottom: .18rem; }
        .dd-type { font-size: .7rem; font-weight: 700; }
        .dd-sep { height: 1px; background: rgba(255,255,255,.06); margin: .3rem 0; }
        .dd-item { display: flex; align-items: center; gap: .5rem; padding: .46rem .65rem; border-radius: 7px; color: #9baab8; text-decoration: none; font-size: .81rem; transition: all .12s; }
        .dd-item:hover { background: rgba(255,255,255,.05); color: #dde4f5; }
        .dd-out { width: 100%; display: flex; align-items: center; gap: .5rem; padding: .46rem .65rem; border-radius: 7px; color: #f85149; background: none; border: none; font-size: .81rem; cursor: pointer; font-family: inherit; }
        .dd-out:hover { background: rgba(248,81,73,.08); }

        /* Ham */
        .ham { display: none; flex-direction: column; gap: 5px; background: none; border: none; cursor: pointer; padding: 4px; }
        .hb { width: 20px; height: 2px; background: #9baab8; border-radius: 2px; display: block; transition: all .25s; transform-origin: center; }
        .hb1 { transform: translateY(7px) rotate(45deg); }
        .hb2 { opacity: 0; transform: scaleX(0); }
        .hb3 { transform: translateY(-7px) rotate(-45deg); }

        /* Mobile */
        .mob { display: none; flex-direction: column; position: sticky; top: 54px; z-index: 499; background: rgba(4,7,14,.98); border-bottom: 1px solid rgba(255,255,255,.07); padding: .55rem 1.25rem 1rem; transform: translateY(-110%); opacity: 0; transition: all .22s ease; pointer-events: none; font-family: 'Outfit', sans-serif; }
        .mob-open { transform: translateY(0); opacity: 1; pointer-events: all; }
        .mi { color: #68788f; text-decoration: none; font-size: .9rem; padding: .58rem .2rem; display: flex; align-items: center; gap: .45rem; border-bottom: 1px solid rgba(255,255,255,.04); transition: color .15s; }
        .mi:hover { color: #dde4f5; }
        .mi-smart { font-weight: 600; }
        .mi-ide { color: #58a6ff; }
        .mi-dev { color: #f59e0b; }
        .mi-locked { opacity: .32; cursor: default; font-size: .82rem; font-style: italic; }
        .mchip { margin-left: auto; font-size: .62rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; }
        .mchip-a { background: #f59e0b; color: #04070e; }
        .mchip-b { background: rgba(88,166,255,.15); color: #58a6ff; }
        .msep { height: 1px; background: rgba(255,255,255,.06); margin: .35rem 0; }
        .m-user { padding: .4rem .2rem .2rem; }
        .m-name { font-size: .88rem; font-weight: 600; color: #dde4f5; }
        .m-email { font-size: .7rem; color: #68788f; word-break: break-all; }
        .m-ws { font-size: .75rem; font-weight: 700; margin-top: .2rem; }
        .m-out { background: none; border: none; color: #f85149; font-size: .88rem; padding: .58rem .2rem; cursor: pointer; font-family: inherit; text-align: left; }
        .m-cta { background: #00ff88; color: #04070e; text-decoration: none; text-align: center; padding: .65rem; border-radius: 8px; font-weight: 700; margin-top: .4rem; display: block; }

        @media (max-width: 768px) {
          .links, .btn-ghost, .btn-solid, .prof { display: none; }
          .ham { display: flex; }
          .mob { display: flex; }
          .snav { padding: 0 1.25rem; }
        }
      `}</style>
    </>
  )
}
