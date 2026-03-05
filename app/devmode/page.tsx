"use client"

import { useState, useRef, useEffect, useCallback } from "react"

type Plan = "none" | "basic" | "premium"
type Tab = "generate" | "analyze" | "optimize"
type UploadedFile = { name: string; content: string; language: string }

declare global {
  interface Window {
    Razorpay: new (o: Record<string, unknown>) => { open(): void }
  }
}

const PLANS = {
  basic: {
    id: "basic", name: "Basic", price: "$2.5", priceINR: "₹209",
    amountPaise: 20900, color: "#f59e0b", icon: "⚡",
    features: [
      { text: "Code Generation", ok: true },
      { text: "Code Analysis", ok: true },
      { text: "Code Optimization", ok: false },
      { text: "24-hour memory only", ok: true, italic: true },
      { text: "Permanent memory", ok: false },
    ],
    desc: "Generate & analyze code. Memory auto-resets every 24 hours.",
  },
  premium: {
    id: "premium", name: "Premium", price: "$5", priceINR: "₹419",
    amountPaise: 41900, color: "#00ff88", icon: "👑",
    features: [
      { text: "Code Generation", ok: true },
      { text: "Code Analysis", ok: true },
      { text: "Code Optimization", ok: true },
      { text: "Permanent memory saved", ok: true },
      { text: "Priority AI processing", ok: true },
    ],
    desc: "Full access: generate, analyze, optimize + permanent memory.",
  },
}

function detectLanguage(filename: string, content: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() || ""
  const m: Record<string, string> = {
    py: "python", js: "javascript", ts: "typescript", tsx: "typescript",
    jsx: "javascript", java: "java", cpp: "cpp", c: "c", cs: "csharp",
    rs: "rust", go: "go", rb: "ruby", php: "php", swift: "swift",
    kt: "kotlin", r: "r", lua: "lua", sh: "bash", html: "html", css: "css",
  }
  if (m[ext]) return m[ext]
  if (content.includes("def ") && content.includes(":")) return "python"
  if (content.includes("function") || content.includes("const ")) return "javascript"
  return "text"
}

const STORAGE_KEY = "errorless_devmode_v1"
function saveMemory(plan: Plan, data: object) {
  if (plan === "none") return
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...data, plan, ts: Date.now() })) } catch {}
}
function loadMemory(plan: Plan) {
  if (plan === "none") return null
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const p = JSON.parse(raw)
    if (plan === "basic" && Date.now() - p.ts > 86400000) {
      localStorage.removeItem(STORAGE_KEY); return null
    }
    return p
  } catch { return null }
}

export default function DevModePage() {
  const [plan, setPlan] = useState<Plan>("none")
  const [paywall, setPaywall] = useState(true)
  const [payLoading, setPayLoading] = useState<string | null>(null)
  const [toastMsg, setToastMsg] = useState("")

  const [tab, setTab] = useState<Tab>("generate")
  const [file, setFile] = useState<UploadedFile | null>(null)
  const [prompt, setPrompt] = useState("")
  const [lang, setLang] = useState("python")
  const [generated, setGenerated] = useState("")
  const [analysis, setAnalysis] = useState<Record<string, string> | null>(null)
  const [optimized, setOptimized] = useState("")
  const [loading, setLoading] = useState(false)
  const [drag, setDrag] = useState(false)
  const [copied, setCopied] = useState(false)

  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const saved = loadMemory(plan)
    if (saved) {
      if (saved.generated) setGenerated(saved.generated)
      if (saved.analysis) setAnalysis(saved.analysis)
      if (saved.optimized) setOptimized(saved.optimized)
    }
  }, [plan])

  useEffect(() => {
    saveMemory(plan, { generated, analysis, optimized })
  }, [generated, analysis, optimized, plan])

  const showToast = (msg: string) => {
    setToastMsg(msg)
    setTimeout(() => setToastMsg(""), 4000)
  }

  const loadRzp = (): Promise<boolean> =>
    new Promise((res) => {
      if (window.Razorpay) return res(true)
      const s = document.createElement("script")
      s.src = "https://checkout.razorpay.com/v1/checkout.js"
      s.onload = () => res(true); s.onerror = () => res(false)
      document.body.appendChild(s)
    })

  const handlePay = async (pid: "basic" | "premium") => {
    setPayLoading(pid)
    const ok = await loadRzp()
    if (!ok) { alert("Payment gateway failed."); setPayLoading(null); return }

    try {
      const r = await fetch("/api/devmode/create-order", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan: pid }),
      })
      const order = await r.json()
      if (order.error) { alert(order.error); setPayLoading(null); return }

      new window.Razorpay({
        key: "rzp_live_S9yC5XI9qt6E9a",
        amount: order.amount, currency: order.currency || "INR",
        name: "Errorless Dev Mode", description: PLANS[pid].desc,
        order_id: order.orderId, theme: { color: PLANS[pid].color },
        handler: async (resp: { razorpay_payment_id: string; razorpay_order_id: string; razorpay_signature: string }) => {
          const v = await fetch("/api/devmode/verify-payment", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...resp, plan: pid }),
          })
          const vd = await v.json()
          if (vd.success) {
            setPlan(pid); setPaywall(false)
            showToast(`🎉 ${PLANS[pid].name} activated! Welcome to Dev Mode.`)
          } else alert("Verification failed. Contact support.")
        },
        modal: { ondismiss: () => setPayLoading(null) },
      }).open()
    } catch { alert("Payment error. Try again."); setPayLoading(null) }
  }

  const uploadFile = useCallback((f: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target?.result as string
      setFile({ name: f.name, content, language: detectLanguage(f.name, content) })
      setTab("analyze"); setAnalysis(null); setOptimized("")
    }
    reader.readAsText(f)
  }, [])

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setLoading(true); setGenerated("")
    try {
      const res = await fetch("/api/devmode/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, language: lang, plan }),
      })
      const d = await res.json()
      setGenerated(d.code || d.error || "No response received.")
    } catch { setGenerated("Generation failed. Check connection.") }
    setLoading(false)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true); setAnalysis(null)
    try {
      const res = await fetch("/api/devmode/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: file.content, language: file.language, plan }),
      })
      const d = await res.json()
      setAnalysis(d)
    } catch { setAnalysis({ error: "Analysis failed." }) }
    setLoading(false)
  }

  const handleOptimize = async () => {
    if (!file || plan !== "premium") return
    setLoading(true); setOptimized("")
    try {
      const res = await fetch("/api/devmode/optimize", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: file.content, language: file.language, analysis }),
      })
      const d = await res.json()
      setOptimized(d.optimized || d.error || "")
    } catch { setOptimized("Optimization failed.") }
    setLoading(false)
  }

  const copy = (txt: string) => {
    navigator.clipboard.writeText(txt); setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadFile = (content: string, filename: string) => {
    const b = new Blob([content], { type: "text/plain" })
    const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = filename; a.click()
  }

  // ─── PAYWALL ────────────────────────────────────────────────────────────────
  if (paywall) return (
    <>
      <div className="dv-root">
        <div className="mesh-bg" />
        <nav className="top-nav">
          <a href="/" className="nav-brand"><span className="brand-lt">&lt;</span>Errorless<span className="brand-lt">/&gt;</span></a>
          <span className="nav-chip">🔒 Dev Mode</span>
          <a href="/ide" className="nav-ghost">← Back to IDE</a>
        </nav>

        <div className="pw-wrap">
          <div className="pw-eyebrow">
            <div className="terminal-strip">
              <span className="dot r" /><span className="dot y" /><span className="dot g" />
              <code>$ errorless devmode --unlock</code>
            </div>
          </div>
          <h1 className="pw-h1">Unlock<br /><em className="pw-em">Dev Mode</em></h1>
          <p className="pw-p">Generate, analyze, and optimize code with AI. Pick your plan to begin.</p>

          <div className="plans-duo">
            {(["basic", "premium"] as const).map((pid) => {
              const pl = PLANS[pid]
              return (
                <div key={pid} className={`plan-card ${pid === "premium" ? "plan-premium" : ""}`}>
                  {pid === "premium" && <div className="best-badge">👑 Best Value</div>}
                  <div className="plan-glyph">{pl.icon}</div>
                  <h3 className="plan-h3" style={{ color: pl.color }}>{pl.name}</h3>
                  <div className="plan-pricing">
                    <span className="plan-big">{pl.price}</span>
                    <span className="plan-sep"> / </span>
                    <span className="plan-inr">{pl.priceINR}</span>
                    <span className="plan-mo">/mo</span>
                  </div>
                  <p className="plan-blurb">{pl.desc}</p>
                  <ul className="feat-list">
                    {pl.features.map((f, i) => (
                      <li key={i} className={f.ok ? "f-yes" : "f-no"}>
                        <span className="f-bullet">{f.ok ? "✓" : "✕"}</span>
                        <span className={(f as {italic?: boolean}).italic ? "f-italic" : ""}>{f.text}</span>
                      </li>
                    ))}
                  </ul>
                  <button
                    className="pay-btn" style={{ background: pl.color }}
                    onClick={() => handlePay(pid)} disabled={payLoading === pid}
                  >
                    {payLoading === pid
                      ? <><span className="spin" />Processing…</>
                      : <>Pay {pl.priceINR}/month & Unlock</>}
                  </button>
                </div>
              )
            })}
          </div>

          <div className="trust-row">
            <span>🔒 Razorpay Secured</span><span className="bull">·</span>
            <span>💳 UPI · Cards · NetBanking</span><span className="bull">·</span>
            <span>↩️ Cancel anytime</span>
          </div>
        </div>
      </div>
      <GlobalStyles />
    </>
  )

  // ─── MAIN DEV MODE ──────────────────────────────────────────────────────────
  const isPremium = plan === "premium"
  const planInfo = PLANS[plan as "basic" | "premium"]

  return (
    <>
      <div className="dv-root dv-app">
        <div className="mesh-bg" />

        {toastMsg && <div className="toast">{toastMsg}</div>}

        <nav className="top-nav">
          <a href="/" className="nav-brand"><span className="brand-lt">&lt;</span>Errorless<span className="brand-lt">/&gt;</span></a>
          <div className="nav-status">
            <span className="status-pulse" />
            Dev Mode · <span style={{ color: planInfo.color }}>{planInfo.icon} {planInfo.name}</span>
          </div>
          <div className="nav-links-right">
            {plan === "basic" && (
              <button className="upgrade-chip" onClick={() => { setPlan("none"); setPaywall(true) }}>
                ↑ Upgrade to Premium
              </button>
            )}
            <a href="/ide" className="nav-ghost">IDE</a>
            <a href="/" className="nav-ghost">Home</a>
          </div>
        </nav>

        <div className="app-shell">
          {/* SIDEBAR */}
          <aside className="app-side">
            {/* Tabs */}
            <div className="side-section">
              <p className="side-label">Mode</p>
              <div className="side-tabs">
                {(["generate", "analyze", "optimize"] as Tab[]).map((t) => {
                  const locked = t === "optimize" && !isPremium
                  const icons = { generate: "✦", analyze: "◎", optimize: "⟳" }
                  return (
                    <button key={t} className={`side-tab ${tab === t ? "side-tab-on" : ""} ${locked ? "side-tab-locked" : ""}`}
                      onClick={() => !locked && setTab(t)}
                    >
                      <span className="side-tab-icon">{icons[t]}</span>
                      <span>{t.charAt(0).toUpperCase() + t.slice(1)}</span>
                      {locked && <span className="lock-chip">PRO</span>}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Generate controls */}
            {tab === "generate" && (
              <div className="side-section">
                <p className="side-label">Language</p>
                <select className="ctrl-sel" value={lang} onChange={e => setLang(e.target.value)}>
                  {["python","javascript","typescript","java","c","cpp","rust","go","csharp","ruby","php","swift","kotlin","bash"].map(l => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
                <p className="side-label" style={{ marginTop: "0.85rem" }}>What to build?</p>
                <textarea className="ctrl-ta" rows={7} value={prompt} placeholder="Describe your code… e.g. A REST API endpoint in Express.js that handles user authentication with JWT tokens"
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && e.ctrlKey) handleGenerate() }}
                />
                <span className="ctrl-hint">Ctrl+Enter to run</span>
                <button className="action-btn amber-btn" onClick={handleGenerate} disabled={loading || !prompt.trim()}>
                  {loading ? <><span className="spin" />Generating…</> : <><span>✦</span> Generate Code</>}
                </button>
              </div>
            )}

            {/* Analyze controls */}
            {tab === "analyze" && (
              <div className="side-section">
                <p className="side-label">Upload Code File</p>
                <div
                  className={`dropzone ${drag ? "dz-over" : ""} ${file ? "dz-has-file" : ""}`}
                  onDragOver={e => { e.preventDefault(); setDrag(true) }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) uploadFile(f) }}
                  onClick={() => fileRef.current?.click()}
                >
                  <input ref={fileRef} type="file" hidden
                    accept=".py,.js,.ts,.tsx,.jsx,.java,.cpp,.c,.cs,.rs,.go,.rb,.php,.swift,.kt,.r,.lua,.sh,.html,.css"
                    onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f) }}
                  />
                  {file ? (
                    <div className="file-row">
                      <span style={{ fontSize: "1.5rem" }}>📄</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="file-nm">{file.name}</div>
                        <div className="file-mt">{file.language} · {file.content.split("\n").length} lines</div>
                      </div>
                      <button className="file-x" onClick={e => { e.stopPropagation(); setFile(null); setAnalysis(null); setOptimized("") }}>✕</button>
                    </div>
                  ) : (
                    <div className="dz-empty">
                      <span className="dz-icon">⬆</span>
                      <span className="dz-text">Drop your code file here</span>
                      <span className="dz-sub">Any language · Click to browse</span>
                    </div>
                  )}
                </div>
                {file && (
                  <button className="action-btn blue-btn" onClick={handleAnalyze} disabled={loading}>
                    {loading ? <><span className="spin" />Analyzing…</> : <><span>◎</span> Analyze Code</>}
                  </button>
                )}
              </div>
            )}

            {/* Optimize controls */}
            {tab === "optimize" && isPremium && (
              <div className="side-section">
                <p className="side-label">Optimize Code</p>
                {!file ? (
                  <div className="info-box">
                    <span>📁</span>
                    <p>Upload & analyze code first in the <strong>Analyze</strong> tab.</p>
                  </div>
                ) : (
                  <>
                    <div className="mini-file">
                      <span>📄 {file.name}</span>
                      <span className="mini-lang">{file.language}</span>
                    </div>
                    {!analysis && (
                      <div className="info-box" style={{ marginTop: "0.5rem" }}>
                        <span>💡</span><p>Run <strong>Analyze</strong> first for better results.</p>
                      </div>
                    )}
                    <button className="action-btn green-btn" onClick={handleOptimize} disabled={loading} style={{ marginTop: "0.75rem" }}>
                      {loading ? <><span className="spin" />Optimizing…</> : <><span>⟳</span> Optimize Code</>}
                    </button>
                  </>
                )}
              </div>
            )}

            {/* Memory indicator */}
            <div className="mem-card">
              <span className="mem-glyph">🧠</span>
              <div>
                <div className="mem-title">Memory: {isPremium ? "Permanent" : "24h only"}</div>
                <div className="mem-sub">{isPremium ? "Sessions saved forever" : "Resets after 24 hours"}</div>
              </div>
            </div>
          </aside>

          {/* OUTPUT PANEL */}
          <main className="app-main">
            {/* Generate output */}
            {tab === "generate" && (
              <div className="out-panel">
                <div className="out-top">
                  <div className="out-title"><span className="title-glyph amber">✦</span> Generated Code — <span className="lang-badge">{lang}</span></div>
                  {generated && (
                    <div className="out-acts">
                      <button className="icon-btn" onClick={() => copy(generated)}>{copied ? "✓ Copied" : "⎘ Copy"}</button>
                      <button className="icon-btn" onClick={() => downloadFile(generated, `generated.${lang === "python" ? "py" : lang === "javascript" ? "js" : lang}`)}>↓ Save</button>
                    </div>
                  )}
                </div>
                <div className="out-body">
                  {loading && <LoadingSpinner color="#f59e0b" label={`Generating ${lang} code…`} />}
                  {!loading && !generated && <EmptyState icon="✦" color="#f59e0b" text="Describe what you want to build" sub="Supports 14+ languages" />}
                  {!loading && generated && <pre className="code-pre amber-pre"><code>{generated}</code></pre>}
                </div>
              </div>
            )}

            {/* Analyze output */}
            {tab === "analyze" && (
              <div className="out-panel">
                <div className="out-top">
                  <div className="out-title"><span className="title-glyph blue">◎</span> Code Analysis</div>
                </div>
                <div className="out-body">
                  {loading && <LoadingSpinner color="#58a6ff" label="Deep scanning your code…" />}
                  {!loading && !analysis && !file && <EmptyState icon="◎" color="#58a6ff" text="Upload a code file to analyze" sub="Drag & drop any source file" />}
                  {!loading && !analysis && file && <EmptyState icon="📄" color="#58a6ff" text={`${file.name} is ready`} sub="Click Analyze Code →" />}
                  {!loading && analysis && !analysis.error && (
                    <div className="analysis-cards">
                      {[
                        ["📋 Summary", "summary"],
                        ["📊 Complexity", "complexity"],
                        ["⚠️ Issues", "issues"],
                        ["🔒 Security", "security"],
                        ["⚡ Performance", "performance"],
                        ["💡 Suggestions", "suggestions"],
                      ].map(([label, key]) =>
                        analysis[key] ? (
                          <div key={key} className="a-card">
                            <div className="a-card-title">{label}</div>
                            <div className="a-card-body">{analysis[key]}</div>
                          </div>
                        ) : null
                      )}
                      {!isPremium && (
                        <div className="upgrade-card">
                          <span style={{ fontSize: "1.5rem" }}>🔒</span>
                          <div>
                            <strong>Want to optimize this code?</strong>
                            <p>Upgrade to Premium to unlock AI-powered code optimization</p>
                          </div>
                          <button className="up-cta" onClick={() => { setPlan("none"); setPaywall(true) }}>Upgrade →</button>
                        </div>
                      )}
                    </div>
                  )}
                  {!loading && analysis?.error && <div className="err-box">{analysis.error}</div>}
                </div>
              </div>
            )}

            {/* Optimize output */}
            {tab === "optimize" && (
              <div className="out-panel">
                <div className="out-top">
                  <div className="out-title"><span className="title-glyph green">⟳</span> Optimized Code</div>
                  {optimized && (
                    <div className="out-acts">
                      <button className="icon-btn" onClick={() => copy(optimized)}>{copied ? "✓ Copied" : "⎘ Copy"}</button>
                      <button className="icon-btn" onClick={() => downloadFile(optimized, `optimized_${file?.name || "code"}`)}>↓ Save</button>
                    </div>
                  )}
                </div>
                <div className="out-body">
                  {loading && <LoadingSpinner color="#00ff88" label="AI is refactoring your code…" />}
                  {!loading && !optimized && <EmptyState icon="⟳" color="#00ff88" text="Click Optimize Code to begin" sub="AI will refactor, clean, and improve performance" />}
                  {!loading && optimized && <pre className="code-pre green-pre"><code>{optimized}</code></pre>}
                </div>
              </div>
            )}
          </main>
        </div>
      </div>
      <GlobalStyles />
    </>
  )
}

function LoadingSpinner({ color, label }: { color: string; label: string }) {
  return (
    <div className="loading-wrap">
      <div className="loading-ring" style={{ borderTopColor: color }} />
      <p className="loading-label">{label}</p>
    </div>
  )
}

function EmptyState({ icon, color, text, sub }: { icon: string; color: string; text: string; sub: string }) {
  return (
    <div className="empty-wrap">
      <div className="empty-icon" style={{ color }}>{icon}</div>
      <p className="empty-text">{text}</p>
      <span className="empty-sub">{sub}</span>
    </div>
  )
}

function GlobalStyles() {
  return (
    <style jsx global>{`
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800;900&display=swap');
      *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
      :root {
        --bg:#050810; --s:#0c1120; --s2:#101828; --border:rgba(255,255,255,0.07);
        --b2:rgba(255,255,255,0.12); --tx:#dce4f0; --tx2:#6b7a99;
        --green:#00ff88; --amber:#f59e0b; --blue:#58a6ff; --red:#f85149;
        --font:'Outfit',system-ui,sans-serif; --mono:'JetBrains Mono',monospace; --display:'Syne',sans-serif;
      }
      body { background:var(--bg); color:var(--tx); font-family:var(--font); overflow-x:hidden; }
      select option { background: #0c1120; }

      .dv-root { min-height:100vh; position:relative; }
      .dv-app { display:flex; flex-direction:column; height:100vh; overflow:hidden; }

      .mesh-bg {
        position:fixed; inset:0; z-index:0; pointer-events:none;
        background:
          radial-gradient(ellipse 70% 50% at 5% 0%, rgba(0,255,136,.05) 0%, transparent 55%),
          radial-gradient(ellipse 50% 40% at 95% 100%, rgba(88,166,255,.05) 0%, transparent 55%);
      }

      /* NAV */
      .top-nav {
        position:sticky; top:0; z-index:100; display:flex; align-items:center;
        justify-content:space-between; padding:0 1.5rem; height:50px;
        background:rgba(5,8,16,.9); backdrop-filter:blur(24px);
        border-bottom:1px solid var(--border); flex-shrink:0;
      }
      .nav-brand { font-family:var(--display); font-size:.95rem; font-weight:900; text-decoration:none; color:var(--tx); letter-spacing:.06em; }
      .brand-lt { color:var(--green); }
      .nav-chip { background:rgba(0,255,136,.08); border:1px solid rgba(0,255,136,.18); color:var(--green); padding:.28rem .85rem; border-radius:100px; font-size:.75rem; font-weight:600; }
      .nav-status { display:flex; align-items:center; gap:.45rem; font-size:.8rem; font-weight:500; }
      .status-pulse { width:7px; height:7px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
      @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.6)} }
      .nav-links-right { display:flex; align-items:center; gap:.75rem; }
      .nav-ghost { color:var(--tx2); text-decoration:none; font-size:.8rem; transition:color .15s; }
      .nav-ghost:hover { color:var(--tx); }
      .upgrade-chip { background:linear-gradient(135deg,#f59e0b,#f97316); color:#050810; border:none; padding:.3rem .85rem; border-radius:100px; font-size:.75rem; font-weight:700; cursor:pointer; transition:all .2s; }
      .upgrade-chip:hover { transform:translateY(-1px); box-shadow:0 4px 16px rgba(245,158,11,.3); }

      /* PAYWALL */
      .pw-wrap { position:relative; z-index:1; max-width:840px; margin:0 auto; padding:4rem 1.5rem; }
      .pw-eyebrow { text-align:center; margin-bottom:1.5rem; }
      .terminal-strip { display:inline-flex; align-items:center; gap:.5rem; background:var(--s); border:1px solid var(--b2); padding:.4rem 1rem; border-radius:8px; font-family:var(--mono); font-size:.78rem; color:var(--tx2); }
      .dot { width:10px; height:10px; border-radius:50%; display:block; }
      .dot.r{background:#ff5f57;} .dot.y{background:#febc2e;} .dot.g{background:#28c840;}
      .pw-h1 { font-family:var(--display); font-size:clamp(2.8rem,6vw,4.5rem); font-weight:900; text-align:center; line-height:1.05; margin-bottom:1rem; letter-spacing:-.02em; }
      .pw-em { font-style:normal; color:var(--green); }
      .pw-p { text-align:center; color:var(--tx2); font-size:1rem; margin-bottom:3rem; line-height:1.6; }

      .plans-duo { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:2rem; }
      .plan-card { background:var(--s); border:1px solid var(--b2); border-radius:20px; padding:2rem 1.75rem; position:relative; transition:all .3s; overflow:hidden; }
      .plan-card::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at top left,rgba(0,255,136,.03) 0%,transparent 60%); pointer-events:none; }
      .plan-card:hover { transform:translateY(-5px); box-shadow:0 24px 80px rgba(0,0,0,.5); }
      .plan-premium { border-color:rgba(0,255,136,.28); }
      .best-badge { position:absolute; top:14px; right:14px; background:rgba(0,255,136,.1); border:1px solid rgba(0,255,136,.25); color:var(--green); font-size:.68rem; font-weight:700; padding:3px 10px; border-radius:100px; }
      .plan-glyph { font-size:1.8rem; margin-bottom:.75rem; }
      .plan-h3 { font-family:var(--display); font-size:1rem; font-weight:800; margin-bottom:.5rem; letter-spacing:.04em; }
      .plan-pricing { display:flex; align-items:baseline; gap:3px; margin-bottom:.5rem; flex-wrap:wrap; }
      .plan-big { font-family:var(--display); font-size:2.4rem; font-weight:900; }
      .plan-sep { color:var(--tx2); }
      .plan-inr { font-size:1.1rem; font-weight:600; color:var(--tx2); }
      .plan-mo { font-size:.75rem; color:var(--tx2); margin-left:2px; }
      .plan-blurb { color:var(--tx2); font-size:.8rem; margin-bottom:1.2rem; padding-bottom:1.2rem; border-bottom:1px solid var(--border); line-height:1.55; }
      .feat-list { list-style:none; display:flex; flex-direction:column; gap:.5rem; margin-bottom:1.5rem; }
      .feat-list li { display:flex; align-items:center; gap:.5rem; font-size:.82rem; }
      .f-yes { color:var(--tx); } .f-no { color:var(--tx2); }
      .f-bullet { width:16px; font-size:.72rem; font-weight:700; flex-shrink:0; }
      .f-yes .f-bullet { color:var(--green); } .f-no .f-bullet { color:var(--red); }
      .f-italic { font-style:italic; }
      .pay-btn { display:flex; align-items:center; justify-content:center; gap:.5rem; width:100%; padding:.85rem; border:none; border-radius:12px; font-family:var(--font); font-size:.88rem; font-weight:700; cursor:pointer; transition:all .2s; }
      .pay-btn:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 8px 28px rgba(0,0,0,.4); filter:brightness(1.1); }
      .pay-btn:disabled { opacity:.65; cursor:not-allowed; }

      .trust-row { display:flex; justify-content:center; align-items:center; gap:.75rem; color:var(--tx2); font-size:.75rem; flex-wrap:wrap; margin-top:1rem; }
      .bull { opacity:.4; }

      .toast { position:fixed; top:60px; left:50%; transform:translateX(-50%); background:rgba(0,255,136,.12); border:1px solid rgba(0,255,136,.3); color:var(--green); padding:.6rem 1.5rem; border-radius:100px; font-size:.85rem; font-weight:600; z-index:200; animation:tin .4s ease; }
      @keyframes tin { from{opacity:0;top:44px} to{opacity:1;top:60px} }

      /* APP SHELL */
      .app-shell { display:grid; grid-template-columns:280px 1fr; flex:1; overflow:hidden; position:relative; z-index:1; }

      /* SIDEBAR */
      .app-side { background:var(--s); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:1rem; gap:.75rem; overflow-y:auto; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }
      .side-section { display:flex; flex-direction:column; gap:.5rem; }
      .side-label { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--tx2); padding-left:2px; }
      .side-tabs { display:flex; flex-direction:column; gap:3px; }
      .side-tab { display:flex; align-items:center; gap:.55rem; padding:.6rem .8rem; border:1px solid transparent; background:transparent; color:var(--tx2); border-radius:9px; font-family:var(--font); font-size:.85rem; font-weight:500; cursor:pointer; transition:all .15s; text-align:left; }
      .side-tab:hover:not(.side-tab-locked) { background:var(--s2); color:var(--tx); }
      .side-tab-on { background:var(--s2); border-color:var(--b2); color:var(--tx); }
      .side-tab-locked { opacity:.5; cursor:not-allowed; }
      .side-tab-icon { font-size:.95rem; width:18px; text-align:center; }
      .lock-chip { margin-left:auto; background:rgba(245,158,11,.1); color:var(--amber); border:1px solid rgba(245,158,11,.25); font-size:.62rem; font-weight:700; padding:1px 6px; border-radius:4px; }

      .ctrl-sel { background:var(--s2); border:1px solid var(--b2); color:var(--tx); padding:.5rem .7rem; border-radius:8px; font-family:var(--font); font-size:.83rem; outline:none; cursor:pointer; width:100%; }
      .ctrl-sel:focus { border-color:var(--green); }
      .ctrl-ta { background:var(--s2); border:1px solid var(--b2); color:var(--tx); padding:.6rem .7rem; border-radius:8px; font-family:var(--font); font-size:.8rem; resize:vertical; outline:none; line-height:1.55; width:100%; }
      .ctrl-ta:focus { border-color:rgba(0,255,136,.3); }
      .ctrl-ta::placeholder { color:var(--tx2); }
      .ctrl-hint { font-size:.66rem; color:var(--tx2); }

      .action-btn { display:flex; align-items:center; justify-content:center; gap:.45rem; padding:.7rem; border:none; border-radius:10px; font-family:var(--font); font-size:.85rem; font-weight:700; cursor:pointer; transition:all .2s; margin-top:.25rem; }
      .action-btn:hover:not(:disabled) { transform:translateY(-1px); }
      .action-btn:disabled { opacity:.6; cursor:not-allowed; }
      .amber-btn { background:var(--amber); color:#050810; }
      .amber-btn:hover:not(:disabled) { box-shadow:0 5px 18px rgba(245,158,11,.35); }
      .blue-btn { background:var(--blue); color:#050810; }
      .blue-btn:hover:not(:disabled) { box-shadow:0 5px 18px rgba(88,166,255,.35); }
      .green-btn { background:var(--green); color:#050810; }
      .green-btn:hover:not(:disabled) { box-shadow:0 5px 18px rgba(0,255,136,.35); }

      .dropzone { border:2px dashed var(--b2); border-radius:12px; padding:1.2rem 1rem; cursor:pointer; transition:all .2s; }
      .dropzone:hover { border-color:var(--blue); background:rgba(88,166,255,.03); }
      .dz-over { border-color:var(--green)!important; background:rgba(0,255,136,.04)!important; }
      .dz-has-file { border-style:solid; border-color:rgba(0,255,136,.25); background:rgba(0,255,136,.03); }
      .dz-empty { display:flex; flex-direction:column; align-items:center; gap:.35rem; text-align:center; }
      .dz-icon { font-size:1.4rem; color:var(--tx2); }
      .dz-text { font-size:.82rem; font-weight:500; }
      .dz-sub { font-size:.7rem; color:var(--tx2); }
      .file-row { display:flex; align-items:center; gap:.6rem; width:100%; }
      .file-nm { font-size:.8rem; font-weight:600; color:var(--green); word-break:break-all; }
      .file-mt { font-size:.7rem; color:var(--tx2); font-family:var(--mono); }
      .file-x { background:none; border:none; color:var(--tx2); cursor:pointer; font-size:.82rem; transition:color .15s; flex-shrink:0; }
      .file-x:hover { color:var(--red); }

      .info-box { background:var(--s2); border:1px solid var(--border); border-radius:9px; padding:.75rem; display:flex; gap:.5rem; font-size:.8rem; color:var(--tx2); line-height:1.5; }
      .info-box strong { color:var(--tx); }
      .mini-file { display:flex; justify-content:space-between; font-size:.75rem; color:var(--tx2); font-family:var(--mono); background:var(--s2); padding:.45rem .7rem; border-radius:7px; }
      .mini-lang { color:var(--green); font-weight:600; }

      .mem-card { margin-top:auto; display:flex; align-items:center; gap:.6rem; background:var(--s2); border:1px solid var(--border); border-radius:10px; padding:.7rem; }
      .mem-glyph { font-size:1rem; }
      .mem-title { font-size:.78rem; font-weight:600; }
      .mem-sub { font-size:.68rem; color:var(--tx2); }

      /* OUTPUT */
      .app-main { display:flex; flex-direction:column; overflow:hidden; }
      .out-panel { display:flex; flex-direction:column; height:100%; }
      .out-top { display:flex; align-items:center; justify-content:space-between; padding:.8rem 1.25rem; border-bottom:1px solid var(--border); background:var(--s); flex-shrink:0; }
      .out-title { display:flex; align-items:center; gap:.5rem; font-size:.88rem; font-weight:600; }
      .title-glyph { font-size:1rem; }
      .title-glyph.amber { color:var(--amber); } .title-glyph.blue { color:var(--blue); } .title-glyph.green { color:var(--green); }
      .lang-badge { background:rgba(245,158,11,.1); color:var(--amber); font-size:.72rem; font-weight:700; padding:2px 8px; border-radius:4px; font-family:var(--mono); }
      .out-acts { display:flex; gap:.5rem; }
      .icon-btn { background:var(--s2); border:1px solid var(--b2); color:var(--tx2); padding:.28rem .75rem; border-radius:6px; font-size:.75rem; font-family:var(--mono); cursor:pointer; transition:all .15s; }
      .icon-btn:hover { color:var(--tx); border-color:var(--green); }
      .out-body { flex:1; overflow-y:auto; padding:1.25rem; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }

      /* States */
      .loading-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:1.25rem; }
      .loading-ring { width:52px; height:52px; border:3px solid var(--border); border-top-color:var(--amber); border-radius:50%; animation:spin .75s linear infinite; }
      @keyframes spin { to{transform:rotate(360deg)} }
      .loading-label { font-size:.85rem; color:var(--tx2); }
      .empty-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:.5rem; text-align:center; }
      .empty-icon { font-size:2.8rem; margin-bottom:.25rem; }
      .empty-text { font-size:.95rem; font-weight:600; color:var(--tx); }
      .empty-sub { font-size:.82rem; color:var(--tx2); }

      .code-pre { background:#080b14; border:1px solid var(--border); border-radius:12px; padding:1.25rem; font-family:var(--mono); font-size:.8rem; line-height:1.75; color:#a8b8cc; overflow-x:auto; white-space:pre-wrap; word-break:break-word; }
      .amber-pre { border-color:rgba(245,158,11,.18); }
      .green-pre { border-color:rgba(0,255,136,.18); color:#b0ffd4; }

      /* Analysis */
      .analysis-cards { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
      .a-card { background:var(--s); border:1px solid var(--border); border-radius:12px; padding:1rem; }
      .a-card-title { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--tx2); margin-bottom:.5rem; }
      .a-card-body { font-size:.8rem; color:var(--tx); line-height:1.65; white-space:pre-wrap; }
      .upgrade-card { grid-column:1/-1; display:flex; align-items:center; gap:1rem; background:rgba(0,255,136,.04); border:1px solid rgba(0,255,136,.2); border-radius:12px; padding:1rem; }
      .upgrade-card strong { font-size:.88rem; display:block; margin-bottom:.2rem; }
      .upgrade-card p { font-size:.78rem; color:var(--tx2); }
      .up-cta { margin-left:auto; background:var(--green); color:#050810; border:none; padding:.5rem 1rem; border-radius:8px; font-weight:700; font-size:.8rem; cursor:pointer; white-space:nowrap; transition:all .2s; }
      .up-cta:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(0,255,136,.3); }
      .err-box { color:var(--red); background:rgba(248,81,73,.06); border:1px solid rgba(248,81,73,.2); padding:1rem; border-radius:10px; font-size:.85rem; }

      .spin { width:13px; height:13px; border:2px solid rgba(0,0,0,.25); border-top-color:#050810; border-radius:50%; animation:spin .6s linear infinite; display:inline-block; }

      @media (max-width:768px) {
        .app-shell { grid-template-columns:1fr; }
        .plans-duo { grid-template-columns:1fr; }
        .analysis-cards { grid-template-columns:1fr; }
      }
    `}</style>
  )
}