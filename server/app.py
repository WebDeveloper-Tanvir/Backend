"""
server/app.py

Errorless AI Backend — Powered by Claude claude-sonnet-4-6
Handles: Generate, Analyze, Optimize, Agentic Loop

Start locally:
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

On Render:
  uvicorn server.app:app --host 0.0.0.0 --port $PORT

Environment variables (set in Render dashboard):
  CLAUDE_API_KEY = sk-ant-api03-...   ← your new key after revoking the old one
  ALLOWED_ORIGINS = https://your-app.vercel.app
"""

import asyncio, os, time, uuid, json, re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ── Config ────────────────────────────────────────────────────────────────────

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-6"
CLAUDE_URL     = "https://api.anthropic.com/v1/messages"

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    os.environ.get("ALLOWED_ORIGINS", "https://your-app.vercel.app"),
    "https://*.vercel.app",
]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    if not CLAUDE_API_KEY:
        print("⚠️  CLAUDE_API_KEY not set. Add it in Render → Environment Variables.")
    else:
        print(f"✅ Claude {CLAUDE_MODEL} ready.")
    yield
    print("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Errorless AI",
    description="Code generation, analysis & optimization via Claude",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Claude helper ─────────────────────────────────────────────────────────────

async def call_claude(
    system: str,
    user:   str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Call Claude API and return the text response."""
    if not CLAUDE_API_KEY:
        raise HTTPException(500, "CLAUDE_API_KEY not configured on server.")

    headers = {
        "x-api-key":         CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":       CLAUDE_MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      system,
        "messages":    [{"role": "user", "content": user}],
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(CLAUDE_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        detail = resp.json().get("error", {}).get("message", resp.text)
        raise HTTPException(502, f"Claude API error: {detail}")

    return resp.json()["content"][0]["text"]


async def stream_claude(
    system: str,
    user:   str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Stream Claude response token by token via SSE."""
    if not CLAUDE_API_KEY:
        yield "[ERROR] CLAUDE_API_KEY not configured."
        return

    headers = {
        "x-api-key":         CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":       CLAUDE_MODEL,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      system,
        "stream":      True,
        "messages":    [{"role": "user", "content": user}],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", CLAUDE_URL, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        if obj.get("type") == "content_block_delta":
                            token = obj["delta"].get("text", "")
                            if token:
                                yield token
                    except Exception:
                        continue


# ── Prompt templates ──────────────────────────────────────────────────────────

def generate_system(language: str) -> str:
    return f"""You are an expert {language} developer for Errorless — an AI coding platform.

RULES:
1. Output ONLY raw code — no markdown fences (no ```), no "here is your code"
2. Add brief inline comments on non-obvious lines
3. Handle edge cases and errors properly
4. Use modern {language} best practices
5. After the code write exactly: ---EXPLANATION---
6. Then write 3-5 bullet points (start each with •) explaining the code"""


ANALYZE_SYSTEM = """You are a senior code reviewer for Errorless DevMode.

Return ONLY valid JSON — no markdown, no backticks, no extra text.

Schema:
{
  "summary": "1-2 sentence description",
  "quality_score": <0-100>,
  "complexity": {
    "time_complexity": "Big-O",
    "space_complexity": "Big-O",
    "cyclomatic": <integer>,
    "readability": <1-10>
  },
  "bugs": ["specific bug with line reference"],
  "security_issues": ["specific security risk"],
  "performance_issues": ["specific bottleneck"],
  "suggestions": ["actionable fix with example"],
  "language": "detected language"
}"""


def optimize_system(language: str) -> str:
    return f"""You are an elite code optimizer for Errorless Premium.

Rewrite the {language} code to be production-grade:
• Fix ALL bugs
• Improve algorithmic efficiency
• Add error handling and input validation
• Add type hints / TypeScript types
• Remove dead code and duplication
• Add docstrings for public functions

OUTPUT FORMAT:
1. Complete rewritten code (raw — NO markdown fences)
2. Then write: ---CHANGES---
3. Bullet list of what you improved"""


# ── Request / Response Schemas ────────────────────────────────────────────────

class GenerateReq(BaseModel):
    prompt:      str   = Field(..., min_length=3)
    language:    str   = Field("python")
    max_tokens:  int   = Field(2048, ge=128, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    plan:        str   = Field("basic")
    context:     str   = Field("")

class AnalyzeReq(BaseModel):
    code:     str = Field(..., min_length=5)
    language: str = Field("python")
    plan:     str = Field("basic")

class OptimizeReq(BaseModel):
    code:             str            = Field(..., min_length=5)
    language:         str            = Field("python")
    prior_analysis:   Optional[dict] = None
    plan:             str            = Field("premium")

class AgentReq(BaseModel):
    prompt:            str = Field(..., min_length=3)
    language:          str = Field("python")
    max_loops:         int = Field(3, ge=1, le=5)
    quality_threshold: int = Field(75, ge=40, le=95)
    plan:              str = Field("premium")
    context:           str = Field("")


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_code(text: str) -> str:
    """Strip markdown fences and AI preamble from output."""
    text = re.sub(r"```[\w]*\n?|```", "", text)
    text = re.sub(
        r"^(here is|here's|below is|certainly|sure)[^:\n]*[:：]?\s*\n",
        "", text, flags=re.IGNORECASE
    )
    return text.strip()


def split_code_explanation(text: str):
    if "---EXPLANATION---" in text:
        parts = text.split("---EXPLANATION---", 1)
        return clean_code(parts[0]), parts[1].strip()
    return clean_code(text), ""


def split_code_changes(text: str):
    if "---CHANGES---" in text:
        parts = text.split("---CHANGES---", 1)
        code  = clean_code(parts[0])
        changes = [
            l.lstrip("-•* ").strip()
            for l in parts[1].splitlines() if l.strip()
        ]
        return code, changes
    return clean_code(text), []


def parse_analysis_json(text: str) -> Optional[dict]:
    try:
        return json.loads(re.sub(r"```json\s*|```", "", text).strip())
    except Exception:
        pass
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "model":   CLAUDE_MODEL,
        "api_key": "configured" if CLAUDE_API_KEY else "MISSING — set CLAUDE_API_KEY",
        "version": "2.0.0",
    }


@app.post("/generate")
async def generate(req: GenerateReq):
    """Generate production-ready code from a natural language prompt."""
    rid = f"gen_{uuid.uuid4().hex[:8]}"
    t0  = time.time()

    context_line = f"\nContext from previous session: {req.context}" if req.context else ""
    user_msg     = f"Generate {req.language} code for:\n\n{req.prompt}{context_line}"

    raw  = await call_claude(
        system=generate_system(req.language) ,
        user=user_msg,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    code, explanation = split_code_explanation(raw)

    return {
        "request_id":  rid,
        "success":     True,
        "code":        code,
        "explanation": explanation,
        "language":    req.language,
        "time":        round(time.time() - t0, 2),
        "model":       CLAUDE_MODEL,
    }


@app.post("/generate/stream")
async def generate_stream(req: GenerateReq):
    """Streaming code generation via Server-Sent Events."""
    context_line = f"\nContext: {req.context}" if req.context else ""
    user_msg     = f"Generate {req.language} code for:\n\n{req.prompt}{context_line}"

    async def sse():
        try:
            async for token in stream_claude(
                system=generate_system(req.language),
                user=user_msg,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    """Deep code analysis — returns quality score, bugs, security issues."""
    rid = f"anl_{uuid.uuid4().hex[:8]}"
    t0  = time.time()

    raw    = await call_claude(
        system=ANALYZE_SYSTEM,
        user=f"Analyze this {req.language} code:\n\n{req.code}",
        max_tokens=2048,
        temperature=0.2,
    )
    result = parse_analysis_json(raw)

    # Fallback if JSON parse fails
    if result is None:
        result = {
            "summary":           f"{req.language} code analyzed.",
            "quality_score":     50,
            "complexity":        {"time_complexity": "?", "cyclomatic": 1, "readability": 5},
            "bugs":              [],
            "security_issues":   [],
            "performance_issues":[],
            "suggestions":       ["Re-run for detailed analysis."],
            "language":          req.language,
        }

    return {
        "request_id": rid,
        "success":    True,
        "time":       round(time.time() - t0, 2),
        "model":      CLAUDE_MODEL,
        **result,
    }


@app.post("/optimize")
async def optimize(req: OptimizeReq):
    """Premium: Rewrite code to production quality using Claude."""
    rid = f"opt_{uuid.uuid4().hex[:8]}"
    t0  = time.time()

    analysis_ctx = json.dumps(req.prior_analysis, indent=2) if req.prior_analysis else "No prior analysis."
    user_msg     = f"Optimize this {req.language} code:\n\n{req.code}\n\nAnalysis context:\n{analysis_ctx}"

    raw  = await call_claude(
        system=optimize_system(req.language),
        user=user_msg,
        max_tokens=4096,
        temperature=0.3,
    )
    optimized_code, changes = split_code_changes(raw)

    return {
        "request_id":      rid,
        "success":         True,
        # page.tsx reads d.optimized — provide both field names
        "optimized":       optimized_code,
        "optimized_code":  optimized_code,
        "changes":         changes,
        "original_lines":  len(req.code.splitlines()),
        "optimized_lines": len(optimized_code.splitlines()),
        "time":            round(time.time() - t0, 2),
        "model":           CLAUDE_MODEL,
    }


@app.post("/agent")
async def run_agent(req: AgentReq):
    """
    🤖 Emergent agentic loop — powered by Claude:
    Generate → Analyze → Optimize → until quality_threshold met.
    Returns best code + full reasoning chain.
    """
    rid     = f"agt_{uuid.uuid4().hex[:8]}"
    t_start = time.time()
    chain   = []
    best    = {"code": "", "score": 0, "loop": 0, "analysis": {}}

    # ── Step 1: Generate ─────────────────────────────────────────────────────
    chain.append({"step": 1, "action": "generate", "status": "running",
                  "thought": f"Generating {req.language} code for: {req.prompt[:60]}…"})

    gen_raw  = await call_claude(
        system=generate_system(req.language),
        user=f"Generate {req.language} code for:\n\n{req.prompt}",
        max_tokens=2048, temperature=0.7,
    )
    gen_code, _ = split_code_explanation(gen_raw)
    current_code = gen_code
    chain[-1].update({"status": "done", "lines": len(current_code.splitlines())})

    # ── Agentic loop ─────────────────────────────────────────────────────────
    for loop in range(1, req.max_loops + 1):

        # Analyze
        chain.append({"step": len(chain)+1, "action": "analyze", "loop": loop,
                      "status": "running", "thought": f"Analyzing quality (loop {loop})…"})

        anl_raw  = await call_claude(
            system=ANALYZE_SYSTEM,
            user=f"Analyze this {req.language} code:\n\n{current_code}",
            max_tokens=1024, temperature=0.2,
        )
        analysis = parse_analysis_json(anl_raw) or {"quality_score": 50, "bugs": []}
        score    = analysis.get("quality_score", 50)
        chain[-1].update({"status": "done", "score": score,
                          "bugs": len(analysis.get("bugs", []))})

        if score > best["score"]:
            best = {"code": current_code, "score": score, "loop": loop, "analysis": analysis}

        # Quality gate
        if score >= req.quality_threshold:
            chain.append({"step": len(chain)+1, "action": "quality_gate", "status": "passed",
                          "thought": f"✅ Score {score} ≥ {req.quality_threshold}. Done!"})
            break

        # Optimize if loops remain
        if loop < req.max_loops:
            chain.append({"step": len(chain)+1, "action": "optimize", "loop": loop,
                          "status": "running",
                          "thought": f"Score {score} < {req.quality_threshold}. Optimizing…"})
            opt_raw      = await call_claude(
                system=optimize_system(req.language),
                user=f"Optimize this {req.language} code:\n\n{current_code}\n\nAnalysis:\n{json.dumps(analysis)}",
                max_tokens=4096, temperature=0.3,
            )
            current_code, changes = split_code_changes(opt_raw)
            chain[-1].update({"status": "done", "top_changes": changes[:3]})
        else:
            chain.append({"step": len(chain)+1, "action": "max_loops", "status": "done",
                          "thought": f"Max loops reached. Best score: {best['score']}/100."})

    return {
        "request_id":      rid,
        "success":         True,
        "final_code":      best["code"] or current_code,
        "final_score":     best["score"],
        "loops_ran":       best["loop"],
        "was_optimized":   best["loop"] > 1,
        "final_analysis":  best["analysis"],
        "initial_code":    gen_code,
        "reasoning_chain": chain,
        "total_time":      round(time.time() - t_start, 2),
        "model":           CLAUDE_MODEL,
    }


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """WebSocket streaming. Send: {prompt, language}. Receive: token chunks."""
    await ws.accept()
    try:
        data = await ws.receive_json()
        async for token in stream_claude(
            system=generate_system(data.get("language", "python")),
            user=f"Generate {data.get('language','python')} code for:\n\n{data.get('prompt','')}",
        ):
            await ws.send_json({"token": token})
        await ws.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")