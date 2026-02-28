"""
server/app.py

FastAPI server that exposes ErrorlessLM via HTTP.

Start:
  python server/app.py

Or (production):
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

NOTE: Use workers=1 — the model is loaded once and shared across requests.
"""

import asyncio, time, uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.code_agent import CodeAgent


# ── Global agent (loaded once on startup) ─────────────────────────────────────
_agent: Optional[CodeAgent] = None
MODEL_PATH = Path("checkpoints/model_best.pt")
TOK_PATH   = Path("checkpoints/tokenizer.json")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _agent
    print("🚀 Loading ErrorlessLM…")
    if MODEL_PATH.exists():
        _agent = CodeAgent.from_checkpoint(str(MODEL_PATH), str(TOK_PATH))
        print("✅ Model loaded. Server ready.")
    else:
        print(f"⚠️  No model found at {MODEL_PATH}.")
        print(f"   Run training first:  python trainer/train.py --config small")
        print(f"   Server will respond with 503 until a model is trained.")
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title="Errorless AI — Custom Code LLM",
    description="100% custom-trained transformer. No external APIs.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-app.vercel.app",   # ← replace with your actual Vercel URL
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def _require_agent():
    if _agent is None:
        raise HTTPException(
            503,
            detail="Model not loaded. Train first: python trainer/train.py --config small"
        )
    return _agent


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateReq(BaseModel):
    prompt:      str   = Field(..., min_length=3)
    language:    str   = Field("python")
    max_tokens:  int   = Field(300, ge=32, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=1.5)
    plan:        str   = Field("basic")

class AnalyzeReq(BaseModel):
    code:     str = Field(..., min_length=5)
    language: str = Field("python")
    plan:     str = Field("basic")

class OptimizeReq(BaseModel):
    code:     str = Field(..., min_length=5)
    language: str = Field("python")
    plan:     str = Field("premium")

class AgentReq(BaseModel):
    prompt:            str = Field(..., min_length=3)
    language:          str = Field("python")
    max_loops:         int = Field(3, ge=1, le=5)
    quality_threshold: int = Field(70, ge=30, le=95)
    plan:              str = Field("premium")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok" if _agent else "no_model",
        "model_loaded": _agent is not None,
        "model_path":   str(MODEL_PATH),
        "version":      "1.0.0",
    }


@app.post("/generate")
async def generate(req: GenerateReq):
    """Generate code from a natural language prompt."""
    agent = _require_agent()
    rid   = f"gen_{uuid.uuid4().hex[:8]}"
    loop  = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None, lambda: agent.generate(
            req.prompt, req.language, req.max_tokens, req.temperature, rid
        )
    )
    return {"request_id": rid, "success": True, **result}


@app.post("/generate/stream")
async def generate_stream(req: GenerateReq):
    """
    Streaming code generation — tokens arrive in real-time.
    Returns Server-Sent Events.
    """
    agent = _require_agent()

    async def sse():
        try:
            loop = asyncio.get_event_loop()
            # Run sync generator in thread, yield tokens via queue
            queue: asyncio.Queue = asyncio.Queue()

            def _run():
                for token in agent.generate_stream(req.prompt, req.language, req.temperature):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

            import threading
            threading.Thread(target=_run, daemon=True).start()

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    """Analyze code — returns quality score, bugs, security issues, suggestions."""
    agent  = _require_agent()
    rid    = f"anl_{uuid.uuid4().hex[:8]}"
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: agent.analyze(req.code, req.language, rid)
    )
    return {"request_id": rid, "success": True, **result}


@app.post("/optimize")
async def optimize(req: OptimizeReq):
    """Premium: Rewrite code to production quality."""
    agent  = _require_agent()
    rid    = f"opt_{uuid.uuid4().hex[:8]}"
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: agent.optimize(req.code, req.language, rid)
    )
    return {"request_id": rid, "success": True, **result}


@app.post("/agent")
async def run_agent(req: AgentReq):
    """
    🤖 Emergent agentic loop:
    Generate → Analyze → Optimize → until quality_threshold met.
    Returns best code + full reasoning chain.
    """
    agent  = _require_agent()
    rid    = f"agt_{uuid.uuid4().hex[:8]}"
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: agent.agentic_loop(
            req.prompt, req.language, req.max_loops, req.quality_threshold, rid
        )
    )
    return {"request_id": rid, "success": True, **result}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    WebSocket streaming.
    Send: {"prompt": "...", "language": "python"}
    Receive: token strings, then {"done": true}
    """
    await ws.accept()
    try:
        data  = await ws.receive_json()
        agent = _require_agent()
        queue: asyncio.Queue = asyncio.Queue()
        loop  = asyncio.get_event_loop()

        def _run():
            for token in agent.generate_stream(
                data.get("prompt",""), data.get("language","python")
            ):
                asyncio.run_coroutine_threadsafe(queue.put({"token": token}), loop)
            asyncio.run_coroutine_threadsafe(queue.put({"done": True}), loop)

        import threading
        threading.Thread(target=_run, daemon=True).start()

        while True:
            msg = await queue.get()
            await ws.send_json(msg)
            if msg.get("done"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await ws.send_json({"error": str(e)})
        except: pass


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
