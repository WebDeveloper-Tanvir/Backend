"""
agent/code_agent.py

The Emergent Agent — uses ErrorlessLM for all 3 modes.

Single-shot:
  generate(prompt, language) → code
  analyze(code, language)    → structured dict
  optimize(code, language)   → better code

Agentic loop (emergent behavior):
  1. Generate code from prompt
  2. Analyze it → quality score (0-100)
  3. If score < threshold → optimize → re-analyze
  4. Loop up to max_loops
  5. Return best result + full reasoning chain

The model itself handles everything — no external APIs.
"""

import json, re, time
from typing import AsyncGenerator, Dict, Generator, List, Optional
from pathlib import Path

import torch

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import ErrorlessLM


# ── Prompt templates ──────────────────────────────────────────────────────────
# These teach the model what we want by prefix — matching training format

GEN_PREFIX  = "<GEN><{LANG}>{prompt}<SEP>"
ANL_PREFIX  = "<ANL><{LANG}>{code}<SEP>"
OPT_PREFIX  = "<OPT><{LANG}>{code}<SEP>"

LANG_TOKEN  = {
    "python":"PYTHON","javascript":"JS","typescript":"TS",
    "java":"JAVA","cpp":"CPP","rust":"RUST","go":"GO","csharp":"CSHARP",
}


class CodeAgent:
    """
    Agentic inference engine powered by a custom-trained ErrorlessLM.

    Load:
        agent = CodeAgent.from_checkpoint("checkpoints/model_best.pt")

    Use:
        code   = agent.generate("write a binary search", "python")
        report = agent.analyze(code, "python")
        better = agent.optimize(code, "python")
        result = agent.agentic_loop("build a REST endpoint", "python")
    """

    def __init__(self, model: ErrorlessLM, tokenizer: BPETokenizer):
        self.model     = model
        self.tok       = tokenizer
        self.device    = next(model.parameters()).device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, model_path: str, tok_path: str = None) -> "CodeAgent":
        """Load agent from trained checkpoint."""
        device = (
            torch.device("cuda")  if torch.cuda.is_available()  else
            torch.device("mps")   if torch.backends.mps.is_available() else
            torch.device("cpu")
        )

        # Auto-find tokenizer if not specified
        if tok_path is None:
            tok_path = str(Path(model_path).parent / "tokenizer.json")
        if not Path(tok_path).exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tok_path}. "
                f"Run training first: python trainer/train.py"
            )

        print(f"[Agent] Loading model from {model_path} on {device}")
        model     = ErrorlessLM.load(model_path, device=str(device))
        tokenizer = BPETokenizer.load(tok_path)
        return cls(model, tokenizer)

    # ── Core inference ────────────────────────────────────────────────────────

    def _infer(
        self,
        prefix:         str,
        max_new_tokens: int   = 300,
        temperature:    float = 0.7,
        top_k:          int   = 40,
        top_p:          float = 0.9,
    ) -> str:
        """Run the LM on a prefix, return decoded completion."""
        ids = self.tok.encode(prefix, add_special=False, max_len=400)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.model.generate(
                ids_t,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.tok.eos_id,
            )

        # Only decode the newly generated tokens
        new_ids = out[0][len(ids):].tolist()
        return self.tok.decode(new_ids, skip_special=True).strip()

    def _infer_stream(
        self,
        prefix: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Token-by-token streaming generator."""
        ids   = self.tok.encode(prefix, add_special=False, max_len=400)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        gen   = ids_t.clone()

        self.model.eval()
        caches = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                cur = gen if caches is None else gen[:, -1:]
                logits, _, caches = self.model(cur, kv_caches=caches)
                logits = logits[:, -1, :] / max(temperature, 1e-5)

                # Top-K sampling
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, -1:]] = float("-inf")
                next_tok = torch.multinomial(logits.softmax(-1), 1)

                if next_tok.item() == self.tok.eos_id:
                    break

                gen = torch.cat([gen, next_tok], dim=1)
                token_text = self.tok.decode([next_tok.item()], skip_special=True)
                if token_text:
                    yield token_text

    # ── Generate mode ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:         str,
        language:       str   = "python",
        max_tokens:     int   = 512,
        temperature:    float = 0.7,
        request_id:     str   = "",
        _attempt:       int   = 1,
    ) -> Dict:
        """Generate production-ready code from a natural language prompt."""
        lang_tok = LANG_TOKEN.get(language.lower(), "PYTHON")
        prefix   = GEN_PREFIX.format(LANG=lang_tok, prompt=prompt)

        t0   = time.time()
        code = self._infer(prefix, max_new_tokens=max_tokens, temperature=temperature)
        code = _clean_code(code)

        # Self-correction: if output too short, retry with lower temperature
        if len(code.strip()) < 20 and _attempt < 3:
            print(f"[Agent] Output too short → retrying (attempt {_attempt+1})")
            return self.generate(prompt, language, max_tokens,
                                 max(temperature - 0.15, 0.3), request_id, _attempt + 1)

        return {
            "code":     code,
            "language": language,
            "time":     round(time.time() - t0, 3),
            "attempts": _attempt,
        }

    def generate_stream(
        self, prompt: str, language: str = "python", temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Streaming token-by-token generation."""
        lang_tok = LANG_TOKEN.get(language.lower(), "PYTHON")
        prefix   = GEN_PREFIX.format(LANG=lang_tok, prompt=prompt)
        yield from self._infer_stream(prefix, temperature=temperature)

    # ── Analyze mode ──────────────────────────────────────────────────────────

    def analyze(
        self,
        code:       str,
        language:   str = "python",
        request_id: str = "",
        _attempt:   int = 1,
    ) -> Dict:
        """Analyze code and return a structured quality report."""
        lang_tok = LANG_TOKEN.get(language.lower(), "PYTHON")
        prefix   = ANL_PREFIX.format(LANG=lang_tok, code=code[:800])

        t0  = time.time()
        raw = self._infer(prefix, max_new_tokens=400, temperature=0.2)

        result = _parse_analysis(raw)

        if result is None and _attempt < 3:
            print(f"[Agent] Analysis parse failed → retrying (attempt {_attempt+1})")
            return self.analyze(code, language, request_id, _attempt + 1)

        if result is None:
            result = _static_fallback(code, language)

        # Enrich with static metrics
        result["static"] = _static_metrics(code)
        result["time"]   = round(time.time() - t0, 3)
        return result

    # ── Optimize mode ─────────────────────────────────────────────────────────

    def optimize(
        self,
        code:       str,
        language:   str  = "python",
        request_id: str  = "",
    ) -> Dict:
        """Rewrite code to be faster, safer, and more readable."""
        lang_tok = LANG_TOKEN.get(language.lower(), "PYTHON")
        prefix   = OPT_PREFIX.format(LANG=lang_tok, code=code[:800])

        t0  = time.time()
        raw = self._infer(prefix, max_new_tokens=512, temperature=0.3)
        optimized = _clean_code(raw)

        # Parse out ---CHANGES--- section if present
        changes = []
        if "---CHANGES---" in optimized:
            parts     = optimized.split("---CHANGES---", 1)
            optimized = parts[0].strip()
            changes   = [l.lstrip("-• ").strip() for l in parts[1].splitlines() if l.strip()]

        return {
            "optimized_code":  optimized,
            "changes":         changes,
            "original_lines":  len(code.splitlines()),
            "optimized_lines": len(optimized.splitlines()),
            "time": round(time.time() - t0, 3),
        }

    # ── 🤖 Emergent Agentic Loop ──────────────────────────────────────────────

    def agentic_loop(
        self,
        prompt:            str,
        language:          str = "python",
        max_loops:         int = 3,
        quality_threshold: int = 70,
        request_id:        str = "",
    ) -> Dict:
        """
        EMERGENT AGENT — self-improving pipeline:

        Loop:
          1. generate(prompt)  → initial code
          2. analyze(code)     → quality score
          3. score >= threshold → DONE ✅
          4. score <  threshold → optimize(code) → back to 2
          (repeat up to max_loops)

        Returns best result + full reasoning chain showing every decision.
        """
        t_start = time.time()
        chain   = []
        best    = {"code": "", "score": 0, "loop": 0}

        print(f"[Agent] 🤖 Agentic loop | loops={max_loops} threshold={quality_threshold}")

        # Step 1: Initial generation
        chain.append({"step": 1, "action": "generate",
                       "thought": f"Generating {language} code for: {prompt[:60]}…"})
        gen_result   = self.generate(prompt, language, request_id=request_id)
        current_code = gen_result["code"]
        chain[-1]["lines"]  = len(current_code.splitlines())
        chain[-1]["status"] = "done"

        # Agentic loop
        for loop in range(1, max_loops + 1):
            print(f"[Agent] Loop {loop}/{max_loops}")

            # Step 2: Analyze
            chain.append({"step": len(chain)+1, "action": "analyze", "loop": loop,
                           "thought": f"Analyzing code quality…"})
            analysis = self.analyze(current_code, language, request_id=request_id)
            score    = analysis.get("quality_score", 50)
            chain[-1].update({"status": "done", "score": score,
                              "bugs": len(analysis.get("bugs", []))})

            # Track best
            if score > best["score"]:
                best = {"code": current_code, "score": score, "loop": loop, "analysis": analysis}

            print(f"[Agent]   Score: {score}/100")

            # Quality gate
            if score >= quality_threshold:
                chain.append({"step": len(chain)+1, "action": "done",
                               "thought": f"✅ Score {score} ≥ {quality_threshold}. Stopping.",
                               "score": score})
                break

            # Step 3: Optimize
            if loop < max_loops:
                chain.append({"step": len(chain)+1, "action": "optimize", "loop": loop,
                               "thought": f"Score {score} < {quality_threshold}. Optimizing…"})
                opt = self.optimize(current_code, language, request_id=request_id)
                current_code = opt["optimized_code"]
                chain[-1].update({"status": "done", "changes": opt["changes"][:3]})
            else:
                chain.append({"step": len(chain)+1, "action": "max_loops",
                               "thought": f"Max loops reached. Best score: {best['score']}"})

        total = round(time.time() - t_start, 2)
        print(f"[Agent] ✅ Complete in {total}s | final score: {best['score']}/100")

        return {
            "final_code":      best["code"] or current_code,
            "final_score":     best["score"],
            "loops_ran":       best["loop"],
            "was_optimized":   best["loop"] > 1,
            "final_analysis":  best.get("analysis", {}),
            "initial_code":    gen_result["code"],
            "reasoning_chain": chain,
            "total_time":      total,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_code(text: str) -> str:
    """Remove markdown fences and leading prose from LM output."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\w]*\n?|```", "", text)
    text = re.sub(r"^(here is|here's|below is|sure|certainly)[^:\n]*[:：]?\s*\n",
                  "", text, flags=re.IGNORECASE)
    return text.strip()


def _parse_analysis(text: str) -> Optional[Dict]:
    """Try to extract JSON from analysis output."""
    # Direct parse
    try:
        clean = re.sub(r"```json\s*|```", "", text).strip()
        return json.loads(clean)
    except Exception:
        pass
    # Find JSON block
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _static_fallback(code: str, language: str) -> Dict:
    """Static analysis fallback when LM JSON parse fails."""
    lines    = code.splitlines()
    branches = len(re.findall(r"\bif\b|\bfor\b|\bwhile\b|\belif\b|\bcatch\b", code))
    return {
        "summary":           f"{language} code ({len(lines)} lines)",
        "quality_score":     50,
        "complexity":        {"time_complexity": "?", "cyclomatic": branches},
        "bugs":              [],
        "security_issues":   [],
        "performance_issues":[],
        "suggestions":       ["Re-run for full analysis"],
        "language":          language,
    }


def _static_metrics(code: str) -> Dict:
    """Fast static code metrics (no LM needed)."""
    lines    = code.splitlines()
    non_emp  = [l for l in lines if l.strip()]
    comments = [l for l in lines if l.strip().startswith(("#", "//", "/*", "*"))]
    funcs    = re.findall(r"\bdef \w+|\bfunction \w+|\bfunc \w+", code)
    return {
        "total_lines":    len(lines),
        "code_lines":     len(non_emp),
        "comment_lines":  len(comments),
        "functions":      len(funcs),
        "comment_ratio":  round(len(comments) / max(len(non_emp), 1), 2),
    }
