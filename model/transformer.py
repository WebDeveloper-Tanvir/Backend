"""
model/transformer.py

ErrorlessLM — Transformer Language Model built from scratch.

Architecture (modern code-LLM design):
  Embeddings
  → N × [RMSNorm → Grouped Query Attention (RoPE) → RMSNorm → SwiGLU FFN]
  → RMSNorm → LM Head

Size presets:
  nano   ~1M   params  → unit tests / CPU debugging
  tiny   ~15M  params  → fast experiments
  small  ~85M  params  → recommended starting point
  base   ~200M params  → best quality (needs 4GB+ VRAM)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:     int   = 8000
    n_layers:       int   = 12
    d_model:        int   = 512
    n_heads:        int   = 8
    n_kv_heads:     int   = 4       # GQA: fewer KV heads than Q heads
    d_ff:           int   = 2048    # FFN hidden dim (≈ 4 × d_model)
    max_seq_len:    int   = 2048
    dropout:        float = 0.1
    rope_theta:     float = 10000.0
    tie_embeddings: bool  = True    # share input/output embedding weights
    pad_token_id:   int   = 0

    @property
    def head_dim(self): return self.d_model // self.n_heads


def nano_config(vs=8000):
    return ModelConfig(vs, n_layers=2,  d_model=128, n_heads=4,  n_kv_heads=4, d_ff=512,  max_seq_len=512)

def tiny_config(vs=8000):
    return ModelConfig(vs, n_layers=6,  d_model=256, n_heads=8,  n_kv_heads=4, d_ff=1024, max_seq_len=1024)

def small_config(vs=8000):
    return ModelConfig(vs, n_layers=12, d_model=512, n_heads=8,  n_kv_heads=4, d_ff=2048, max_seq_len=2048)

def base_config(vs=8000):
    return ModelConfig(vs, n_layers=16, d_model=768, n_heads=12, n_kv_heads=4, d_ff=3072, max_seq_len=2048)


# ── Building blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm — faster than LayerNorm (no mean subtract)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    Encodes relative positions into Q and K via rotation in complex space.
    Better extrapolation to long sequences vs absolute embeddings.
    """
    def __init__(self, dim: int, max_seq: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        pos  = torch.arange(max_seq, dtype=inv_freq.dtype)
        freqs = torch.outer(pos, inv_freq)
        emb  = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    SwiGLU(x) = Swish(W1·x) ⊙ (W2·x) → W3
    Used in LLaMA, PaLM, Mistral. Outperforms ReLU/GeLU FFNs.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Groups of Q heads share K/V heads → smaller KV cache, faster inference.
    n_kv_heads=n_heads → standard MHA; n_kv_heads=1 → MQA.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads    = cfg.n_heads
        self.n_kv       = cfg.n_kv_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.hd         = cfg.head_dim
        self.scale      = self.hd ** -0.5

        self.q   = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k   = nn.Linear(cfg.d_model, cfg.n_kv   * self.hd, bias=False)
        self.v   = nn.Linear(cfg.d_model, cfg.n_kv   * self.hd, bias=False)
        self.out = nn.Linear(cfg.n_heads  * self.hd, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.hd, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv,   self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv,   self.hd).transpose(1, 2)

        cos, sin = self.rope(T)
        q, k     = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_cache = (k, v)

        # Expand KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Flash attention when available (PyTorch 2.0+), else manual
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.drop.p if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            T_k    = k.shape[2]
            mask   = torch.ones(T, T_k, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            out    = self.drop(scores.float().softmax(-1).to(x.dtype)) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.hd)
        return self.out(out), new_cache


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Norm→Attn→Add, Norm→FFN→Add."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.attn  = GroupedQueryAttention(cfg)
        self.ffn   = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache


# ── Main Model ────────────────────────────────────────────────────────────────

class ErrorlessLM(nn.Module):
    """
    ErrorlessLM — full transformer LM for code generation, analysis, optimization.

    Training:
        model = ErrorlessLM(small_config(vocab_size=8000))
        logits, loss = model(input_ids, labels=input_ids)
        loss.backward()

    Inference:
        out = model.generate(input_ids, max_new_tokens=256, temperature=0.8)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.drop   = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight  # weight tying saves params + helps training

        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 trick for stable training)
        for n, p in self.named_parameters():
            if n.endswith(("out.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ErrorlessLM] {n_params/1e6:.1f}M params | "
              f"{cfg.n_layers}L {cfg.d_model}d {cfg.n_heads}H")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, input_ids, kv_caches=None, labels=None):
        B, T = input_ids.shape
        x    = self.drop(self.embed(input_ids))

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, nc = layer(x, cache)
            new_caches.append(nc)

        logits = self.head(self.norm(x))  # (B, T, vocab)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
        return logits, loss, new_caches

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int   = 256,
        temperature:        float = 0.8,
        top_k:              int   = 50,
        top_p:              float = 0.95,
        repetition_penalty: float = 1.1,
        eos_id:             int   = 3,
    ) -> torch.Tensor:
        """
        Autoregressive generation with:
          • Temperature scaling
          • Top-K + Top-P (nucleus) sampling
          • Repetition penalty
          • KV caching for O(1) per-step computation
        """
        self.eval()
        B, _   = input_ids.shape
        gen    = input_ids.clone()
        caches = None

        for step in range(max_new_tokens):
            # First step: full forward; subsequent: only last token (KV cache)
            cur = gen if caches is None else gen[:, -1:]
            logits, _, caches = self.forward(cur, kv_caches=caches)
            logits = logits[:, -1, :]  # (B, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok in gen[b].unique():
                        logits[b, tok] = (
                            logits[b, tok] / repetition_penalty
                            if logits[b, tok] > 0
                            else logits[b, tok] * repetition_penalty
                        )

            # Temperature + sampling
            if temperature > 0:
                logits = logits / max(temperature, 1e-5)
                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")
                # Top-P (nucleus)
                if top_p < 1.0:
                    sorted_l, sorted_i = torch.sort(logits, descending=True)
                    cum = sorted_l.softmax(-1).cumsum(-1)
                    remove = (cum - sorted_l.softmax(-1)) > top_p
                    sorted_l[remove] = float("-inf")
                    logits.scatter_(1, sorted_i, sorted_l)
                next_tok = torch.multinomial(logits.softmax(-1), 1)
            else:
                next_tok = logits.argmax(-1, keepdim=True)

            gen = torch.cat([gen, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break

        return gen

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state": self.state_dict(),
            "config": {k: getattr(self.cfg, k) for k in [
                "vocab_size","n_layers","d_model","n_heads","n_kv_heads",
                "d_ff","max_seq_len","dropout","rope_theta","pad_token_id"
            ]},
        }, path)
        print(f"[ErrorlessLM] Saved → {path}")

    @classmethod
    def load(cls, path: str, device="cpu") -> "ErrorlessLM":
        ck = torch.load(path, map_location=device, weights_only=True)
        m  = cls(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state"])
        return m.to(device)

    @property
    def num_params(self): return sum(p.numel() for p in self.parameters())
