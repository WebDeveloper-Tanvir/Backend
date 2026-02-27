# ErrorlessLM — Custom Code LLM from Scratch

A transformer language model built entirely from scratch in PyTorch.
**Zero external APIs. Zero pre-trained models. 100% yours.**

---

## Architecture

```
Input IDs
   ↓
Embedding Layer  (vocab_size × d_model)
   ↓
Dropout
   ↓
N × Transformer Block:
   ├─ RMSNorm
   ├─ Grouped Query Attention (RoPE)   ← faster KV cache
   │   └─ Rotary Positional Embedding  ← better position encoding
   ├─ Residual connection
   ├─ RMSNorm
   ├─ SwiGLU FFN                       ← better than ReLU/GeLU
   └─ Residual connection
   ↓
RMSNorm
   ↓
LM Head → logits (vocab_size)
```

### Model Sizes

| Name   | Layers | Dim | Params  | VRAM   | Use case              |
|--------|--------|-----|---------|--------|-----------------------|
| nano   | 2      | 128 | ~1M     | 512MB  | Testing / debugging   |
| tiny   | 6      | 256 | ~15M    | 1GB    | Fast experiments      |
| small  | 12     | 512 | ~85M    | 2GB    | **Recommended start** |
| base   | 16     | 768 | ~200M   | 4GB+   | Best quality          |

---

## Quick Start (5 minutes)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Quick demo (trains nano model in ~2 min on CPU, then runs agent)
python scripts/quick_demo.py

# 3. Full training (small model, GPU recommended)
python trainer/train.py --config small --epochs 50

# 4. Start server (after training)
python server/app.py
```

---

## Training

### Basic
```bash
python trainer/train.py --config small --epochs 50
```

### With your own code files
```bash
# Point it at your codebase — it learns from your style!
python trainer/train.py \
  --config small \
  --epochs 100 \
  --data-dir /path/to/your/code \
  --batch-size 16 \
  --lr 3e-4
```

### With CodeSearchNet (needs `pip install datasets`)
The training script automatically downloads CodeSearchNet if `datasets` is installed.
This gives you 50,000+ real Python functions from GitHub.

### Resume interrupted training
```bash
python trainer/train.py --config small --epochs 100 --resume
```

### Training output
```
checkpoints/
  tokenizer.json     ← trained BPE tokenizer (8000 tokens)
  model.pt           ← latest checkpoint
  model_best.pt      ← best validation loss checkpoint
  training_state.pt  ← epoch counter for resuming
```

---

## The Emergent Agent

After training, the agent chains the 3 modes autonomously:

```
python scripts/quick_demo.py

[4] EMERGENT AGENT: full loop
    Prompt: 'check if a number is prime'

    Step 1: [GENERATE] Generating python code for: check if a number is prime…
    Step 2: [ANALYZE]  Analyzing code quality…  → score: 58/100
    Step 3: [OPTIMIZE] Score 58 < 70. Optimizing…
    Step 4: [ANALYZE]  Analyzing code quality…  → score: 81/100
    Step 5: [DONE]     ✅ Score 81 ≥ 70. Stopping.

    Final score: 81/100
    Loops ran:   2
    Optimized?:  True
    Total time:  3.2s
```

---

## API (after `python server/app.py`)

### Generate
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a binary search", "language": "python"}'
```

### Analyze
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def div(a,b): return a/b", "language": "python"}'
```

### Agent (emergent loop)
```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "REST API with auth", "language": "python", "max_loops": 3}'
```

### Streaming
```javascript
// Frontend (EventSource)
const source = new EventSource('/generate/stream?...')
source.onmessage = (e) => {
  if (e.data === '[DONE]') source.close()
  else appendToken(e.data)
}
```

---

## Connect to Next.js DevMode

```bash
# .env.local
ERRORLESS_LM_URL=http://localhost:8000
```

```typescript
// app/api/devmode/route.ts
const res = await fetch(`${process.env.ERRORLESS_LM_URL}/agent`, {
  method: 'POST',
  body: JSON.stringify({ prompt, language, max_loops: 3, plan: 'premium' })
})
```

---

## Training Tips

| Situation             | Recommendation                              |
|-----------------------|---------------------------------------------|
| No GPU                | `--config nano` or `--config tiny`          |
| 4GB VRAM              | `--config small --batch-size 8`             |
| 8GB+ VRAM             | `--config base --batch-size 16`             |
| Slow CPU              | `--config nano --epochs 100` (overnight)    |
| Want best results     | Add your own code with `--data-dir`         |
| Overfitting           | Reduce epochs, add dropout=0.15             |
| Loss not decreasing   | Lower `--lr 1e-4`, increase `--batch-size`  |

---

## File Structure

```
errorless-llm/
├── tokenizer/
│   └── bpe_tokenizer.py    ← Custom BPE tokenizer
├── model/
│   └── transformer.py      ← Full transformer (RoPE, GQA, SwiGLU)
├── data/
│   └── dataset.py          ← Training data builder
├── trainer/
│   └── train.py            ← Training loop (AdamW, cosine LR)
├── agent/
│   └── code_agent.py       ← Emergent agentic loop
├── server/
│   └── app.py              ← FastAPI server
├── scripts/
│   └── quick_demo.py       ← Run in 2 min, no GPU needed
├── tests/
│   └── test_all.py         ← 25+ tests, run without GPU
├── requirements.txt
└── README.md
```

---

## How It Learns All 3 Modes

Training data has 3 prefixes, one per mode:

```
GEN samples:  <GEN><PYTHON>write a binary search<SEP>def binary_search(arr, t):…
ANL samples:  <ANL><PYTHON>def div(a,b): return a/b<SEP>{"quality_score":35,"bugs":["ZeroDivisionError"]}
OPT samples:  <OPT><PYTHON>def div(a,b): return a/b<SEP>def div(a:float,b:float)->float:…
```

One model, three behaviors — controlled entirely by the prefix token.

---

## Run Tests
```bash
pytest tests/ -v
# All 25 tests pass with random-weight nano model (no training needed)
```
