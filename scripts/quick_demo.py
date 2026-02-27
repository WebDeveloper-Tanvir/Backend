"""
scripts/quick_demo.py

Quick demo — train a nano model in ~2 minutes on CPU, then run the agent.
No GPU needed. Output won't be amazing (nano model, few epochs) but
shows the full pipeline working end-to-end.

Run: python scripts/quick_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import ErrorlessLM, nano_config
from data.dataset import GENERATE_EXAMPLES, ANALYZE_EXAMPLES, OPTIMIZE_EXAMPLES, CodeDataset, collate_fn
from agent.code_agent import CodeAgent
from torch.utils.data import DataLoader


def train_quick(epochs=30, batch_size=4):
    print("\n" + "="*60)
    print("  ErrorlessLM — Quick Demo (Nano Model)")
    print("="*60)

    # ── 1. Train tokenizer ────────────────────────────────────────────────────
    print("\n📚 Step 1: Training tokenizer…")
    corpus = []
    for p, l, c in GENERATE_EXAMPLES: corpus.extend([p, c])
    for c, l, a in ANALYZE_EXAMPLES:  corpus.extend([c, a])
    for b, l, g in OPTIMIZE_EXAMPLES: corpus.extend([b, g])

    tok = BPETokenizer(vocab_size=1000)
    tok.train(corpus, min_freq=1, verbose=False)
    print(f"   Vocab size: {len(tok)}")

    # ── 2. Build dataset ──────────────────────────────────────────────────────
    print("\n📦 Step 2: Building dataset…")
    ds = CodeDataset(tok, max_len=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tok.pad_id))

    # ── 3. Create model ───────────────────────────────────────────────────────
    print("\n🧠 Step 3: Creating nano model…")
    cfg   = nano_config(vocab_size=len(tok))
    model = ErrorlessLM(cfg)
    device = torch.device("cpu")
    model.to(device)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n🏋️  Step 4: Training for {epochs} epochs…")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inp, lbl in loader:
            inp, lbl = inp.to(device), lbl.to(device)
            lbl[lbl == -100] = tok.pad_id
            optimizer.zero_grad()
            _, loss, _ = model(inp, labels=lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

    return model, tok


def demo_agent(model, tok):
    print("\n🤖 Step 5: Running the Emergent Agent…")
    print("="*60)

    agent = CodeAgent(model, tok)

    # ── Generate ─────────────────────────────────────────────────────────────
    print("\n[1] GENERATE: 'write a fibonacci function'")
    result = agent.generate("write a fibonacci function", "python")
    print(f"  Code:\n{result['code']}")
    print(f"  Time: {result['time']}s")

    # ── Analyze ──────────────────────────────────────────────────────────────
    print("\n[2] ANALYZE: division by zero bug")
    bug_code = "def div(a, b):\n    return a / b"
    report   = agent.analyze(bug_code, "python")
    print(f"  Quality score: {report.get('quality_score', '?')}/100")
    print(f"  Bugs: {report.get('bugs', [])}")
    print(f"  Suggestions: {report.get('suggestions', [])}")

    # ── Optimize ─────────────────────────────────────────────────────────────
    print("\n[3] OPTIMIZE: same buggy code")
    opt = agent.optimize(bug_code, "python")
    print(f"  Optimized:\n{opt['optimized_code']}")
    print(f"  Changes: {opt['changes']}")

    # ── Agentic loop ─────────────────────────────────────────────────────────
    print("\n[4] EMERGENT AGENT: full loop")
    print("    Prompt: 'check if a number is prime'")
    final = agent.agentic_loop(
        "check if a number is prime",
        language="python",
        max_loops=2,
        quality_threshold=65,
    )
    print(f"\n  Final code:\n{final['final_code']}")
    print(f"\n  Final score: {final['final_score']}/100")
    print(f"  Loops ran:   {final['loops_ran']}")
    print(f"  Optimized?:  {final['was_optimized']}")
    print(f"  Total time:  {final['total_time']}s")
    print("\n  Reasoning chain:")
    for step in final["reasoning_chain"]:
        print(f"    Step {step['step']}: [{step['action'].upper()}] {step['thought']}")

    print("\n✅ Demo complete!")
    print("   Next steps:")
    print("   1. Train longer:  python trainer/train.py --config small --epochs 50")
    print("   2. Start server:  python server/app.py")
    print("   3. Run tests:     pytest tests/ -v")


if __name__ == "__main__":
    model, tok = train_quick(epochs=30)
    demo_agent(model, tok)
