"""
trainer/train.py

Training loop for ErrorlessLM.

Features:
  • AdamW optimizer with cosine LR schedule + warmup
  • Gradient clipping (prevents exploding gradients)
  • Mixed precision training (FP16/BF16) when GPU available
  • Checkpoint saving + resuming
  • Live loss logging every N steps
  • Validation loss tracking

Run:
  python trainer/train.py --config small --epochs 5
"""

import argparse, math, os, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Allow running from project root ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import ErrorlessLM, small_config, tiny_config, nano_config, base_config, ModelConfig
from data.dataset import CodeDataset, collate_fn, load_code_files, try_load_codesearchnet


CONFIG_MAP = {
    "nano":  nano_config,
    "tiny":  tiny_config,
    "small": small_config,
    "base":  base_config,
}


class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = self._get_device()
        print(f"[Trainer] Device: {self.device}")

        # ── Paths ─────────────────────────────────────────────────────────────
        self.out_dir = Path(args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tok_path   = self.out_dir / "tokenizer.json"
        self.model_path = self.out_dir / "model.pt"
        self.best_path  = self.out_dir / "model_best.pt"

        # ── Tokenizer ─────────────────────────────────────────────────────────
        if self.tok_path.exists():
            print("[Trainer] Loading existing tokenizer…")
            self.tokenizer = BPETokenizer.load(str(self.tok_path))
        else:
            print("[Trainer] Training tokenizer from scratch…")
            self.tokenizer = self._train_tokenizer()

        # ── Dataset ───────────────────────────────────────────────────────────
        full_ds = CodeDataset(self.tokenizer, max_len=args.max_len)
        val_size = max(1, int(len(full_ds) * 0.1))
        trn_size = len(full_ds) - val_size
        self.train_ds, self.val_ds = random_split(full_ds, [trn_size, val_size])

        pad_id = self.tokenizer.pad_id
        self.train_loader = DataLoader(
            self.train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        cfg_fn  = CONFIG_MAP.get(args.config, small_config)
        self.model = ErrorlessLM(cfg_fn(vocab_size=len(self.tokenizer))).to(self.device)

        # Resume from checkpoint if exists
        self.start_epoch = 0
        if self.model_path.exists() and args.resume:
            self._load_checkpoint()

        # ── Optimizer ─────────────────────────────────────────────────────────
        # Separate weight decay: apply to weights, NOT biases/norms
        decay_params     = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and p.ndim >= 2]
        no_decay_params  = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params,    "weight_decay": args.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )

        # Cosine LR schedule with linear warmup
        total_steps   = len(self.train_loader) * args.epochs
        warmup_steps  = int(total_steps * 0.05)
        self.scheduler = self._cosine_schedule(total_steps, warmup_steps)

        # Mixed precision scaler (GPU only)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        print(f"[Trainer] Train: {len(self.train_ds)} | Val: {len(self.val_ds)} samples")
        print(f"[Trainer] Steps/epoch: {len(self.train_loader)} | Total: {total_steps}")

    # ── Tokenizer training ────────────────────────────────────────────────────

    def _train_tokenizer(self) -> BPETokenizer:
        """Build corpus and train BPE tokenizer."""
        corpus = []

        # 1. Synthetic examples (always available)
        from data.dataset import GENERATE_EXAMPLES, ANALYZE_EXAMPLES, OPTIMIZE_EXAMPLES
        for prompt, lang, code in GENERATE_EXAMPLES:
            corpus.extend([prompt, code])
        for code, lang, analysis in ANALYZE_EXAMPLES:
            corpus.extend([code, analysis])
        for bad, lang, good in OPTIMIZE_EXAMPLES:
            corpus.extend([bad, good])

        # 2. User's own code files (if provided)
        if self.args.data_dir and os.path.isdir(self.args.data_dir):
            corpus.extend(load_code_files(self.args.data_dir))

        # 3. CodeSearchNet (if datasets installed)
        corpus.extend(try_load_codesearchnet("python", max_samples=10_000))

        tok = BPETokenizer(vocab_size=self.args.vocab_size)
        tok.train(corpus, verbose=True)
        tok.save(str(self.tok_path))
        return tok

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        best_val_loss = float("inf")
        global_step   = 0

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            t0         = time.time()

            for step, (inp, lbl) in enumerate(self.train_loader):
                inp = inp.to(self.device)
                lbl = lbl.to(self.device)
                # Replace label padding -100 → pad_id for model, keep -100 for loss
                lbl_for_loss = lbl.clone()
                lbl_for_loss[lbl == -100] = self.tokenizer.pad_id

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    _, loss, _ = self.model(inp, labels=lbl_for_loss)

                self.scaler.scale(loss).backward()

                # Gradient clipping — critical for stable transformer training
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                epoch_loss  += loss.item()
                global_step += 1

                if step % self.args.log_every == 0:
                    lr  = self.optimizer.param_groups[0]["lr"]
                    ppl = math.exp(min(loss.item(), 20))
                    print(f"Epoch {epoch+1}/{self.args.epochs} | "
                          f"Step {step}/{len(self.train_loader)} | "
                          f"Loss {loss.item():.4f} | PPL {ppl:.1f} | LR {lr:.2e}")

            # ── Validation ────────────────────────────────────────────────────
            val_loss = self._validate()
            avg_loss = epoch_loss / len(self.train_loader)
            elapsed  = time.time() - t0

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} done | "
                  f"Train loss: {avg_loss:.4f} | Val loss: {val_loss:.4f} | "
                  f"Time: {elapsed:.0f}s")
            print(f"{'='*60}\n")

            # Save checkpoint
            self._save_checkpoint(epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save(str(self.best_path))
                print(f"[Trainer] ✅ New best model saved (val_loss={val_loss:.4f})")

        print(f"\n🎉 Training complete! Best model: {self.best_path}")

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for inp, lbl in self.val_loader:
            inp = inp.to(self.device)
            lbl = lbl.to(self.device)
            lbl[lbl == -100] = self.tokenizer.pad_id
            _, loss, _ = self.model(inp, labels=lbl)
            total_loss += loss.item()
        self.model.train()
        return total_loss / max(len(self.val_loader), 1)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _cosine_schedule(self, total_steps: int, warmup_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save_checkpoint(self, epoch: int):
        self.model.save(str(self.model_path))
        torch.save({"epoch": epoch + 1}, self.out_dir / "training_state.pt")

    def _load_checkpoint(self):
        print(f"[Trainer] Resuming from {self.model_path}")
        ck = torch.load(self.out_dir / "training_state.pt", map_location="cpu")
        self.start_epoch = ck.get("epoch", 0)
        sd = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(sd["state"])
        self.model.to(self.device)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():    return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train ErrorlessLM from scratch")
    p.add_argument("--config",     default="small",   choices=["nano","tiny","small","base"])
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=8,    dest="batch_size")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--max-len",    type=int,   default=512,  dest="max_len")
    p.add_argument("--vocab-size", type=int,   default=8000, dest="vocab_size")
    p.add_argument("--grad-clip",  type=float, default=1.0,  dest="grad_clip")
    p.add_argument("--weight-decay", type=float, default=0.1, dest="weight_decay")
    p.add_argument("--log-every",  type=int,   default=10,   dest="log_every")
    p.add_argument("--output-dir", default="checkpoints",    dest="output_dir")
    p.add_argument("--data-dir",   default=None,             dest="data_dir",
                   help="Directory of .py/.js files for extra training data")
    p.add_argument("--resume",     action="store_true", help="Resume from checkpoint")
    args = p.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
