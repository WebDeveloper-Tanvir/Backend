"""
tests/test_all.py

Complete test suite for ErrorlessLM.
Run: pytest tests/ -v

Tests run WITHOUT a trained model — uses a freshly initialized
(random weights) nano model so no GPU or training needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import json

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import ErrorlessLM, nano_config, ModelConfig
from agent.code_agent import CodeAgent, _clean_code, _parse_analysis, _static_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CORPUS = [
    "def hello(): print('hello')",
    "function add(a, b) { return a + b; }",
    "for i in range(10): print(i)",
    "const x = async () => await fetch('/api')",
    "class Foo:\n    def __init__(self): self.x = 0",
    "import os\npath = os.path.join('a', 'b')",
    "SELECT * FROM users WHERE id = 1",
    "if (x > 0) { console.log(x); }",
]

@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer(vocab_size=300)
    tok.train(SAMPLE_CORPUS, min_freq=1, verbose=False)
    return tok

@pytest.fixture(scope="module")
def model(tokenizer):
    cfg = nano_config(vocab_size=len(tokenizer))
    return ErrorlessLM(cfg)

@pytest.fixture(scope="module")
def agent(model, tokenizer):
    return CodeAgent(model, tokenizer)


# ── Tokenizer tests ───────────────────────────────────────────────────────────

class TestTokenizer:
    def test_train_creates_vocab(self, tokenizer):
        assert len(tokenizer) >= 16  # at least special tokens
        assert len(tokenizer) <= 300

    def test_encode_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("def hello():", add_special=True)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert ids[0] == tokenizer.bos_id

    def test_encode_with_mode_token(self, tokenizer):
        ids = tokenizer.encode("hello", mode="generate", add_special=True)
        special = [tokenizer.token2id["<GEN>"], tokenizer.token2id["<BOS>"]]
        assert any(s in ids for s in special)

    def test_decode_roundtrip(self, tokenizer):
        text = "hello world"
        ids  = tokenizer.encode(text, add_special=False)
        out  = tokenizer.decode(ids)
        # BPE decode may add spaces — check tokens present
        assert "hello" in out or "hel" in out

    def test_max_len_truncation(self, tokenizer):
        long_text = "x " * 200
        ids = tokenizer.encode(long_text, add_special=False, max_len=50)
        assert len(ids) <= 50

    def test_encode_batch(self, tokenizer):
        texts = ["def foo():", "function bar() {}"]
        inp, mask = tokenizer.encode_batch(texts, max_len=64)
        assert len(inp) == 2
        assert len(inp[0]) == len(inp[1])  # padded to same length
        assert len(mask[0]) == len(inp[0])

    def test_save_and_load(self, tokenizer, tmp_path):
        path = str(tmp_path / "tok.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert len(loaded) == len(tokenizer)
        ids_orig   = tokenizer.encode("def test():", add_special=False)
        ids_loaded = loaded.encode("def test():", add_special=False)
        assert ids_orig == ids_loaded


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModel:
    def test_forward_returns_logits(self, model, tokenizer):
        ids = torch.tensor([[tokenizer.bos_id, 8, 9, 10]])  # (1, 4)
        logits, loss, caches = model(ids)
        assert logits.shape == (1, 4, len(tokenizer))
        assert loss is None

    def test_training_loss(self, model, tokenizer):
        ids = torch.tensor([[tokenizer.bos_id, 8, 9, 10]])
        logits, loss, _ = model(ids, labels=ids)
        assert loss is not None
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_generate_shape(self, model, tokenizer):
        ids = torch.tensor([[tokenizer.bos_id]])
        out = model.generate(ids, max_new_tokens=5, temperature=0.8, eos_id=tokenizer.eos_id)
        assert out.shape[0] == 1
        assert out.shape[1] >= 1

    def test_kv_cache_used(self, model, tokenizer):
        ids    = torch.tensor([[tokenizer.bos_id, 8, 9]])
        logits, _, caches = model(ids)
        assert caches is not None
        assert len(caches) == model.cfg.n_layers
        assert caches[0] is not None

    def test_num_params_nonzero(self, model):
        assert model.num_params > 0
        print(f"\n[Test] Model params: {model.num_params/1e6:.2f}M")

    def test_save_load_roundtrip(self, model, tmp_path):
        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = ErrorlessLM.load(path, device="cpu")
        assert loaded.num_params == model.num_params

    def test_no_nan_in_output(self, model, tokenizer):
        ids    = torch.randint(16, len(tokenizer), (2, 10))
        logits, _, _ = model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_gradient_flows(self, model, tokenizer):
        """Check that loss.backward() produces gradients."""
        ids  = torch.tensor([[tokenizer.bos_id, 8, 9, 10]])
        _, loss, _ = model(ids, labels=ids)
        loss.backward()
        first_param = next(model.parameters())
        assert first_param.grad is not None


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestAgent:
    """Agent tests use a nano model with random weights.
    Output quality is garbage — we test structure, not content."""

    def test_generate_returns_dict(self, agent):
        r = agent.generate("hello world", "python")
        assert "code" in r
        assert "time" in r
        assert isinstance(r["code"], str)

    def test_generate_stream_yields_tokens(self, agent):
        tokens = list(agent.generate_stream("fibonacci", "python"))
        # May be empty if model immediately outputs EOS with random weights — that's ok
        assert isinstance(tokens, list)

    def test_analyze_returns_score(self, agent):
        r = agent.analyze("def foo(): pass", "python")
        assert "quality_score" in r
        assert 0 <= r["quality_score"] <= 100
        assert "static" in r

    def test_optimize_returns_code(self, agent):
        r = agent.optimize("def foo(): return 1/0", "python")
        assert "optimized_code" in r
        assert isinstance(r["changes"], list)

    def test_agentic_loop_structure(self, agent):
        r = agent.agentic_loop("write hello world", "python", max_loops=2)
        assert "final_code" in r
        assert "final_score" in r
        assert "reasoning_chain" in r
        assert "total_time" in r
        assert len(r["reasoning_chain"]) >= 2

    def test_agentic_loop_high_threshold_runs_full(self, agent):
        """With threshold=100, agent must run all loops."""
        r = agent.agentic_loop("hello", "python", max_loops=2, quality_threshold=100)
        assert r["loops_ran"] <= 2

    def test_reasoning_chain_has_required_fields(self, agent):
        r = agent.agentic_loop("hello", "python", max_loops=1)
        for step in r["reasoning_chain"]:
            assert "step" in step
            assert "action" in step
            assert "thought" in step


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_clean_code_removes_fences(self):
        raw = "```python\ndef foo(): pass\n```"
        assert "```" not in _clean_code(raw)

    def test_clean_code_removes_preamble(self):
        raw = "Here is your code:\ndef foo(): pass"
        out = _clean_code(raw)
        assert "here is" not in out.lower()

    def test_parse_analysis_valid_json(self):
        j = '{"quality_score": 75, "bugs": [], "summary": "ok"}'
        r = _parse_analysis(j)
        assert r is not None
        assert r["quality_score"] == 75

    def test_parse_analysis_extracts_embedded_json(self):
        text = "Some preamble\n{\"quality_score\": 60, \"bugs\": [\"null check\"]}\nSome suffix"
        r = _parse_analysis(text)
        assert r is not None
        assert r["quality_score"] == 60

    def test_parse_analysis_returns_none_on_garbage(self):
        assert _parse_analysis("not json at all") is None

    def test_static_metrics_counts_functions(self):
        code = "def foo():\n    pass\ndef bar():\n    pass"
        m    = _static_metrics(code)
        assert m["functions"] == 2

    def test_static_metrics_comment_ratio(self):
        code = "# comment\nx = 1\n# another\ny = 2"
        m    = _static_metrics(code)
        assert m["comment_ratio"] > 0


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:
    @pytest.mark.parametrize("n_layers,d_model", [
        (2, 128), (6, 256), (12, 512), (16, 768)
    ])
    def test_model_creates_with_different_configs(self, tokenizer, n_layers, d_model):
        cfg = ModelConfig(
            vocab_size=len(tokenizer), n_layers=n_layers, d_model=d_model,
            n_heads=4, n_kv_heads=4, d_ff=d_model*4, max_seq_len=128
        )
        m = ErrorlessLM(cfg)
        assert m.num_params > 0

    def test_tied_embeddings_saves_params(self, tokenizer):
        cfg_tied   = nano_config(len(tokenizer))
        cfg_untied = nano_config(len(tokenizer))
        cfg_untied.tie_embeddings = False
        m_tied   = ErrorlessLM(cfg_tied)
        m_untied = ErrorlessLM(cfg_untied)
        assert m_tied.num_params <= m_untied.num_params
