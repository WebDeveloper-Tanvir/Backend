"""
tokenizer/bpe_tokenizer.py

Custom Byte-Pair Encoding (BPE) tokenizer built specifically for code.

How BPE works:
  1. Start with individual characters as vocab
  2. Find the most frequent adjacent pair
  3. Merge them → new token
  4. Repeat until vocab_size reached

Special tokens:
  <PAD> <UNK> <BOS> <EOS> <SEP>
  <GEN> <ANL> <OPT>          ← mode tokens (generate/analyze/optimize)
  <PYTHON> <JS> <TS> ...     ← language tokens
"""

import json, os, re
from collections import Counter
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS = {
    "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<SEP>": 4,
    "<GEN>": 5, "<ANL>": 6, "<OPT>": 7,
    "<PYTHON>": 8, "<JS>": 9, "<TS>": 10, "<JAVA>": 11,
    "<CPP>": 12, "<RUST>": 13, "<GO>": 14, "<CSHARP>": 15,
}

LANG_MAP = {
    "python": "<PYTHON>", "javascript": "<JS>", "typescript": "<TS>",
    "java": "<JAVA>", "cpp": "<CPP>", "rust": "<RUST>",
    "go": "<GO>", "csharp": "<CSHARP>",
}

MODE_MAP = {"generate": "<GEN>", "analyze": "<ANL>", "optimize": "<OPT>"}

# Code-aware pre-tokenizer: splits on identifiers, numbers, operators, strings
_PRE_TOK = re.compile(
    r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\''  # strings
    r'|[a-zA-Z_]\w*'        # identifiers
    r'|\d+\.?\d*'           # numbers
    r'|[+\-*/=<>!&|^~%@#.:,;?\\]'  # operators
    r'|[\(\)\[\]\{\}]'      # brackets
    r'|\s+'                 # whitespace
)


class BPETokenizer:
    """
    Custom BPE tokenizer optimised for source code.

    Train:
        tok = BPETokenizer(vocab_size=8000)
        tok.train(list_of_code_strings)
        tok.save("tokenizer/vocab.json")

    Use:
        tok = BPETokenizer.load("tokenizer/vocab.json")
        ids = tok.encode("def hello():", mode="generate", language="python")
        txt = tok.decode(ids)
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size  = vocab_size
        self.token2id:   Dict[str, int]        = dict(SPECIAL_TOKENS)
        self.id2token:   Dict[int, str]        = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.merges:     List[Tuple[str, str]] = []
        self._trained    = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, texts: List[str], min_freq: int = 2, verbose: bool = True):
        print(f"[BPE] Training on {len(texts)} docs | target vocab: {self.vocab_size}")

        # Count word frequencies from pre-tokenised corpus
        word_freq: Counter = Counter()
        for text in texts:
            for tok in self._pre_tokenize(text):
                if tok.strip():
                    word_freq[tok] += 1

        # Represent words as tuple of chars + end-of-word marker
        vocab: Dict[tuple, int] = {}
        char_set = set()
        for word, freq in word_freq.items():
            chars = tuple(list(word) + ["</w>"])
            vocab[chars] = freq
            char_set.update(chars)

        # Seed token2id with individual characters
        for ch in sorted(char_set):
            if ch not in self.token2id:
                idx = len(self.token2id)
                self.token2id[ch] = idx
                self.id2token[idx] = ch

        target = self.vocab_size - len(self.token2id)
        done   = 0

        while done < target:
            # Count all adjacent pairs weighted by frequency
            pairs: Counter = Counter()
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += freq

            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < min_freq:
                break

            # Merge best pair across whole vocab
            merged = best[0] + best[1]
            new_vocab = {}
            for word, freq in vocab.items():
                new_word, i = [], 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == best[0] and word[i+1] == best[1]:
                        new_word.append(merged); i += 2
                    else:
                        new_word.append(word[i]); i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

            self.merges.append(best)
            if merged not in self.token2id:
                idx = len(self.token2id)
                self.token2id[merged] = idx
                self.id2token[idx]    = merged

            done += 1
            if verbose and done % 500 == 0:
                print(f"[BPE]  {done}/{target} merges | vocab: {len(self.token2id)}")

        self._trained = True
        print(f"[BPE] ✅ Done. Final vocab size: {len(self.token2id)}")

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def _pre_tokenize(self, text: str) -> List[str]:
        return [m.group() for m in _PRE_TOK.finditer(text)]

    def _bpe_word(self, word: str) -> List[int]:
        chars = list(word) + ["</w>"]
        for a, b in self.merges:
            merged, i = [], 0
            while i < len(chars):
                if i < len(chars)-1 and chars[i] == a and chars[i+1] == b:
                    merged.append(a+b); i += 2
                else:
                    merged.append(chars[i]); i += 1
            chars = merged
        return [self.token2id.get(c, self.token2id["<UNK>"]) for c in chars]

    def encode(self, text: str, mode: str = None, language: str = None,
               add_special: bool = True, max_len: int = None) -> List[int]:
        ids = []
        if add_special:
            ids.append(self.token2id["<BOS>"])
        if mode and mode in MODE_MAP:
            ids.append(self.token2id[MODE_MAP[mode]])
        if language:
            lt = LANG_MAP.get(language.lower())
            if lt and lt in self.token2id:
                ids.append(self.token2id[lt])
        for word in self._pre_tokenize(text):
            if word.strip():
                ids.extend(self._bpe_word(word))
        if add_special:
            ids.append(self.token2id["<EOS>"])
        return ids[:max_len] if max_len else ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            t = self.id2token.get(i, "<UNK>")
            if skip_special and t in SPECIAL_TOKENS:
                continue
            toks.append(t)
        return "".join(toks).replace("</w>", " ").strip()

    def encode_batch(self, texts: List[str], max_len: int = 512):
        encoded = [self.encode(t, max_len=max_len) for t in texts]
        max_l   = max(len(e) for e in encoded)
        pad_id  = self.token2id["<PAD>"]
        input_ids    = [e + [pad_id] * (max_l - len(e)) for e in encoded]
        attn_mask    = [[1]*len(e) + [0]*(max_l - len(e)) for e in encoded]
        return input_ids, attn_mask

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "token2id": self.token2id,
                       "merges": self.merges}, f, indent=2)
        print(f"[BPE] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            d = json.load(f)
        t = cls(vocab_size=d["vocab_size"])
        t.token2id  = {k: int(v) for k, v in d["token2id"].items()}
        t.id2token  = {int(v): k for k, v in d["token2id"].items()}
        t.merges    = [tuple(m) for m in d["merges"]]
        t._trained  = True
        return t

    def __len__(self): return len(self.token2id)
    @property
    def pad_id(self): return self.token2id["<PAD>"]
    @property
    def eos_id(self): return self.token2id["<EOS>"]
    @property
    def bos_id(self): return self.token2id["<BOS>"]
