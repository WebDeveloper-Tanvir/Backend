"""
data/dataset.py

Training data pipeline for ErrorlessLM.

Data sources:
  1. CodeSearchNet (HuggingFace datasets) — real code from GitHub
  2. Synthetic pairs — (prompt, code) pairs we generate ourselves
  3. Your own code files — can point it at any directory

Each sample is formatted as:
  <BOS> <GEN> <PYTHON> [natural language prompt] <SEP> [code] <EOS>
  <BOS> <ANL> <PYTHON> [code] <SEP> [analysis JSON] <EOS>
  <BOS> <OPT> <PYTHON> [bad code] <SEP> [good code] <EOS>

This teaches the model all 3 modes in one training run.
"""

import os, json, random
from typing import List, Dict, Optional, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ── Synthetic training examples (no downloads needed) ────────────────────────

GENERATE_EXAMPLES = [
    # (prompt, language, code)
    ("write a function to reverse a string", "python",
     "def reverse_string(s: str) -> str:\n    return s[::-1]"),
    ("create a fibonacci function", "python",
     "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
    ("binary search in a sorted list", "python",
     "def binary_search(arr: list, target) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
    ("check if a string is a palindrome", "python",
     "def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
    ("flatten a nested list", "python",
     "def flatten(lst: list) -> list:\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"),
    ("merge sort algorithm", "python",
     "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    return result + left[i:] + right[j:]"),
    ("count word frequency in a string", "python",
     "from collections import Counter\ndef word_freq(text: str) -> dict:\n    return dict(Counter(text.lower().split()))"),
    ("find all prime numbers up to n", "python",
     "def sieve(n: int) -> list:\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n + 1, i):\n                is_prime[j] = False\n    return [x for x in range(n + 1) if is_prime[x]]"),
    ("async fetch url with aiohttp", "python",
     "import aiohttp\nasync def fetch(url: str) -> str:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            response.raise_for_status()\n            return await response.text()"),
    ("create a simple linked list", "python",
     "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, val):\n        node = Node(val)\n        if not self.head:\n            self.head = node\n            return\n        cur = self.head\n        while cur.next:\n            cur = cur.next\n        cur.next = node"),
    ("debounce function in javascript", "javascript",
     "function debounce(fn, delay) {\n  let timer;\n  return function(...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), delay);\n  };\n}"),
    ("deep clone object in javascript", "javascript",
     "function deepClone(obj) {\n  if (obj === null || typeof obj !== 'object') return obj;\n  if (Array.isArray(obj)) return obj.map(deepClone);\n  return Object.fromEntries(\n    Object.entries(obj).map(([k, v]) => [k, deepClone(v)])\n  );\n}"),
    ("throttle function javascript", "javascript",
     "function throttle(fn, limit) {\n  let lastCall = 0;\n  return function(...args) {\n    const now = Date.now();\n    if (now - lastCall >= limit) {\n      lastCall = now;\n      return fn.apply(this, args);\n    }\n  };\n}"),
    ("flatten nested array javascript", "javascript",
     "function flatten(arr) {\n  return arr.reduce((flat, item) =>\n    flat.concat(Array.isArray(item) ? flatten(item) : item), []);\n}"),
    ("group array by property javascript", "javascript",
     "function groupBy(arr, key) {\n  return arr.reduce((groups, item) => {\n    const val = item[key];\n    groups[val] = groups[val] ?? [];\n    groups[val].push(item);\n    return groups;\n  }, {});\n}"),
]

ANALYZE_EXAMPLES = [
    # (code, language, analysis JSON string)
    (
        "def div(a, b):\n    return a / b",
        "python",
        '{"summary":"Division function without zero-check","quality_score":35,"bugs":["No check for b==0 causes ZeroDivisionError"],"security_issues":[],"performance_issues":[],"suggestions":["Add if b == 0: raise ValueError"],"complexity":{"time_complexity":"O(1)","cyclomatic":1}}'
    ),
    (
        "def search(lst, val):\n    for i in range(len(lst)):\n        if lst[i] == val:\n            return i\n    return -1",
        "python",
        '{"summary":"Linear search implementation","quality_score":65,"bugs":[],"security_issues":[],"performance_issues":["O(n) — use dict or binary_search for sorted data"],"suggestions":["Use enumerate() instead of range(len())","Consider sorted array + bisect for O(log n)"],"complexity":{"time_complexity":"O(n)","cyclomatic":2}}'
    ),
    (
        "password = '12345'\ndef check(p):\n    return p == password",
        "python",
        '{"summary":"Hardcoded password comparison","quality_score":10,"bugs":["Hardcoded credentials in source code","Timing attack vulnerable — use hmac.compare_digest"],"security_issues":["Never hardcode secrets — use env vars","Plain text password comparison"],"performance_issues":[],"suggestions":["Use os.environ for secrets","Use hmac.compare_digest to prevent timing attacks"],"complexity":{"time_complexity":"O(1)","cyclomatic":1}}'
    ),
]

OPTIMIZE_EXAMPLES = [
    # (bad_code, language, good_code)
    (
        "def div(a, b):\n    return a / b",
        "python",
        "def div(a: float, b: float) -> float:\n    \"\"\"Divide a by b. Raises ValueError if b is zero.\"\"\"\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b"
    ),
    (
        "def get_evens(lst):\n    result = []\n    for x in lst:\n        if x % 2 == 0:\n            result.append(x)\n    return result",
        "python",
        "def get_evens(lst: list) -> list:\n    \"\"\"Return even numbers from lst.\"\"\"\n    return [x for x in lst if x % 2 == 0]"
    ),
    (
        "function fetchData(url) {\n  var xhr = new XMLHttpRequest();\n  xhr.open('GET', url);\n  xhr.send();\n  return xhr.response;\n}",
        "javascript",
        "async function fetchData(url) {\n  const response = await fetch(url);\n  if (!response.ok) throw new Error(`HTTP ${response.status}`);\n  return response.json();\n}"
    ),
]


class CodeDataset(Dataset):
    """
    PyTorch Dataset that returns tokenized (input_ids, labels) pairs.

    Each sample is one of:
      GEN mode: [<BOS><GEN><LANG> prompt <SEP> code <EOS>]
      ANL mode: [<BOS><ANL><LANG> code <SEP> analysis <EOS>]
      OPT mode: [<BOS><OPT><LANG> bad_code <SEP> good_code <EOS>]
    """

    def __init__(self, tokenizer, max_len: int = 512, augment: bool = True):
        from tokenizer.bpe_tokenizer import LANG_MAP, MODE_MAP
        self.tok     = tokenizer
        self.max_len = max_len
        self.samples: List[str] = []
        self._build_samples(augment)

    def _build_samples(self, augment: bool):
        """Build raw text samples from all examples."""
        sep = "<SEP>"

        # GEN samples
        for prompt, lang, code in GENERATE_EXAMPLES:
            # Forward: prompt → code
            self.samples.append(f"<GEN><{lang.upper()}>{prompt}{sep}{code}")
            if augment:
                # Augment: code → "what does this do?" (reverse understanding)
                self.samples.append(f"<ANL><{lang.upper()}>{code}{sep}{{\"summary\":\"{prompt}\"}}")

        # ANL samples
        for code, lang, analysis in ANALYZE_EXAMPLES:
            self.samples.append(f"<ANL><{lang.upper()}>{code}{sep}{analysis}")

        # OPT samples
        for bad, lang, good in OPTIMIZE_EXAMPLES:
            self.samples.append(f"<OPT><{lang.upper()}>{bad}{sep}{good}")

        print(f"[Dataset] {len(self.samples)} training samples built")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tok.encode(self.samples[idx], add_special=True, max_len=self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)
        return ids, ids.clone()  # (input_ids, labels)


def collate_fn(batch, pad_id: int = 0):
    """Pad batch to same length."""
    inputs, labels = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    inp_pad = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in inputs
    ])
    lbl_pad = torch.stack([
        F.pad(x, (0, max_len - x.size(0)), value=-100) for x in labels  # -100 = ignore in CE loss
    ])
    return inp_pad, lbl_pad

import torch.nn.functional as F


def load_code_files(directory: str, extensions=(".py", ".js", ".ts", ".java", ".cpp")) -> List[str]:
    """Load your own code files for training. Point at any directory."""
    texts = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                try:
                    text = Path(os.path.join(root, f)).read_text(encoding="utf-8", errors="ignore")
                    if len(text) > 50:  # skip empty files
                        texts.append(text)
                except Exception:
                    continue
    print(f"[Dataset] Loaded {len(texts)} code files from {directory}")
    return texts


def try_load_codesearchnet(language="python", split="train", max_samples=50_000) -> List[str]:
    """
    Try to load CodeSearchNet from HuggingFace datasets.
    Falls back gracefully if not installed.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("code_search_net", language, split=split, trust_remote_code=True)
        texts = [row["whole_func_string"] for row in ds.select(range(min(max_samples, len(ds))))]
        print(f"[Dataset] Loaded {len(texts)} samples from CodeSearchNet ({language})")
        return texts
    except Exception as e:
        print(f"[Dataset] CodeSearchNet unavailable ({e}) — using synthetic data only")
        return []
