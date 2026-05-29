"""extract_identities.py — convert hexagram-data.ts → identities.json for the planner.

Reads the v0.8.84 hexagram-data.ts (TypeScript Record<number,HexagramData>)
and produces a minimal {1..64: [keywords...]} JSON that
`hexagram_ring_planner.py --identities` consumes.

Pulled fields:
    name           → split on whitespace, lowercased
    chineseName    → kept as a single token
    coreMeaning    → bag of significant words (length >= 4, no stopwords)
    judgment       → bag of significant words, sampled (capped to keep
                     the per-hexagram bag short so tokenizer encoding
                     doesn't explode into a long signature vector)

The output is intentionally shallow — the planner just needs enough
tokens per hexagram for the affinity matmul to discriminate. Bigger
bags don't help because rare tokens dominate the cosine score.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

STOPWORDS = {
    "the", "and", "this", "that", "with", "from", "into", "your", "their",
    "have", "been", "they", "them", "when", "where", "what", "which", "while",
    "than", "then", "there", "those", "these", "such", "like", "also", "even",
    "only", "very", "well", "well,", "well.", "must", "without", "would", "could",
    "should", "shall", "will", "does", "doing", "made", "make", "makes", "more",
    "less", "many", "much", "some", "any", "all", "but", "yet", "for", "are",
    "was", "were", "his", "her", "its", "him", "you", "she", "one", "two",
    "now", "not", "yes", "ever", "never", "always", "may", "can", "who", "how",
    "why", "out", "off", "way", "ways", "down", "upon", "into", "onto",
    "above", "below", "before", "after", "again", "still", "thus", "here",
    "hence", "comes", "come", "came", "goes", "gone", "going",
}
WORD_RX = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")

# Match each hexagram block:
#   { number: N,  ... name: "...", ... chineseName: "...", ... judgment: ... coreMeaning: "..." }
# The TS file uses backtick template strings for multi-line judgment/image,
# and ordinary double-quoted strings for name/chineseName/coreMeaning.

HEX_BLOCK_RX = re.compile(
    r"\bnumber\s*:\s*(\d+)\s*,",
    re.DOTALL,
)
NAME_RX = re.compile(r'\bname\s*:\s*"([^"]+)"')
CN_NAME_RX = re.compile(r'\bchineseName\s*:\s*"([^"]+)"')
CORE_RX = re.compile(r'\bcoreMeaning\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
JUDGMENT_RX = re.compile(r"\bjudgment\s*:\s*[`\"]((?:[^`\"\\]|\\.)*)[`\"]", re.DOTALL)


def _wordbag(text: str, cap: int = 20) -> list[str]:
    """Lowercase significant-word bag, dedup-preserve-order, capped."""
    if not text: return []
    seen: set[str] = set()
    out: list[str] = []
    for m in WORD_RX.finditer(text):
        w = m.group(0).lower()
        if w in STOPWORDS: continue
        if w in seen: continue
        seen.add(w)
        out.append(w)
        if len(out) >= cap: break
    return out


def extract(ts_path: str) -> dict:
    text = Path(ts_path).read_text(encoding="utf-8", errors="replace")
    # find the start of each hexagram block by "number: N,"
    starts = [(int(m.group(1)), m.start()) for m in HEX_BLOCK_RX.finditer(text)]
    if not starts:
        raise SystemExit("No hexagram blocks found — check the source file")
    print(f"[extract] found {len(starts)} candidate hexagram blocks")
    # block boundaries: each block ends at the next block's start (or EOF)
    blocks: dict[int, str] = {}
    for i, (num, off) in enumerate(starts):
        end = starts[i+1][1] if i+1 < len(starts) else len(text)
        if 1 <= num <= 64 and num not in blocks:
            blocks[num] = text[off:end]
    print(f"[extract] kept {len(blocks)} valid (1..64) blocks")

    identities: dict[int, list[str]] = {}
    for num in range(1, 65):
        chunk = blocks.get(num, "")
        bag: list[str] = []
        # name → lowercase tokens
        m = NAME_RX.search(chunk)
        if m:
            for w in WORD_RX.findall(m.group(1)):
                w = w.lower()
                if w not in STOPWORDS and w not in bag:
                    bag.append(w)
        # Chinese pinyin name → single token
        m = CN_NAME_RX.search(chunk)
        if m and m.group(1).strip():
            cn = m.group(1).strip().lower()
            if cn not in bag: bag.append(cn)
        # coreMeaning bag
        m = CORE_RX.search(chunk)
        if m:
            for w in _wordbag(m.group(1), cap=8):
                if w not in bag: bag.append(w)
        # judgment bag
        m = JUDGMENT_RX.search(chunk)
        if m:
            for w in _wordbag(m.group(1), cap=8):
                if w not in bag: bag.append(w)
        if not bag:
            bag = [f"hexagram_{num}"]
        identities[num] = bag[:24]    # cap total at 24 tokens / hex
    return identities


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  py extract_identities.py <hexagram-data.ts> [out.json]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "identities.json"
    ids = extract(src)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    print(f"[extract] wrote {out} ({sum(len(v) for v in ids.values())} total tokens)")
    # spot-check
    for h in (1, 2, 11, 23, 49, 64):
        print(f"  {h:2d}: {ids[h][:8]}…")
