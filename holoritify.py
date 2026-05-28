"""holoritify.py — convert a standard HF model into a Holorite.

A Holorite is the same model with its input-embedding matrix rearranged
onto the 64x64x64 torus lattice (the bit-slice bijection from the build
brief). Lookup is byte-exact identical to nn.Embedding, but at runtime
only the rings that the current context touches need to be on the GPU —
so on a 4 GB GPU you can run models whose full embedding matrix would
otherwise not fit (~80% of the embedding bytes stay on CPU).

Usage:
    py holoritify.py <hf_model_path_or_id> [--out <Holorite-dir>]

Examples:
    py holoritify.py Qwen/Qwen2.5-0.5B
    py holoritify.py "D:/0000_Raw_LLM Models/unsloth--Qwen3-4B-GGUF"   # if HF format

Writes to D:\\Holorites\\Holorite-<short-name>\\:
    manifest.json          - {model_id, vocab, hidden, dtype, src_path, ...}
    embeddings_torus.pt    - (64, 64, 64, hidden) torus-shaped tensor
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import embedding_to_torus, CELLS, RINGS, NODES, SLOTS

HOLORITES_ROOT = os.path.dirname(os.path.abspath(__file__))

def short_name(model_id: str) -> str:
    """Qwen/Qwen2.5-0.5B -> Qwen2.5-0.5B ; absolute paths -> last segment."""
    base = os.path.basename(model_id.rstrip("/\\"))
    return base.replace("/", "-").replace("--", "-")

def fmt_mb(b: int) -> str: return f"{b / 1_048_576:.1f} MiB"

def holoritify(src: str, out_dir: str | None = None) -> str:
    """Returns the path to the produced Holorite folder."""
    print(f"=== holoritify '{src}' ===")
    name = short_name(src)
    out_dir = out_dir or os.path.join(HOLORITES_ROOT, f"Holorite-{name}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    print("  loading tokenizer + model (CPU, fp16 if possible) ...")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.eval()
    emb = model.get_input_embeddings()
    V, D = emb.weight.shape
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"  vocab={V}, hidden={D}, dtype={emb.weight.dtype}")
    if V > CELLS:
        print(f"  ! vocab {V} > 64**3 = {CELLS}.  Need a wider lattice for this model.")
        print(f"  ! Skipping for now (deferred — needs a different bit layout, e.g. 128*64*64).")
        return ""

    print(f"  reshaping (V,D)=({V},{D}) -> torus ({RINGS},{NODES},{SLOTS},{D}) ...")
    torus = embedding_to_torus(emb.weight.detach().clone(), pad_token_id=pad_id)
    print(f"  torus tensor: {fmt_mb(torus.numel()*torus.element_size())}")

    out_pt = os.path.join(out_dir, "embeddings_torus.pt")
    torch.save({
        "model_id": src,
        "vocab_size": int(V),
        "hidden_dim": int(D),
        "torus_shape": list(torus.shape),
        "dtype": str(emb.weight.dtype).replace("torch.", ""),
        "pad_token_id": int(pad_id) if pad_id is not None else None,
        "torus": torus,
    }, out_pt)
    manifest = {
        "name": name,
        "model_id": src,
        "vocab_size": int(V),
        "hidden_dim": int(D),
        "dtype": str(emb.weight.dtype).replace("torch.", ""),
        "torus_shape": list(torus.shape),
        "embeddings_torus": "embeddings_torus.pt",
        "produced_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {out_pt} ({fmt_mb(os.path.getsize(out_pt))})")
    print(f"  wrote {os.path.join(out_dir, 'manifest.json')}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF id (e.g. Qwen/Qwen2.5-0.5B) or local HF-format dir")
    ap.add_argument("--out", help="Output dir (default: D:\\Holorites\\Holorite-<name>)")
    a = ap.parse_args()
    holoritify(a.src, a.out)

if __name__ == "__main__":
    main()
