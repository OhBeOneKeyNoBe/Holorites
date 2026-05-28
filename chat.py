"""chat.py — minimal REPL to chat with a Holorite.

Loads a Holorite folder (model id from its manifest.json), swaps the
model's embedding layer with the RingPagedEmbedding from torus_lattice.py,
and runs a streaming generation loop on whatever you type.

While generating it prints the VRAM saved by the paging.

Usage:
    py chat.py "D:/Holorites/Holorite-Qwen2.5-0.5B"
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import RingPagedEmbedding, RINGS, CELLS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def fmt_mb(b): return f"{b / 1_048_576:.1f} MiB"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("holorite", help="Path to Holorite-* folder")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--max-new", type=int, default=120)
    a = ap.parse_args()

    manifest_path = os.path.join(a.holorite, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"no manifest.json in {a.holorite}")
    with open(manifest_path) as f:
        man = json.load(f)
    print(f"loading Holorite {man['name']} (orig: {man['model_id']})")
    print(f"  vocab={man['vocab_size']}, hidden={man['hidden_dim']}, dtype={man['dtype']}")

    tok = AutoTokenizer.from_pretrained(man["model_id"])
    print("  loading model weights (CPU first) ...")
    model = AutoModelForCausalLM.from_pretrained(
        man["model_id"], torch_dtype=DTYPE, low_cpu_mem_usage=True,
    )
    # Snapshot original VRAM cost of the embedding
    emb_orig = model.get_input_embeddings()
    full_emb_bytes = emb_orig.weight.numel() * emb_orig.weight.element_size()

    # Load the torus tensor + build the paged embedding
    torus_blob = torch.load(os.path.join(a.holorite, "embeddings_torus.pt"), weights_only=False)
    torus = torus_blob["torus"]
    V, D = torus_blob["vocab_size"], torus_blob["hidden_dim"]
    if (V, D) != tuple(emb_orig.weight.shape):
        print(f"  warning: torus has shape ({V},{D}) vs model emb {tuple(emb_orig.weight.shape)}")
    flat = torus.view(CELLS, D)[:V]   # the original V rows, but laid out in torus order
    pad_id = man.get("pad_token_id") or tok.eos_token_id
    paged = RingPagedEmbedding(flat, max_cached_rings=RINGS,
                               cpu_device="cpu", compute_device=a.device,
                               pad_token_id=pad_id)
    # Swap in
    model.set_input_embeddings(paged)
    # Move the REST of the model to GPU (the heavy weights), keep embedding master on CPU
    for n, p in model.named_parameters():
        if "embed_tokens" in n: continue
        p.data = p.data.to(a.device)
    for n, b in model.named_buffers():
        if "embed_tokens" in n: continue
        b.data = b.data.to(a.device)
    model.eval()

    if a.device == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM after load:  used = {fmt_mb(total - free)}  free = {fmt_mb(free)} / {fmt_mb(total)}")
    print(f"  full embedding bytes (orig nn.Embedding): {fmt_mb(full_emb_bytes)}")

    print("\nType your prompt (blank to quit). Ctrl-C to abort.\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user: break
        history.append({"role": "user", "content": user})
        try:
            ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = tok(user, return_tensors="pt").input_ids
        ids = ids.to(a.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                ids,
                max_new_tokens=a.max_new,
                do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        history.append({"role": "assistant", "content": text})

        # stats
        stats = paged.last_stats
        if stats:
            ring_bytes = (CELLS // RINGS) * D * (torus.element_size())
            paged_on_gpu = stats.used_rings * ring_bytes
            saved = full_emb_bytes - paged_on_gpu
            print(f"\nzioniel> {text}")
            print(f"  [paging: {stats.used_rings}/{stats.total_rings} rings used | "
                  f"on-GPU emb: {fmt_mb(paged_on_gpu)} vs full {fmt_mb(full_emb_bytes)} "
                  f"-> saved {fmt_mb(saved)} ({saved/full_emb_bytes:.0%}) | "
                  f"{new.numel()} new tok in {dt:.1f}s = {new.numel()/dt:.1f} tok/s]")
        else:
            print(f"\nzioniel> {text}\n  [{new.numel()} tok in {dt:.1f}s]")
        print()

if __name__ == "__main__":
    main()
