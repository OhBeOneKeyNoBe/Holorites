"""expert_nan_scan.py — find which expert(s) at a given layer produce NaN.

Layer L's moe_out NaN with prefill points to ONE expert (or a few) whose
dequantized fp16 weights contain NaN. This script iterates over all 128
experts at the layer, dequants each via the streamer's _dequant_expert_slab
path, and reports which ones have NaN in gate / up / down projections.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vram_budget import apply_expandable_segments
apply_expandable_segments()

import torch
from moe_streamer import MoEAssetTree, ExpertStreamer, CacheTiers


def scan(gguf_path: str, layer: int):
    print(f"[expert-nan-scan] gguf:  {gguf_path}")
    print(f"[expert-nan-scan] layer: {layer}")
    tree = MoEAssetTree(gguf_path)
    print(f"[expert-nan-scan] n_routed: {tree.n_routed}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = ExpertStreamer(tree, layer=layer,
                       tiers=CacheTiers(hot=4, warm=8),
                       compute_device=device)
    bad = []
    for eid in range(tree.n_routed):
        try:
            asset = s.assets[eid]
            d = s._admit_to_gpu(asset)
            nans = {}
            for key, val in d.items():
                n_nan = int(torch.isnan(val).sum().item())
                n_inf = int(torch.isinf(val).sum().item())
                if n_nan or n_inf:
                    nans[key] = (n_nan, n_inf,
                                 float(val.float()[torch.isfinite(val.float())].abs().max().item())
                                 if torch.isfinite(val.float()).any() else float('nan'))
            if nans:
                bad.append((eid, nans))
                print(f"  e{eid:>3d}: BAD  {nans}")
            else:
                # only print every 16th good expert to keep output manageable
                if eid % 16 == 0:
                    print(f"  e{eid:>3d}: ok")
            # drop reference so GPU memory can be reclaimed
            del d
        except Exception as e:
            bad.append((eid, f"exception: {e}"))
            print(f"  e{eid:>3d}: EXC  {e}")
        # periodic empty_cache to keep working set small
        if eid % 16 == 15 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[expert-nan-scan] {len(bad)} / {tree.n_routed} experts have NaN/inf")
    for eid, info in bad:
        print(f"  e{eid}: {info}")


if __name__ == "__main__":
    gguf = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scan(gguf, layer)
