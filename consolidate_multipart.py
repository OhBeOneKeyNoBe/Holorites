"""consolidate_multipart.py — collapse N "-NNNNN-of-00NNN" Holorites into one.

The auto-Holoritify watchdog ran `gguf_holoritify.py` on every .gguf file it
found, which for multi-part GGUFs created N Holorite directories — one per
shard — even though only part 1's metadata + embedding tensor was valid
content. Parts 2..N produced "Holorites" whose manifests still describe
the model but whose `gguf_path` points to a tensor-data-only shard with
no metadata.

This script:
  1. Walks /d/Holorites for any "Holorite-<name>-NNNNN-of-NNNNN" pattern.
  2. Identifies the part-1 of each group (manifest with the proper arch/vocab/hidden).
  3. Renames the part-1 directory to drop the "-NNNNN-of-NNNNN" suffix
     so it becomes the canonical Holorite for the whole model.
  4. Updates the manifest's `gguf_path` to point at the part-1 .gguf
     (node-llama-cpp resolves the rest automatically by sibling pattern).
  5. Deletes the part-2..N directories (their manifests are misleading;
     the embedding torus in them is identical to part-1's anyway).
  6. Adds `multipart_count` + `multipart_pattern` to the canonical
     manifest so consumers know how many .gguf parts to load.

Idempotent: running it twice does nothing on the second run because the
"-NNNNN-of-NNNNN" suffix is already gone.
"""
from __future__ import annotations
import json, os, re, shutil, sys
from pathlib import Path

HOLO_ROOT = Path(r"D:\Holorites")
PART_RX = re.compile(r"^(Holorite-.+?)-(\d{5})-of-(\d{5})$")


def main():
    if not HOLO_ROOT.exists():
        print(f"no {HOLO_ROOT} — nothing to do")
        return 0

    groups: dict[str, dict] = {}    # base_name -> {parts: {part_num: dir}, total: int}
    for d in HOLO_ROOT.iterdir():
        if not d.is_dir(): continue
        m = PART_RX.match(d.name)
        if not m: continue
        base = m.group(1)
        part = int(m.group(2))
        total = int(m.group(3))
        if base not in groups:
            groups[base] = {"parts": {}, "total": total, "pattern": ""}
        groups[base]["parts"][part] = d
        groups[base]["pattern"] = f"{m.group(2)}-of-{m.group(3)}"

    if not groups:
        print("no multi-part Holorite directories found — nothing to do")
        return 0

    print(f"found {len(groups)} multi-part Holorite group(s):")
    for base, info in groups.items():
        n_parts = len(info["parts"])
        print(f"  {base}: {n_parts} of {info['total']} parts present")

    for base, info in groups.items():
        parts = info["parts"]
        total = info["total"]
        canonical_target = HOLO_ROOT / base
        if canonical_target.exists():
            print(f"  {base}: canonical already exists; just removing partN dirs")
        elif 1 not in parts:
            print(f"  {base}: SKIP — no part 1 present (part 1 has the metadata)")
            continue
        else:
            # rename part-1 to canonical name
            src = parts[1]
            print(f"  {base}: renaming {src.name} -> {canonical_target.name}")
            os.rename(src, canonical_target)
            parts.pop(1)

        # update the canonical manifest
        manifest_path = canonical_target / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f: m = json.load(f)
            m["name"] = base.replace("Holorite-", "")
            # the gguf_path in the manifest will point at part-1's .gguf —
            # update it to the renamed canonical path if we moved it, but
            # actually part-1's .gguf path is still on E:, so keep it.
            m["multipart_count"] = total
            m["multipart_pattern"] = f"-NNNNN-of-{total:05d}.gguf"
            # node-llama-cpp finds siblings via filename pattern; we just
            # need to make sure the manifest points at part 1.
            gp = m.get("gguf_path", "")
            if not re.search(r"-00001-of-\d{5}\.gguf$", gp) and total > 1:
                # try to fix the path to point at part 1
                fixed = re.sub(r"-(\d{5})-of-(\d{5})\.gguf$", r"-00001-of-\2.gguf", gp)
                if fixed != gp:
                    m["gguf_path"] = fixed
                    print(f"    fixed gguf_path -> part 1")
            with open(manifest_path, "w", encoding="utf-8") as f: json.dump(m, f, indent=2)

        # delete the other part directories
        for part_num, part_dir in parts.items():
            if part_num == 1: continue
            print(f"    removing {part_dir.name}")
            try:
                shutil.rmtree(part_dir)
            except Exception as e:
                print(f"      failed: {e}")

    print()
    print("=== final Holorites in /d/Holorites/ (sorted) ===")
    for d in sorted(HOLO_ROOT.iterdir()):
        if not d.is_dir(): continue
        if not d.name.startswith("Holorite-"): continue
        mp = d / "manifest.json"
        if not mp.exists(): continue
        with open(mp, encoding="utf-8") as f: m = json.load(f)
        flag = f" ({m['multipart_count']}-part)" if m.get("multipart_count") else ""
        print(f"  {d.name}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
