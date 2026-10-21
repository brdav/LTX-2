#!/usr/bin/env python3
"""Rebuild the clip index of a ``tokenize_natix.py`` output directory from the packed safetensors files.

Reads only the safetensors headers (tensor names), so it is cheap. Writes ``<output-dir>/index/shard<N>.jsonl`` for
``N in range(--num-shards)``, assigning each source video to shard ``md5(source_video_path) % num_shards`` exactly like
``tokenize_natix.py --shard-index/--num-shards`` does, so the result looks like a single uninterrupted run. The old
index is optionally copied to ``--backup-dir`` first and then replaced; every row of the old index must be present in
the rebuilt one. Run it only when no tokenization job is writing to ``<output-dir>``.

Usage: rebuild_natix_index.py <output-dir> --num-shards 64 [--manifest samples_24fps.parquet] [--backup-dir DIR]
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from safetensors import safe_open


def unflatten(rel: Path) -> str:
    """Inverse of ``tokenize_natix.flat_relpath``: latents/<country>/<a>__<b>__<name>.safetensors -> <country>/<a>/<b>/<name>.mp4"""
    country, name = rel.parts[1], rel.name.removesuffix(".safetensors")
    return "/".join([country, *name.split("__")]) + ".mp4"


def rows_for(path: Path, root: Path) -> list[dict]:
    rel = path.relative_to(root)
    vpath = unflatten(rel)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return [
        {"source_video_path": vpath, "start_frame": int(k.split("/")[0]), "view": k.split("/")[1], "file": str(rel), "key": k}
        for k in keys
    ]


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--num-shards", type=int, required=True, help="Total shards of the tokenization run (4 x array size)")
    ap.add_argument("--manifest", default=None, help="If given, check every clip against the manifest")
    ap.add_argument("--backup-dir", default=None, help="Copy the old index here before replacing it")
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.output_dir).resolve()
    index_dir = root / "index"
    files = sorted((root / "latents").rglob("*.safetensors"))
    log(f"{len(files)} packed files; reading headers with {args.threads} threads")
    rows: list[dict] = []
    with ThreadPoolExecutor(args.threads) as ex:
        for i, part in enumerate(ex.map(lambda p: rows_for(p, root), files), 1):
            rows.extend(part)
            if i % 10000 == 0:
                log(f"  {i}/{len(files)} files")
    keys = {(r["file"], r["key"]) for r in rows}
    assert len(keys) == len(rows), "duplicate (file, key) in packed files"
    log(f"{len(rows)} clip rows")

    if args.manifest:
        import pandas as pd

        m = pd.read_parquet(args.manifest, columns=["source_video_path", "start_frame"])
        expected = set(zip(m.source_video_path, m.start_frame.astype(int)))
        have = {(r["source_video_path"], r["start_frame"]) for r in rows if r["view"] == "front"}
        log(f"manifest clips {len(expected)}; in output {len(have)}; not in manifest {len(have - expected)}; "
            f"missing from output {len(expected - have)}")

    old_rows = 0
    for f in index_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            old_rows += 1
            if (r["file"], r["key"]) not in keys:
                sys.exit(f"old index row not found in packed files: {r} (in {f}); not touching the index")
    log(f"old index: {old_rows} rows (all present in the packed files)")

    by_shard = defaultdict(list)
    for r in rows:
        by_shard[int(hashlib.md5(r["source_video_path"].encode()).hexdigest(), 16) % args.num_shards].append(r)
    new_dir = root / "index.new"
    shutil.rmtree(new_dir, ignore_errors=True)
    new_dir.mkdir()
    for s in range(args.num_shards):
        rs = sorted(by_shard[s], key=lambda r: (r["source_video_path"], r["start_frame"], r["view"]))
        (new_dir / f"shard{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))

    if args.backup_dir and index_dir.exists():
        shutil.rmtree(args.backup_dir, ignore_errors=True)
        shutil.copytree(index_dir, args.backup_dir)
        log(f"old index backed up to {args.backup_dir}")
    shutil.rmtree(index_dir, ignore_errors=True)
    new_dir.rename(index_dir)
    log(f"wrote {len(rows)} rows into {args.num_shards} shard files in {index_dir}")


if __name__ == "__main__":
    main()
