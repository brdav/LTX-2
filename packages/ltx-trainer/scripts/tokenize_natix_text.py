#!/usr/bin/env python3
"""Encode the Natix route captions (``osm_bev/text_predicted``) into LTX text embeddings.

``<text-root>/<country>/<date>/<session>.zip`` holds ``index.jsonl`` and ``text.jsonl`` (``name`` = ``<HH-MM-SS>_sNNNN.webp``,
``sample_id`` = ``<source_video_path>#f<start_frame>``, ``caption``), one caption per 5-second sample, so captions join the
video clips like the OSM maps (on ``(source_video_path, start_frame)``).

Embeddings are what ``process_captions.py`` stores: Gemma hidden states -> ``embeddings_processor.feature_extractor``
(``video_prompt_embeds``, ``[1024, 4096]`` with left padding). Only the real tokens are stored (``mask == 1``, ~0.1k of 1024),
so ``video_prompt_embeds`` is ``[n_tokens, 4096]`` and ``prompt_attention_mask`` all ones; to recover the padded layout,
left-pad with zeros to 1024 and set the mask to 0 there.

Output mirrors ``tokenize_natix_osm.py``: one safetensors per session, ``<output-dir>/embeddings/<country>/<date>__<session>.safetensors``
with keys ``<HH-MM-SS>_sNNNN/predicted/{video_prompt_embeds,prompt_attention_mask}``, plus index lines in
``<output-dir>/index/shard<N>.jsonl``. Sessions are sharded by the md5 of the zip path; run one process per GPU.
Resumable: a session whose file exists with matching ``n_captions`` is skipped (its index rows are backfilled if missing).
"""

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_trainer import logger
from ltx_trainer.model_loader import embedding_weight_paths, load_embeddings_processor, load_text_encoder

VIEW = "predicted"


def out_relpath(zip_rel: str) -> Path:
    """``<country>/<a>/<b>.zip`` -> ``embeddings/<country>/<a>__<b>.safetensors``."""
    parts = Path(zip_rel).with_suffix("").parts
    return Path("embeddings") / parts[0] / ("__".join(parts[1:]) + ".safetensors")


def read_session(zip_path: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        rows = [json.loads(line) for line in zf.read("text.jsonl").decode().splitlines() if line.strip()]
    keys = [Path(r["name"]).stem for r in rows]
    assert len(set(keys)) == len(keys), f"duplicate names in {zip_path}"
    return [
        {
            "stem": k,
            "source_video_path": r["sample_id"].split("#f")[0],
            "start_frame": int(r["sample_id"].split("#f")[1]),
            "caption": r["caption"].strip(),
        }
        for k, r in zip(keys, rows, strict=True)
    ]


def index_rows(rows: list[dict], rel: str) -> list[dict]:
    return [
        {
            "source_video_path": r["source_video_path"],
            "start_frame": r["start_frame"],
            "view": VIEW,
            "file": rel,
            "embedding_key": f"{r['stem']}/{VIEW}/video_prompt_embeds",
            "mask_key": f"{r['stem']}/{VIEW}/prompt_attention_mask",
        }
        for r in rows
    ]


@torch.inference_mode()
def encode(captions: list[str], text_encoder, processor, batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for b in range(0, len(captions), batch):
        enc = text_encoder.encode(captions[b : b + batch])
        hidden = tuple(torch.cat([e[0][layer] for e in enc]) for layer in range(len(enc[0][0])))
        mask = torch.cat([e[1] for e in enc])
        embeds, _ = processor.feature_extractor(hidden, mask, "left")
        for i in range(len(enc)):
            keep = mask[i].bool()
            out.append((embeds[i][keep].cpu().contiguous(), torch.ones(int(keep.sum()), dtype=torch.int64)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-root", required=True, help="Directory holding <country>/<date>/<session>.zip")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", required=True, help="LTX transformer checkpoint (for the embeddings processor)")
    ap.add_argument("--text-encoder-path", required=True)
    ap.add_argument("--encode-batch", type=int, default=16, help="Captions per Gemma forward")
    ap.add_argument("--num-sessions", type=int, default=None, help="Only the first N sessions (testing)")
    ap.add_argument("--country", default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    text_root, out_dir = Path(args.text_root), Path(args.output_dir)
    zips = sorted(str(p.relative_to(text_root)) for p in text_root.glob("*/*/*.zip"))
    if args.country:
        zips = [z for z in zips if z.split("/")[0] == args.country]
    if args.num_sessions:
        zips = zips[: args.num_sessions]
    zips = [z for z in zips if int(hashlib.md5(z.encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    logger.info(f"Shard {args.shard_index}/{args.num_shards}: {len(zips)} sessions")

    index_file = out_dir / "index" / f"shard{args.shard_index}.jsonl"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    indexed: set[str] = set()
    if index_file.is_file():
        with index_file.open("rb+") as f:  # cut a partial last line left by a crash
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)
                data = data[: data.rfind(b"\n") + 1]
        indexed = {json.loads(line)["file"] for line in data.decode().splitlines()}

    device = torch.device("cuda")
    text_encoder = load_text_encoder(args.text_encoder_path, device=device, dtype=torch.bfloat16)
    processor = load_embeddings_processor(
        embedding_weight_paths(args.model_path, args.text_encoder_path),
        gemma_model_path=args.text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
    )

    fail_log = out_dir / f"failures_shard{args.shard_index}.jsonl"
    done = skipped = 0
    t0 = time.time()
    for si, z in enumerate(zips):
        out = out_dir / out_relpath(z)
        rel = str(out_relpath(z))
        try:
            rows = read_session(text_root / z)
            for stale in out.parent.glob(f"{out.name}.tmp.*"):
                stale.unlink()
            if out.is_file():
                with safe_open(str(out), framework="pt", device="cpu") as f:
                    n_done = int((f.metadata() or {}).get("n_captions", -1))
                if n_done == len(rows):
                    skipped += len(rows)
                    if rel not in indexed:
                        with index_file.open("a") as f:
                            f.writelines(json.dumps(r) + "\n" for r in index_rows(rows, rel))
                    continue
            enc = encode([r["caption"] for r in rows], text_encoder, processor, args.encode_batch)
            tensors = {}
            for r, (emb, mask) in zip(rows, enc, strict=True):
                tensors[f"{r['stem']}/{VIEW}/video_prompt_embeds"] = emb
                tensors[f"{r['stem']}/{VIEW}/prompt_attention_mask"] = mask
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
            save_file(tensors, str(tmp), metadata={"n_captions": str(len(rows)), "caption_kind": "route_text_predicted"})
            tmp.replace(out)
            with index_file.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in index_rows(rows, rel))
            done += len(rows)
        except Exception as e:
            logger.error(f"{z}: {e}")
            with fail_log.open("a") as f:
                f.write(json.dumps({"zip": z, "error": str(e)[:500]}) + "\n")
        if (si + 1) % 20 == 0 or si + 1 == len(zips):
            logger.info(f"{si + 1}/{len(zips)} sessions, {done} captions ({done / max(time.time() - t0, 1e-9):.1f}/s)")
    logger.info(f"Done: {done} encoded, {skipped} already existed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
